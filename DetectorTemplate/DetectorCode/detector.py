import re
import ijson
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datetime import datetime
from sentence_transformers import SentenceTransformer, models
from huggingface_hub import hf_hub_download

# ------------* custom model and utils *------------
from .hybridv3 import MOEAttention   # updated: use MOEAttention for this version
from .time_utils import generate_time_dna, encode_time_dna_batch_cnn
from .content_utils import generate_content_dna, encode_content_dna_batch_cnn  # updated for consistency with training
from .tweet_embs import sample_evenly_tweets, compute_mean_embeddings  # new tweet embedding utilities

# -------------------------
# Set Seed for Reproducibility
# -------------------------
torch.manual_seed(42)
np.random.seed(42)

BATCH_SIZE = 32  # batch size is only used for encoding individual user texts

# -------------------------
# Load Description Embedding Model (SentenceTransformer)
# -------------------------
transformer_model = models.Transformer("Twitter/twhin-bert-base", model_args={'attn_implementation': 'eager'})
pooling_model = models.Pooling(transformer_model.get_word_embedding_dimension(), pooling_mode_mean_tokens=True)
st_model = SentenceTransformer(modules=[transformer_model, pooling_model])

# -------------------------
# Detector Class Using the Trained PyTorch Model
# -------------------------
from abc_classes import ADetector
from teams_classes import DetectionMark

class Detector(ADetector):
    def __init__(self):
        # Use the same description embedding model.
        self.st_model = st_model

        # Load the trained PyTorch model from Hugging Face.
        model_path = hf_hub_download(repo_id="hzarashid/ForensiX", filename="hybridv3_weights.bin")
        self.model = MOEAttention()  # updated: use MOEAttention as in training
        self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')), strict=False)
        self.model.eval()
        
    def detect_bot(self, session_data):
        """
        For each user in the session, compute the feature vectors from four modalities:
         - Description embedding (768-dim)
         - Mean tweet embedding (via compute_mean_embeddings)
         - Content DNA embedding (384-dim, via CNN)
         - Time DNA embedding (384-dim)
        Then, run these features through the trained model to predict the bot probability.
        Processes one user at a time to reduce memory usage.
        Returns a list of DetectionMark objects.
        """
        # Group posts by user_id.
        user_posts = {}
        for post in session_data.posts:
            uid = post.get("user_id") or post.get("author_id")
            if uid is not None:
                user_posts.setdefault(uid, []).append(post)
                
        marked_accounts = []
        
        # Process each user individually to avoid memory issues
        for user in session_data.users:
            uid = user.get("id")
            if uid is None or uid not in user_posts:
                continue

            # Description embedding.
            description = user.get("description", "")
            # Encode description (single sample encoding: returns a list, extract first element)
            desc_emb = self.st_model.encode([description], batch_size=1)[0]
            
            posts = user_posts[uid]
            # Sort posts by creation time.
            posts_sorted = sorted(
                posts,
                key=lambda x: datetime.fromisoformat(x["created_at"].replace("Z", "+00:00"))
            )
            texts = [post.get("text", "") for post in posts_sorted]
            # Sample tweets evenly (e.g., 5 tweets per user).
            sampled_tweets = sample_evenly_tweets(texts, n=5)
            # Compute mean tweet embedding (again, returns a list)
            tweet_emb = compute_mean_embeddings([sampled_tweets], self.st_model, batch_size=1)[0]
            
            # Generate DNA sequences for content and time modalities.
            content_dna = generate_content_dna(posts)
            time_dna = generate_time_dna(posts)
            # Encode DNA sequences (each returns a list with one element)
            dna_emb = encode_content_dna_batch_cnn([content_dna])[0]
            time_emb = encode_time_dna_batch_cnn([time_dna])[0]
            
            # Convert each modality to a tensor and add a batch dimension.
            desc_tensor = torch.tensor(desc_emb, dtype=torch.float32).unsqueeze(0)
            tweet_tensor = torch.tensor(tweet_emb, dtype=torch.float32).unsqueeze(0)
            dna_tensor = torch.tensor(dna_emb, dtype=torch.float32).unsqueeze(0)
            time_tensor = torch.tensor(time_emb, dtype=torch.float32).unsqueeze(0)
            
            # Run through the trained model for the current user.
            with torch.no_grad():
                logits, _ = self.model(desc_tensor, tweet_tensor, dna_tensor, time_tensor)
                probabilities = torch.softmax(logits, dim=1)
                pred_class = probabilities.argmax(dim=1).item()
                confidence = int(probabilities[0, 1].item() * 100)
                is_bot = (pred_class == 1)  # assuming class '1' represents bots
            
            marked_accounts.append(DetectionMark(user_id=uid, confidence=confidence, bot=is_bot))
        
        print(marked_accounts)
        return marked_accounts
