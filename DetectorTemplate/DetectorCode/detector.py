import re
import ijson
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datetime import datetime
from PIL import Image
from torchvision import transforms, models as tv_models
from sentence_transformers import SentenceTransformer, models
from huggingface_hub import hf_hub_download

# ------------* custom model and utils *------------
from .hybridv3 import MOEAttention   # updated: use GMUAttention from hybridv2 to match training
from .time_utils import generate_time_dna, encode_time_dna_batch_cnn
from .content_utils import generate_content_dna, encode_content_dna_batch_cnn  # updated for consistency with training
from .tweet_embs import sample_evenly_tweets, compute_mean_embeddings  # new tweet embedding utilities

# -------------------------
# Set Seed for Reproducibility
# -------------------------
torch.manual_seed(42)
np.random.seed(42)

BATCH_SIZE = 64

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
        self.model = MOEAttention()  # updated: use GMUAttention as in training
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
        Returns a list of DetectionMark objects.
        """
        # Group posts by user_id.
        user_posts = {}
        for post in session_data.posts:
            uid = post.get("user_id") or post.get("author_id")
            if uid is not None:
                user_posts.setdefault(uid, []).append(post)
        
        # Prepare lists for batch processing.
        user_ids = []
        desc_texts = []
        tweet_texts_list = []  # each entry is a list of sampled tweet texts for the user
        content_dna_list = []
        time_dna_list = []
        
        for user in session_data.users:
            uid = user.get("id")
            if uid is None or uid not in user_posts:
                continue
            user_ids.append(uid)
            description = user.get("description", "")
            desc_texts.append(description)
            
            posts = user_posts[uid]
            # Sort posts by creation time.
            posts_sorted = sorted(posts, key=lambda x: datetime.fromisoformat(x["created_at"].replace("Z", "+00:00")))
            texts = [post.get("text", "") for post in posts_sorted]
            # Sample tweets evenly (e.g., 5 tweets per user).
            sampled_tweets = sample_evenly_tweets(texts, n=5)
            tweet_texts_list.append(sampled_tweets)
            
            # Generate content and time DNA sequences.
            content_dna = generate_content_dna(posts)
            time_dna = generate_time_dna(posts)
            content_dna_list.append(content_dna)
            time_dna_list.append(time_dna)
        
        # Batch encode the modalities.
        desc_embs = self.st_model.encode(desc_texts, batch_size=BATCH_SIZE)
        tweet_embs = compute_mean_embeddings(tweet_texts_list, self.st_model, batch_size=BATCH_SIZE)
        dna_embs = encode_content_dna_batch_cnn(content_dna_list)
        time_embs = encode_time_dna_batch_cnn(time_dna_list)
        
        # Convert embeddings to tensors.
        desc_tensor = torch.tensor(desc_embs, dtype=torch.float32)
        tweet_tensor = torch.tensor(tweet_embs, dtype=torch.float32)
        dna_tensor = torch.tensor(dna_embs, dtype=torch.float32)
        time_tensor = torch.tensor(time_embs, dtype=torch.float32)
        
        # Run through the trained model in batch.
        with torch.no_grad():
            logits, loss = self.model(desc_tensor, tweet_tensor, dna_tensor, time_tensor)
            probabilities = torch.softmax(logits, dim=1)
            pred_classes = probabilities.argmax(dim=1).tolist()
            confidences = (probabilities[:, 1] * 100).tolist()  # assuming class '1' represents bots
        
        marked_accounts = []
        for uid, pred, conf in zip(user_ids, pred_classes, confidences):
            is_bot = (pred == 1)
            marked_accounts.append(DetectionMark(user_id=uid, confidence=int(conf), bot=is_bot))
        
        print(marked_accounts)
        return marked_accounts
