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
from .hybridv2 import GMUAttention   # updated: use GMUAttention from hybridv2 to match training
from .time_utils import generate_time_dna, TimeDNAEncoder, encode_time_dna_batch_cnn
from .content_utils import generate_content_dna, encode_content_dna_batch_cnn  # updated for consistency with training
from .emoji_utils import generate_emoji_dna, encode_emoji_dna_batch_cnn           # added emoji utilities
# -------------------------------------------------

# -------------------------
# Set Seed for Reproducibility
# -------------------------
torch.manual_seed(42)
np.random.seed(42)

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
        model_path = hf_hub_download(repo_id="hzarashid/ForensiX", filename="hybridv2_weights.bin")
        self.model = GMUAttention()  # updated: use GMUAttention as in training
        self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')), strict=False)
        self.model.eval()
        
    def detect_bot(self, session_data):
        """
        For each user in the session, compute the feature vectors from four modalities:
         - Description embedding (768-dim)
         - Content DNA embedding (384-dim, via CNN)
         - Time DNA embedding (384-dim)
         - Emoji DNA embedding (e.g., 384-dim)
        Then, run these features through the trained model to predict the bot probability.
        Returns a list of DetectionMark objects.
        """
        # Group posts by user_id.
        user_posts = {}
        for post in session_data.posts:
            uid = post.get("user_id") or post.get("author_id")
            if uid is not None:
                user_posts.setdefault(uid, []).append(post)
        
        marked_accounts = []
        
        for user in session_data.users:
            uid = user.get("id")
            if uid is None or uid not in user_posts:
                continue
            
            # Compute description embedding.
            description = user.get("description", "")
            desc_emb = self.st_model.encode(description)  # 768-dim vector
            
            # Generate content DNA from user's posts and compute its CNN embedding.
            posts = user_posts[uid]
            content_dna = generate_content_dna(posts)
            dna_emb = encode_content_dna_batch_cnn([content_dna])[0]  # 384-dim vector
            
            # Generate time DNA and compute its CNN embedding.
            time_dna = generate_time_dna(posts)
            time_emb = encode_time_dna_batch_cnn([time_dna])[0]  # 384-dim vector
            
            # Generate emoji DNA and compute its CNN embedding.
            emoji_dna = generate_emoji_dna(posts)
            emoji_emb = encode_emoji_dna_batch_cnn([emoji_dna])[0]  # assumed 384-dim vector
            
            # Convert each modality to a tensor and add a batch dimension.
            text_tensor = torch.tensor(desc_emb, dtype=torch.float32).unsqueeze(0)
            dna_tensor = torch.tensor(dna_emb, dtype=torch.float32).unsqueeze(0)
            time_tensor = torch.tensor(time_emb, dtype=torch.float32).unsqueeze(0)
            emoji_tensor = torch.tensor(emoji_emb, dtype=torch.float32).unsqueeze(0)
            
            # Run through the trained model.
            with torch.no_grad():
                logits = self.model(text_tensor, dna_tensor, time_tensor, emoji_tensor)
                probabilities = torch.softmax(logits, dim=1)
                pred_class = probabilities.argmax(dim=1).item()
                confidence = int(probabilities[0, 1].item() * 100)
                is_bot = (pred_class == 1)  # Assuming class '1' represents bots.
            
            marked_accounts.append(DetectionMark(user_id=uid, confidence=confidence, bot=is_bot))
        
        print(marked_accounts)
        return marked_accounts
