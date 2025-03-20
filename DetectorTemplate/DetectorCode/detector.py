
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
from hybrid import NewMultiModalAttentionFusion # Import custom model
# print(NewMultiModalAttentionFusion())
from time_utils import generate_time_dna, TimeDNAEncoder, encode_time_dna_batch_cnn
from content_utils import generate_content_dna, DNACNNEncoder, encode_dna_batch_cnn
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
# Instantiate Time DNA Encoder (from training)
# -------------------------
time_dna_encoder = TimeDNAEncoder(output_dim=384)
time_dna_encoder.eval()

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
        # Replace 'your-username/model-repo' with your actual repository identifier.
        model_path = hf_hub_download(repo_id="hzarashid/ForensiX", filename="pytorch_model.bin")
        self.model = NewMultiModalAttentionFusion()  # Ensure same initialization parameters as in training if needed.
        self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        self.model.eval()
        
    def detect_bot(self, session_data):
        """
        For each user in the session, compute the feature vector by concatenating:
         - description embedding (768-dim)
         - content DNA embedding (384-dim, via CNN)
         - time DNA embedding (384-dim)
        Then, run the feature vector through the trained model to predict bot probability.
        Returns a list of DetectionMark objects.
        """
        # Group posts by user_id.
        user_posts = {}
        for post in session_data.posts:
            uid = post.get("user_id") or post.get("author_id")
            if uid is not None:
                user_posts.setdefault(uid, []).append(post)
        
        user_ids = []
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
            dna_emb = encode_dna_batch_cnn([content_dna])[0]  # 384-dim vector
            
            # Generate time DNA and compute its embedding.
            time_dna = generate_time_dna(posts)
            time_emb = encode_time_dna_batch_cnn([time_dna], time_dna_encoder=time_dna_encoder)[0]  # 384-dim vector
            
            # Concatenate all three modalities (total 768+384+384 = 1536 dimensions).
            feature_vector = np.concatenate([desc_emb, dna_emb, time_emb])
            user_ids.append(uid)
            
            # Convert feature vector to tensor and run through the trained model.
            input_tensor = torch.tensor(feature_vector, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                logits = self.model(input_tensor)
                probabilities = torch.softmax(logits, dim=1)
                pred_class = probabilities.argmax(dim=1).item()
                confidence = int(probabilities[0, 1].item() * 100)
                is_bot = (pred_class == 1)  # Assuming class '1' represents bots.
            
            marked_accounts.append(DetectionMark(user_id=uid, confidence=confidence, bot=is_bot))
        
        return marked_accounts