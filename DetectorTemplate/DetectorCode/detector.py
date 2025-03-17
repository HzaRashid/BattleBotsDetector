from abc_classes import ADetector
from teams_classes import DetectionMark
import os
import re
import ijson
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
from datetime import datetime
from sentence_transformers import SentenceTransformer, models

# -------------------------
# Set Seed for Reproducibility
# -------------------------
torch.manual_seed(42)
np.random.seed(42)

# -------------------------
# Digital DNA Functions
# -------------------------
emoji_pattern = re.compile("[" 
    u"\U0001F600-\U0001F64F"  # emoticons
    u"\U0001F300-\U0001F5FF"  # symbols & pictographs
    u"\U0001F680-\U0001F6FF"  # transport & map symbols
    u"\U0001F1E0-\U0001F1FF"  # flags
    u"\U0001F700-\U0001F77F"  # alchemical symbols
    u"\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
    u"\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
    u"\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
    u"\U0001FA00-\U0001FA6F"  # Chess Symbols, etc.
    u"\U0001FBA0-\U0001FBAF"  # (optional additional range)
    u"\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
    "]+", flags=re.UNICODE)

def get_content_dna_symbol(tweet_text):
    url_present = bool(re.search(r"https?://\S+", tweet_text))
    hashtag_present = bool(re.search(r"#\w+", tweet_text))
    mention_present = bool(re.search(r"@\w+", tweet_text))
    
    if tweet_text.strip().startswith("@"):
        return "R"
    
    emoji_present = bool(emoji_pattern.search(tweet_text))
    entity_types = sum([url_present, hashtag_present, mention_present, emoji_present])
    
    if entity_types == 0:
        return "N"
    elif entity_types == 1:
        if url_present:
            return "U"
        elif hashtag_present:
            return "H"
        elif mention_present:
            return "M"
        elif emoji_present:
            return "E"
    else:
        return "X"

def generate_content_dna(tweets):
    # Sort tweets by timestamp (assumes ISO format with a trailing "Z")
    tweets.sort(key=lambda x: datetime.fromisoformat(x["created_at"].replace("Z", "+00:00")))
    return "".join(get_content_dna_symbol(tweet["text"]) for tweet in tweets)

# -------------------------
# Token Map and DNA Encoding Parameters
# -------------------------
token_map = {"N": 0, "U": 1, "H": 2, "M": 3, "X": 4, "R": 5, "E": 6}
max_dna_len = 200

def tokenize_dna(dna_string, token_map, max_seq_len):
    tokens = [ch for ch in dna_string if ch in token_map]
    token_ids = [token_map[ch] for ch in tokens]
    if len(token_ids) < max_seq_len:
        token_ids = token_ids + [token_map["N"]] * (max_seq_len - len(token_ids))
    else:
        token_ids = token_ids[:max_seq_len]
    return token_ids

class SimpleDNATransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim=384, num_layers=1, num_heads=2, hidden_dim=256, max_seq_len=200, dropout=0.1):
        super(SimpleDNATransformer, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.positional_embedding = nn.Embedding(max_seq_len, embed_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
    
    def forward(self, x):
        batch_size, seq_len = x.size()
        positions = torch.arange(0, seq_len, device=x.device).unsqueeze(0).expand(batch_size, seq_len)
        x = self.token_embedding(x) + self.positional_embedding(positions)
        x = self.transformer_encoder(x)
        return x.mean(dim=1)

def encode_dna_batch(dna_sequences, max_length=200):
    tokenized = [tokenize_dna(seq, token_map, max_length) for seq in dna_sequences]
    input_tensor = torch.tensor(tokenized, dtype=torch.long)
    with torch.no_grad():
        embeddings = dna_encoder_model(input_tensor)
    return embeddings.numpy()

# -------------------------
# Load Transformer Models
# -------------------------
# Initialize the description embedding model with your desired transformer and pooling.
transformer_model = models.Transformer("Twitter/twhin-bert-base", model_args={'attn_implementation': 'eager'})
pooling_model = models.Pooling(transformer_model.get_word_embedding_dimension(), pooling_mode_mean_tokens=True)
st_model = SentenceTransformer(modules=[transformer_model, pooling_model])

# Instantiate the DNA encoder.
vocab_size = len(token_map)
dna_encoder_model = SimpleDNATransformer(vocab_size=vocab_size, embed_dim=384, num_layers=1, num_heads=2, hidden_dim=256, max_seq_len=max_dna_len)
dna_encoder_model.eval()

# -------------------------
# Detector Class Using XGBoost
# -------------------------
class Detector(ADetector):
    def __init__(self):
        # Use the same description embedding model.
        self.st_model = st_model
        
        # Load the pre-trained XGBoost classifier pipeline.
        cur_dir = os.path.dirname(__file__)
        model_dir = os.path.join(cur_dir, '../../dev/models/GBM')
        model_path = os.path.join(model_dir, "XGBoost.pkl")
        self.clf = joblib.load(model_path)

    def detect_bot(self, session_data):
        # Group posts by user_id.
        user_posts = {}
        for post in session_data.posts:
            uid = post.get("user_id") or post.get("author_id")
            if uid is not None:
                user_posts.setdefault(uid, []).append(post)
        
        user_ids = []
        features = []
        for user in session_data.users:
            uid = user.get("id")
            if uid is None or uid not in user_posts:
                continue
            
            # Compute description embedding.
            description = user.get("description", "")
            desc_emb = self.st_model.encode(description)
            
            # Generate digital DNA from user's posts and compute DNA embedding.
            posts = user_posts[uid]
            dna = generate_content_dna(posts)
            dna_emb = encode_dna_batch([dna], max_length=max_dna_len)[0]
            
            # Concatenate the description and DNA embeddings.
            feature_vector = np.concatenate([desc_emb, dna_emb])
            user_ids.append(uid)
            features.append(feature_vector)
        
        if not features:
            return []
        
        X = np.vstack(features)
        # Predict probability of being a bot.
        y_pred_proba = self.clf.predict_proba(X)[:, 1]
        
        # Generate DetectionMark for each user.
        marked_accounts = []
        for uid, prob in zip(user_ids, y_pred_proba):
            confidence = int(prob * 100)
            is_bot = prob >= 0.5
            marked_accounts.append(DetectionMark(user_id=uid, confidence=confidence, bot=is_bot))
        return marked_accounts
