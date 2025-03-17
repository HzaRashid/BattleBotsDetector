import os
import re
import json
import ijson
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
from datetime import datetime
from sentence_transformers import SentenceTransformer, models
from sklearn.metrics import classification_report

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
    # Sort tweets by created_at timestamp.
    tweets.sort(key=lambda x: datetime.fromisoformat(x["created_at"].replace("Z", "+00:00")))
    dna = "".join(get_content_dna_symbol(tweet["text"]) for tweet in tweets)
    return dna

# -------------------------
# Token Map and DNA Parameters
# -------------------------
token_map = {"N": 0, "U": 1, "H": 2, "M": 3, "X": 4, "R": 5, "E": 6}
vocab_size = len(token_map)
max_dna_len = 200  # maximum sequence length for our DNA tokens

# -------------------------
# Transformer-based Description Embedding Model
# -------------------------
transformer_model = models.Transformer("Twitter/twhin-bert-base", model_args={'attn_implementation': 'eager'})
pooling_model = models.Pooling(transformer_model.get_word_embedding_dimension(), pooling_mode_mean_tokens=True)
st_model = SentenceTransformer(modules=[transformer_model, pooling_model])

# -------------------------
# Simple, Randomly Initialized Transformer for DNA Encoding
# -------------------------
class SimpleDNATransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim=384, num_layers=1, num_heads=2, hidden_dim=256, max_seq_len=200):
        super(SimpleDNATransformer, self).__init__()
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.positional_embedding = nn.Embedding(max_seq_len, embed_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dim_feedforward=hidden_dim, 
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
    
    def forward(self, x):
        batch_size, seq_len = x.size()
        positions = torch.arange(0, seq_len, device=x.device).unsqueeze(0).expand(batch_size, seq_len)
        x = self.token_embedding(x) + self.positional_embedding(positions)
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)
        return x

# Instantiate DNA encoder model and set to evaluation mode.
dna_encoder_model = SimpleDNATransformer(vocab_size=vocab_size, embed_dim=384, num_layers=1, num_heads=2, hidden_dim=256, max_seq_len=max_dna_len)
dna_encoder_model.eval()

def tokenize_dna(dna_string, token_map, max_seq_len):
    tokens = [ch for ch in dna_string if ch in token_map]
    token_ids = [token_map[ch] for ch in tokens]
    if len(token_ids) < max_seq_len:
        token_ids = token_ids + [token_map["N"]] * (max_seq_len - len(token_ids))
    else:
        token_ids = token_ids[:max_seq_len]
    return token_ids

def encode_dna_batch(dna_sequences, max_length=200):
    tokenized = [tokenize_dna(seq, token_map, max_length) for seq in dna_sequences]
    input_tensor = torch.tensor(tokenized, dtype=torch.long)
    with torch.no_grad():
        embeddings = dna_encoder_model(input_tensor)
    return embeddings.numpy()

# -------------------------
# Data Loading Function for New Data
# -------------------------
def load_new_data(file_path):
    """
    Load tweet data from a JSON file.
    Assumes the file has two top-level keys: "users" and "posts".
    Returns:
      - user_info_list: list of user dictionaries.
      - user_posts_dict: dictionary mapping user_id to list of post dictionaries.
    """
    user_info_list = []
    user_posts_dict = {}
    with open(file_path, "r") as f:
        for user in ijson.items(f, "users.item"):
            user_info_list.append(user)
    with open(file_path, "r") as f:
        for post in ijson.items(f, "posts.item"):
            uid = post.get("user_id") or post.get("author_id")
            if uid is not None:
                user_posts_dict.setdefault(uid, []).append(post)
    return user_info_list, user_posts_dict

# -------------------------
# Main Prediction Pipeline
# -------------------------
def main():
    # Paths (adjust these as needed)
    cur_dir = os.path.dirname(__file__)
    model_dir = os.path.join(cur_dir, '../models/GBM')
    tweet_data_file = os.path.join(cur_dir, "../data/session_14_results.json")
    model_path = os.path.join(model_dir, "XGBoost.pkl")
    
    # Load new tweet data.
    user_info_list, user_posts_dict = load_new_data(tweet_data_file)
    
    # Load trained classifier pipeline.
    best_pipeline = joblib.load(model_path)
    
    # Containers for embeddings, DNA strings, and user ids.
    user_ids = []
    desc_embeddings = []
    dna_strings = []
    
    # Process each user in the loaded data.
    for user in user_info_list:
        uid = user.get("user_id")
        if uid is None or uid not in user_posts_dict:
            continue
        
        posts = user_posts_dict[uid]
        dna = generate_content_dna(posts)
        
        # Compute description embedding.
        description = user.get("description", "")
        desc_emb = st_model.encode(description)
        
        user_ids.append(uid)
        desc_embeddings.append(desc_emb)
        dna_strings.append(dna)
    
    if not user_ids:
        print("No valid users found in the dataset.")
        return
    
    # Compute DNA embeddings for all users.
    dna_embeddings = encode_dna_batch(dna_strings, max_length=max_dna_len)
    
    # Concatenate description embeddings and DNA embeddings.
    X = np.array([np.concatenate([desc, dna_emb]) for desc, dna_emb in zip(desc_embeddings, dna_embeddings)])
    
    # Get predictions from the classifier.
    y_pred = best_pipeline.predict(X)
    y_pred_proba = best_pipeline.predict_proba(X)[:, 1]
    
    # If ground truth labels are available in the user info (e.g. "is_bot"), compute and print classification report.
    ground_truth = {}
    for user in user_info_list:
        uid = user.get("user_id")
        if uid is not None and "is_bot" in user:
            ground_truth[uid] = int(user["is_bot"])
    
    if all(uid in ground_truth for uid in user_ids):
        y_true = np.array([ground_truth[uid] for uid in user_ids])
        print("Classification Report:")
        print(classification_report(y_true, y_pred))
    
    # Output predictions.
    # print("User Predictions:")
    # for uid, pred, prob in zip(user_ids, y_pred, y_pred_proba):
    #     print(f"User: {uid}")
    #     print(f"  Prediction (0: human, 1: bot): {pred} (Probability: {prob:.4f})")
    #     print("-" * 40)

if __name__ == "__main__":
    main()
