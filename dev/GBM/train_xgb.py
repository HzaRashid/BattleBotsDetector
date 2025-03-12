import os
import re
import json
import ijson
import numpy as np
from datetime import datetime
import torch
import torch.nn as nn
import polars as pl
import pandas as pd
from sentence_transformers import SentenceTransformer, models
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

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
# Load Pre-trained Model for Description Embeddings
# -------------------------
# For description embeddings, we continue to use the Twitter model.
transformer_model = models.Transformer("Twitter/twhin-bert-base", model_args={'attn_implementation': 'eager'})
pooling_model = models.Pooling(
    transformer_model.get_word_embedding_dimension(),
    pooling_mode_mean_tokens=True
)
st_model = SentenceTransformer(modules=[transformer_model, pooling_model])

# -------------------------
# Simple, Randomly Initialized Transformer for DNA Encoding
# -------------------------
class SimpleDNATransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim=384, num_layers=1, num_heads=2, hidden_dim=256, max_seq_len=200, dropout=0.1):
        super(SimpleDNATransformer, self).__init__()
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
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
        x = x.mean(dim=1)
        return x

# Instantiate the randomly initialized transformer for DNA encoding
dna_encoder_model = SimpleDNATransformer(vocab_size=vocab_size, embed_dim=384, num_layers=1, num_heads=2, hidden_dim=256, max_seq_len=max_dna_len)
dna_encoder_model.eval()

# -------------------------
# Tokenization for DNA Sequences
# -------------------------
def tokenize_dna(dna_string, token_map, max_seq_len):
    tokens = [ch for ch in dna_string if ch in token_map]
    token_ids = [token_map[ch] for ch in tokens]
    if len(token_ids) < max_seq_len:
        token_ids = token_ids + [token_map["N"]] * (max_seq_len - len(token_ids))
    else:
        token_ids = token_ids[:max_seq_len]
    return token_ids

# -------------------------
# DNA Encoding Function Using the Randomly Initialized Transformer
# -------------------------
def encode_dna_batch(dna_sequences, max_length=200):
    tokenized = [tokenize_dna(seq, token_map, max_length) for seq in dna_sequences]
    input_tensor = torch.tensor(tokenized, dtype=torch.long)
    with torch.no_grad():
        embeddings = dna_encoder_model(input_tensor)
    return embeddings.numpy()

# -------------------------
# Data Loading Function using ijson and Train-Test Split by Unique User
# -------------------------
def load_data(data_dir, session_numbers=[], st_model=None, xnums=[]):
    user_info_list = []
    user_posts_dict = {}
    datasets = []
    
    if session_numbers:
         datasets += [f"session_{num}_results.json" for num in session_numbers]
    if xnums:
         datasets += [f"twibot22/processed/tweet_{num}_processed.json" for num in xnums]

    for fname in datasets:
        json_file = os.path.join(data_dir, fname)
        with open(json_file, "r") as f:
            for user in ijson.items(f, "users.item"):
                user_info_list.append(user)
        with open(json_file, "r") as f:
            for post in ijson.items(f, "posts.item"):
                uid = post.get("user_id") or post.get("author_id")
                if uid is not None:
                    user_posts_dict.setdefault(uid, []).append(post)

    user_info_df = pd.DataFrame(user_info_list)[['user_id', 'is_bot']].drop_duplicates()
    train_users, test_users = train_test_split(
        user_info_df, test_size=0.2, random_state=42, stratify=user_info_df['is_bot']
    )
    train_user_ids = set(train_users['user_id'])
    test_user_ids = set(test_users['user_id'])
    
    train_descs, train_dna_list, train_labels = [], [], []
    test_descs, test_dna_list, test_labels = [], [], []
    
    for user in user_info_list:
        uid = user.get("user_id")
        if uid is None or uid not in user_posts_dict:
            continue
        description = user.get("description", "")
        # Compute description embedding using st_model
        desc_emb = st_model.encode(description)
        # Generate the DNA string from the user's posts.
        dna = generate_content_dna(user_posts_dict[uid])
        label = int(user.get("is_bot", 0))
        if uid in train_user_ids:
            train_descs.append(desc_emb)
            train_dna_list.append(dna)
            train_labels.append(label)
        elif uid in test_user_ids:
            test_descs.append(desc_emb)
            test_dna_list.append(dna)
            test_labels.append(label)
    
    # Get DNA embeddings using the randomly initialized transformer
    train_dna_embs = encode_dna_batch(train_dna_list)
    test_dna_embs = encode_dna_batch(test_dna_list)
    
    # Concatenate the description embeddings and DNA embeddings.
    train_X = [np.concatenate([desc, dna_emb]) for desc, dna_emb in zip(train_descs, train_dna_embs)]
    test_X = [np.concatenate([desc, dna_emb]) for desc, dna_emb in zip(test_descs, test_dna_embs)]
    
    return np.array(train_X), np.array(test_X), np.array(train_labels), np.array(test_labels)

# -------------------------
# Main Pipeline with Optuna Hyperparameter Tuning Using Expanded Hyperparameters
# -------------------------
if __name__ == "__main__":
    cur_dir = os.path.dirname(__file__)
    data_dir = os.path.join(cur_dir, "../data")

    # Load data using our custom function
    X_train, X_test, y_train, y_test = load_data(
        data_dir, 
        session_numbers=[12], 
        st_model=st_model, 
        xnums=[]
    )
    
    # Compute scale_pos_weight for class imbalance
    neg, pos = np.bincount(y_train)
    scale_pos_weight = neg / pos

    import optuna
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from xgboost import XGBClassifier
    from sklearn.metrics import f1_score, make_scorer, average_precision_score, roc_auc_score, accuracy_score, classification_report

    def objective(trial):
        # Expanded hyperparameter search
        n_estimators = trial.suggest_int("n_estimators", 300, 1000)
        learning_rate = trial.suggest_loguniform("learning_rate", 0.001, 0.1)
        max_depth = trial.suggest_int("max_depth", 6, 10)
        gamma = trial.suggest_loguniform("gamma", 1e-8, 1.0)
        min_child_weight = trial.suggest_int("min_child_weight", 1, 10)
        
        # Use F1 score for evaluation
        scoring_metric = make_scorer(f1_score, average="binary")
        
        clf = XGBClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            gamma=gamma,
            min_child_weight=min_child_weight,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            use_label_encoder=False,
            eval_metric="auc",  # For internal training monitoring; tuning based on F1
            objective='binary:logistic'
        )
        
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(clf, X_train, y_train, cv=cv, scoring=scoring_metric, n_jobs=-1)
        return np.mean(scores)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50)

    print("Best trial:")
    trial = study.best_trial
    print("  Mean score:", trial.value)
    for key, value in trial.params.items():
        print(f"  {key}: {value}")

    # Build and fit the final classifier using the best hyperparameters.
    best_clf = XGBClassifier(
        n_estimators=trial.params["n_estimators"],
        learning_rate=trial.params["learning_rate"],
        max_depth=trial.params["max_depth"],
        gamma=trial.params["gamma"],
        min_child_weight=trial.params["min_child_weight"],
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        use_label_encoder=False,
        eval_metric="auc",  # For internal training; tuning was based on F1
        objective='binary:logistic'
    )
    best_clf.fit(X_train, y_train)

    y_pred = best_clf.predict(X_test)
    y_pred_proba = best_clf.predict_proba(X_test)[:, 1]

    print("Test Accuracy: {:.4f}".format(accuracy_score(y_test, y_pred)))
    print("Test ROC AUC: {:.4f}".format(roc_auc_score(y_test, y_pred_proba)))
    print("Test AUPR: {:.4f}".format(average_precision_score(y_test, y_pred_proba)))
    print("Classification Report:\n", classification_report(y_test, y_pred))
