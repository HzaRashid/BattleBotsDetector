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
from PIL import Image
from torchvision import transforms, models as tv_models  # Import torchvision models as tv_models
from sentence_transformers import SentenceTransformer, models  # 'models' here is for sentence transformer
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
# CNN-based DNA Encoder
# -------------------------
def dna_to_tensor(dna, 
                  mapping={"N": 0, "U": 64, "H": 128, "M": 192, "X": 255, "R": 32, "E": 32},
                  desired_size=64):
    """
    Converts a DNA string into a grayscale image (then into a normalized RGB tensor).
    """
    values = [mapping[symbol] for symbol in dna if symbol in mapping]
    length = len(values)
    n = int(np.ceil(np.sqrt(length)))
    total = n * n
    values += [mapping["N"]] * (total - length)
    arr = np.array(values, dtype=np.uint8).reshape((n, n))
    img = Image.fromarray(arr, mode="L")
    img = img.resize((desired_size, desired_size), Image.NEAREST)
    img = img.convert("RGB")
    transform = transforms.Compose([
        transforms.ToTensor(),  # scales pixels to [0,1]
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    tensor = transform(img)
    return tensor

class DNACNNEncoder(nn.Module):
    def __init__(self, output_dim=384):
        super(DNACNNEncoder, self).__init__()
        # Use a pretrained MobileNetV2 from torchvision (tv_models)
        self.cnn = tv_models.mobilenet_v2(pretrained=True)
        self.features = self.cnn.features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1280, output_dim)
        
    def forward(self, x):
        with torch.no_grad():
            features = self.features(x)
            pooled = self.avgpool(features)
        pooled = pooled.view(pooled.size(0), -1)
        out = self.fc(pooled)
        return out

# Instantiate the CNN-based encoder and set to eval mode
dna_cnn_encoder = DNACNNEncoder(output_dim=384)
dna_cnn_encoder.eval()

def encode_dna_batch_cnn(dna_sequences, desired_size=64):
    tensors = [dna_to_tensor(seq, desired_size=desired_size) for seq in dna_sequences]
    input_tensor = torch.stack(tensors)  # shape: [batch, 3, desired_size, desired_size]
    with torch.no_grad():
        embeddings = dna_cnn_encoder(input_tensor)
    return embeddings.numpy()

# -------------------------
# (Legacy) Token Map and DNA Parameters - no longer used with CNN encoding
# -------------------------
token_map = {"N": 0, "U": 1, "H": 2, "M": 3, "X": 4, "R": 5, "E": 6}
max_dna_len = 200  # maximum sequence length (unused)

# -------------------------
# Load Pre-trained Model for Description Embeddings
# -------------------------
transformer_model = models.Transformer("Twitter/twhin-bert-base", model_args={'attn_implementation': 'eager'})
pooling_model = models.Pooling(
    transformer_model.get_word_embedding_dimension(),
    pooling_mode_mean_tokens=True
)
st_model = SentenceTransformer(modules=[transformer_model, pooling_model])

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
    
    # Get DNA embeddings using the CNN-based encoder
    train_dna_embs = encode_dna_batch_cnn(train_dna_list)
    test_dna_embs = encode_dna_batch_cnn(test_dna_list)
    
    # Concatenate the description embeddings and DNA embeddings.
    train_X = [np.concatenate([desc, dna_emb]) for desc, dna_emb in zip(train_descs, train_dna_embs)]
    test_X = [np.concatenate([desc, dna_emb]) for desc, dna_emb in zip(test_descs, test_dna_embs)]
    
    return np.array(train_X), np.array(test_X), np.array(train_labels), np.array(test_labels)

# -------------------------
# Main Pipeline with Optuna Hyperparameter Tuning, SMOTE, and Accuracy Scoring
# -------------------------
if __name__ == "__main__":
    cur_dir = os.path.dirname(__file__)
    data_dir = os.path.join(cur_dir, "../data")

    # Load data using our custom function
    X_train, X_test, y_train, y_test = load_data(
        data_dir, 
        session_numbers=[12, 13], 
        st_model=st_model, 
        xnums=[0]
    )
    
    # Compute scale_pos_weight for class imbalance
    neg, pos = np.bincount(y_train)
    scale_pos_weight = neg / pos

    """
    Train & valid (with hyperparameter tuning) + test pipeline
    """
    import numpy as np
    import optuna
    from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
    from xgboost import XGBClassifier
    from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, average_precision_score
    from sklearn.preprocessing import StandardScaler
    from imblearn.pipeline import Pipeline

    # Compute scale_pos_weight for class imbalance
    neg, pos = np.bincount(y_train)
    scale_pos_weight = neg / pos

    def objective(trial):
        # Wrap hyperparameters in a dictionary.
        params = {
            "n_estimators": 850,
            "learning_rate": 0.0345791693662524,
            "max_depth": 7,
            "gamma": 0.005193248963216129,
            "min_child_weight": 1,
            # "scale_pos_weight": scale_pos_weight, not used (yet)
            "objective": "binary:logistic",  
            "random_state": 42,
            "use_label_encoder": False,
            "eval_metric": "logloss"
        }
        
        clf = XGBClassifier(**params)
        
        # Create a pipeline with StandardScaler, SMOTE, and the classifier.
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', clf)
        ])
        
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1)
        return np.mean(scores)

    # Optimize hyperparameters using Optuna.
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50)

    # Print best trial details.
    print("Best trial:")
    trial = study.best_trial
    print("  Mean score:", trial.value)
    for key, value in trial.params.items():
        print(f"  {key}: {value}")

    # Build and fit the final pipeline using the best hyperparameters.
    final_params = {
        **trial.params,
    }

    best_clf = XGBClassifier(**final_params)
    best_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', best_clf)
    ])
    best_pipeline.fit(X_train, y_train)

    # Generate predictions and evaluate.
    y_pred = best_pipeline.predict(X_test)
    y_pred_proba = best_pipeline.predict_proba(X_test)[:, 1]

    print("Test Accuracy: {:.4f}".format(accuracy_score(y_test, y_pred)))
    print("Test ROC AUC: {:.4f}".format(roc_auc_score(y_test, y_pred_proba)))
    print("Test AUPR: {:.4f}".format(average_precision_score(y_test, y_pred_proba)))
    print("Classification Report:\n", classification_report(y_test, y_pred))

    # Save model
    # import pickle
    # model_dir = os.path.join(cur_dir, '../models/GBM')
    # model_file = os.path.join(model_dir, "XGBoost-CNN.pkl")
    # with open(model_file, "wb") as f:
    #     pickle.dump(best_pipeline, f)
    
    # print("Model saved to:", model_file)
