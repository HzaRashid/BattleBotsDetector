import os
import re
import json
import ijson
import math
import numpy as np
from datetime import datetime
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, Subset


from torchvision import transforms, models as tv_models  # for CNN encoder
from sentence_transformers import SentenceTransformer, models  # for text embeddings

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, average_precision_score

import optuna

from time_utils import generate_time_dna, time_dna_to_tensor, TimeDNAEncoder, encode_time_dna_batch_cnn
# Instantiate and set to evaluation mode.
time_dna_encoder = TimeDNAEncoder(output_dim=128)
time_dna_encoder.eval()

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
    Converts a DNA string into a grayscale image that is resized to desired_size
    and then converted to a normalized RGB tensor.
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
        # Use pretrained MobileNetV2 from torchvision.
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

# Instantiate CNN-based encoder and set to evaluation mode.
dna_cnn_encoder = DNACNNEncoder()
dna_cnn_encoder.eval()

def encode_dna_batch_cnn(dna_sequences, desired_size=64):
    tensors = [dna_to_tensor(seq, desired_size=desired_size) for seq in dna_sequences]
    input_tensor = torch.stack(tensors)  # shape: [batch, 3, desired_size, desired_size]
    with torch.no_grad():
        embeddings = dna_cnn_encoder(input_tensor)
    return embeddings.numpy()

# -------------------------
# Text Embeddings using twhin-bert via SentenceTransformer
# -------------------------
transformer_model = models.Transformer("Twitter/twhin-bert-base", model_args={'attn_implementation': 'eager'})
pooling_model = models.Pooling(transformer_model.get_word_embedding_dimension(), pooling_mode_mean_tokens=True)
st_model = SentenceTransformer(modules=[transformer_model, pooling_model])
# Note: twhin-bert outputs 768-dim embeddings.

# -------------------------
# Data Loading Function
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

    train_time_dna_list, test_time_dna_list = [], []
    
    for user in user_info_list:
        uid = user.get("user_id")
        if uid is None or uid not in user_posts_dict:
            continue
        description = user.get("description", "")
        desc_emb = st_model.encode(description)  # 768-dim text embedding

        # Generate content DNA.
        content_dna = generate_content_dna(user_posts_dict[uid])
        # Generate time DNA.
        time_dna = generate_time_dna(user_posts_dict[uid])

        label = int(user.get("is_bot", 0))

        if uid in train_user_ids:
            train_descs.append(desc_emb)
            train_dna_list.append(content_dna)
            train_time_dna_list.append(time_dna)
            train_labels.append(label)
        elif uid in test_user_ids:
            test_descs.append(desc_emb)
            test_dna_list.append(content_dna)
            test_time_dna_list.append(time_dna)
            test_labels.append(label)
    
    # Get DNA embeddings.
    train_dna_embs = encode_dna_batch_cnn(train_dna_list)
    test_dna_embs = encode_dna_batch_cnn(test_dna_list)
    train_time_dna_embs = encode_time_dna_batch_cnn(train_time_dna_list, time_dna_encoder=time_dna_encoder)
    test_time_dna_embs = encode_time_dna_batch_cnn(test_time_dna_list, time_dna_encoder=time_dna_encoder)

    # Concatenate text, content DNA, and time DNA embeddings.
    train_X = [
        np.concatenate([desc, dna_emb, time_emb])
        for desc, dna_emb, time_emb in zip(train_descs, train_dna_embs, train_time_dna_embs)
    ]
    test_X = [
        np.concatenate([desc, dna_emb, time_emb])
        for desc, dna_emb, time_emb in zip(test_descs, test_dna_embs, test_time_dna_embs)
    ]
    
    return np.array(train_X), np.array(test_X), np.array(train_labels), np.array(test_labels)


# -------------------------
# Flexible Fusion MLP Model
# -------------------------
class FlexibleFusionMLP(nn.Module):
    def __init__(self, 
                 text_dim=768,
                 projected_text_dim=256,
                 content_dna_dim=384, 
                 time_dna_dim=128, 
                 cat_dim=768, 
                 fusion_layers=None
                 ):
        """
        text_dim: dimension of raw text embeddings.
        projected_text_dim: dimension after projecting text embeddings.
        fusion_layers: an nn.Sequential module containing the fusion FC layers.
        """
        super(FlexibleFusionMLP, self).__init__()

        # project to lower dimensions
        self.projection = nn.Linear(text_dim + content_dna_dim + time_dna_dim, cat_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.layer_norm= nn.LayerNorm(cat_dim)  # Layer Normalization applied after projection

        # fusion_layers is built dynamically based on hyperparameters.
        self.fusion_layers = fusion_layers

    def forward(self, x):
        # x is a concatenated vector: 
        # [text_embedding (768), content_dna_embedding (384), time_dna_embedding (128)]
        text_embedding = x[:, :768]
        content_dna_embedding = x[:, 768:1152]  # next 384 dims
        time_dna_embedding = x[:, 1152:]        # remaining 128 dims

        fusion_input = torch.cat([text_embedding, 
                                  content_dna_embedding, 
                                  time_dna_embedding], 
                                  dim=1)
        fusion_input = self.projection(fusion_input)
        # fusion_input = self.layer_norm(fusion_input)
        fusion_input = self.relu(fusion_input)
        # Pass through the flexible fusion layers.
        logits = self.fusion_layers(fusion_input)
        return logits
    
# -------------------------
# Evaluation Function
# -------------------------
def evaluate_model(model, data_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch_X, batch_y in data_loader:
            batch_X = batch_X.to(device)
            outputs = model(batch_X)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(batch_y.numpy())
    acc = accuracy_score(all_labels, all_preds)
    return acc


text_proj_dim = 256
content_dna_proj_dim = 256
time_dna_proj_dim = 128
fusion_in_dim = 768

# -------------------------
# -------------------------
# Updated Objective Function for Hyperparameter Tuning
# -------------------------
def objective(trial):
    # Suggest the number of fully connected layers (e.g., between 1 and 3 layers).
    n_layers = trial.suggest_int("n_layers", 2, 4)
    # Suggest a dropout rate (applied in each layer).
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    
    # Build the fusion layers dynamically.
    layers = []

    # input dimension after projection + concatenation:
    in_dim = fusion_in_dim
    
    # Create n_layers blocks of [Linear -> ReLU -> Dropout].
    for i in range(n_layers):
        hidden_units = trial.suggest_int(f"n_units_l{i}", 128, 512)
        layers.append(nn.Linear(in_dim, hidden_units))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        in_dim = hidden_units
    # Final output layer: 2 classes.
    layers.append(nn.Linear(in_dim, 2))
    
    # Pack the layers into a Sequential module.
    fusion_layers = nn.Sequential(*layers)
    
    # Instantiate the flexible model.
    model = FlexibleFusionMLP(
        text_dim=768,
        projected_text_dim=256,
        content_dna_dim=384,
        time_dna_dim=128,
        fusion_layers=fusion_layers
    )
    model.to(device)
    
    # Suggest learning rate and weight decay.
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-4, 1e-2)
    weight_decay = trial.suggest_loguniform("weight_decay", 1e-5, 1e-2)
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    
    # Create a stratified train/validation split.
    indices = np.arange(len(X_train_tensor))
    train_idx, val_idx = train_test_split(
        indices, test_size=0.1, random_state=42, stratify=y_train_tensor.numpy()
    )
    train_subset = Subset(TensorDataset(X_train_tensor, y_train_tensor), train_idx)
    val_subset = Subset(TensorDataset(X_train_tensor, y_train_tensor), val_idx)
    
    train_loader_trial = DataLoader(train_subset, batch_size=64, shuffle=True)
    val_loader_trial = DataLoader(val_subset, batch_size=64, shuffle=False)
    
    best_val_acc = 0.0
    # Train for a fixed number of epochs for tuning.
    for epoch in range(10):
        model.train()
        for batch_X, batch_y in train_loader_trial:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
        val_acc = evaluate_model(model, val_loader_trial, device)
        best_val_acc = max(best_val_acc, val_acc)
    
    return best_val_acc

# -------------------------
# Main Pipeline
# -------------------------
if __name__ == "__main__":
    cur_dir = os.path.dirname(__file__)
    data_dir = os.path.join(cur_dir, "../data")
    
    # Load data.
    X_train, X_test, y_train, y_test = load_data(
        data_dir,
        session_numbers=[12, 13],
        st_model=st_model,
        xnums=[0]
    )
    
    # Convert numpy arrays to torch tensors.
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)
    
    # Create stratified train/validation split for final training.
    indices = np.arange(len(X_train_tensor))
    train_idx, val_idx = train_test_split(
        indices, test_size=0.1, random_state=42, stratify=y_train_tensor.numpy()
    )
    train_dataset = Subset(TensorDataset(X_train_tensor, y_train_tensor), train_idx)
    val_dataset = Subset(TensorDataset(X_train_tensor, y_train_tensor), val_idx)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=64, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Training on device:", device)
    

    # Hyperparameter tuning with Optuna using the updated objective.
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50)
    
    print("Best trial:")
    best_trial = study.best_trial
    print("  Best Validation Accuracy:", best_trial.value)
    for key, value in best_trial.params.items():
        print(f"  {key}: {value}")
    
    # Use the best hyperparameters to build and train the final model.
    final_learning_rate = best_trial.params["learning_rate"]
    final_weight_decay = best_trial.params["weight_decay"]
    # For example:
    final_n_layers = best_trial.params["n_layers"]
    final_dropout = best_trial.params["dropout"]
    
    fusion_layers = []
    in_dim = fusion_in_dim
    for i in range(final_n_layers):
        hidden_units = best_trial.params[f"n_units_l{i}"]
        fusion_layers.append(nn.Linear(in_dim, hidden_units))
        fusion_layers.append(nn.ReLU())
        fusion_layers.append(nn.Dropout(final_dropout))
        in_dim = hidden_units
    fusion_layers.append(nn.Linear(in_dim, 2))
    final_fusion_layers = nn.Sequential(*fusion_layers)
    
    final_model = FlexibleFusionMLP(
        text_dim=768,
        projected_text_dim=256,
        content_dna_dim=384,
        time_dna_dim=128,
        fusion_layers=final_fusion_layers
    )
    final_model.to(device)

    num_epochs = 25
    final_optimizer = optim.NAdam(final_model.parameters(), lr=final_learning_rate, weight_decay=final_weight_decay)
    final_criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ExponentialLR(final_optimizer, gamma=0.9)
    
    best_val_acc = 0.0
    best_state = None
    for epoch in range(num_epochs):
        final_model.train()
        total_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            final_optimizer.zero_grad()
            outputs = final_model(batch_X)
            loss = final_criterion(outputs, batch_y)
            loss.backward()
            final_optimizer.step()
            total_loss += loss.item() * batch_X.size(0)
        avg_loss = total_loss / len(train_loader.dataset)



        # Evaluate on validation set.
        final_model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = final_model(batch_X)
                loss = final_criterion(outputs, batch_y)
                val_loss += loss.item() * batch_X.size(0)
        avg_val_loss = val_loss / len(val_loader.dataset)
        val_acc = evaluate_model(final_model, val_loader, device)
        
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_loss:.4f} - Val Loss: {avg_val_loss:.4f} - Val Acc: {val_acc:.4f} - LR: {final_optimizer.param_groups[0]['lr']:.6f}")
        
        # Step the scheduler at the end of the epoch.
        scheduler.step()


        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = final_model.state_dict()
    
    if best_state is not None:
        final_model.load_state_dict(best_state)
    
    # Evaluate final model on test set.
    test_acc = evaluate_model(final_model, test_loader, device)
    print("Test Accuracy: {:.4f}".format(test_acc))
    
    # Detailed classification report.
    final_model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device)
            outputs = final_model(batch_X)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(batch_y.numpy())
    
    print("Classification Report:\n", classification_report(all_labels, all_preds))
    print("Test ROC AUC: {:.4f}".format(roc_auc_score(all_labels, all_preds)))
    print("Test AUPR: {:.4f}".format(average_precision_score(all_labels, all_preds)))