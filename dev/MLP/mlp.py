import os
import re
import copy
import ijson
import torch
import random
import optuna
import numpy as np
import pandas as pd
from PIL import Image
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from hybrid import NewMultiModalAttentionFusion
from attn_model import MultiModalAttentionFusion
from sklearn.model_selection import train_test_split
from torchvision import transforms, models as tv_models  # for CNN encoder
from sentence_transformers import SentenceTransformer, models  # for text embeddings
from torch.utils.data import TensorDataset, DataLoader, Subset
from time_utils import generate_time_dna, TimeDNAEncoder, encode_time_dna_batch_cnn
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, average_precision_score
from sklearn.utils.class_weight import compute_class_weight

# Instantiate and set to evaluation mode.
time_dna_encoder = TimeDNAEncoder(output_dim=384)
time_dna_encoder.eval()

# -------------------------
# Set Seed for Reproducibility
# -------------------------
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

# -------------------------
# Digital DNA Functions
# -------------------------
def get_content_dna_symbol(tweet_text):
    url_present = bool(re.search(r"https?://\S+", tweet_text))
    hashtag_present = bool(re.search(r"#\w+", tweet_text))
    mention_present = bool(re.search(r"@\w+", tweet_text))
    
    entity_types = sum([url_present, hashtag_present, mention_present])
    
    if entity_types == 0:
        return "N"
    elif entity_types == 1:
        if url_present:
            return "U"
        elif hashtag_present:
            return "H"
        elif mention_present:
            return "M"
    else:
        return "X"

def generate_content_dna(tweets):
    tweets.sort(key=lambda x: datetime.fromisoformat(x["created_at"].replace("Z", "+00:00")))
    dna = "".join(get_content_dna_symbol(tweet["text"]) for tweet in tweets)
    return dna
# ---------------------------------------------------------------------------
# -------------------------
# CNN-based DNA Encoder
# -------------------------
def dna_to_tensor(dna, 
                  mapping={"N": 0, "U": 64, "H": 128, "M": 192, "X": 255},
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
        torch.manual_seed(42)
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
# ------------------------------------------------------------
# -------------------------
# Text Embeddings using twhin-bert via SentenceTransformer
# -------------------------
transformer_model = models.Transformer("Twitter/twhin-bert-base", model_args={'attn_implementation': 'eager'})
pooling_model = models.Pooling(transformer_model.get_word_embedding_dimension(), pooling_mode_mean_tokens=True)
st_model = SentenceTransformer(modules=[transformer_model, pooling_model])
# Note: twhin-bert outputs 768-dim embeddings.
# ------------------------------------------------------------
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
# --------------------------------------------------------------------------------------------------

# -------------------------
# Evaluation Function
# -------------------------
def evaluate_model(model, data_loader, device, criterion):
    """
    Evaluates the model on the provided data_loader.
    Returns: tuple (average loss, accuracy)
    """
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch_X, batch_y in data_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            outputs = model(batch_X)
            
            # Compute batch loss and aggregate
            loss = criterion(outputs, batch_y)
            total_loss += loss.item() * batch_X.size(0)
            total_samples += batch_X.size(0)
            
            # Get predictions
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(batch_y.cpu().numpy())

    avg_loss = total_loss / total_samples
    acc = accuracy_score(all_labels, all_preds)
    return avg_loss, acc

# -------------------------
# Reusable Training Function
# -------------------------
def train_model(model, train_loader, val_loader, device, criterion, optimizer, scheduler,
                num_epochs, verbose=False):
    """
    Trains the model for a fixed number of epochs and returns the validation accuracy
    obtained after training on all (train) data.
    
    Args:
        model: The neural network to train.
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        device: torch.device (e.g., "cuda" or "cpu").
        criterion: Loss function.
        optimizer: Optimizer.
        scheduler: Learning rate scheduler (ReduceLROnPlateau).
        num_epochs: Number of training epochs.
        verbose (bool): If True, prints train loss, val loss, and val accuracy per epoch.
        
    Returns:
        final_val_acc: The validation accuracy obtained after training on all train data.
    """

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0.0
        total_samples = 0
        
        # Training loop: accumulate loss over batches
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item() * batch_X.size(0)
            total_samples += batch_X.size(0)
        
        train_loss = total_train_loss / total_samples
        
        # Optionally evaluate on validation set during training for logging purposes
        val_loss, val_acc = evaluate_model(model, val_loader, device, criterion)
        scheduler.step(val_loss)
        
        if verbose:
            print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f} "
                  f"- Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f} "
                  f"- LR: {optimizer.param_groups[0]['lr']:.6f}")
    
    # Final evaluation on validation set after training on all data
    final_val_loss, final_val_acc = evaluate_model(model, val_loader, device, criterion)
    return final_val_acc
# -----------------------------------------------------------------
# -------------------------
# Hyperparameter Tuning Objective Function using Optuna
# -------------------------
def objective(trial):
    model = NewMultiModalAttentionFusion()
    model.to(device)
    
    # Tune learning rate and weight decay.
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=6)
    
    # Create a stratified train/validation split.
    indices = np.arange(len(X_train_tensor))
    train_idx, val_idx = train_test_split(
        indices, test_size=0.1, random_state=42, stratify=y_train_tensor.numpy()
    )
    train_subset = Subset(TensorDataset(X_train_tensor, y_train_tensor), train_idx)
    val_subset = Subset(TensorDataset(X_train_tensor, y_train_tensor), val_idx)
    
    train_loader_trial = DataLoader(train_subset, batch_size=64, shuffle=True)
    val_loader_trial = DataLoader(val_subset, batch_size=64, shuffle=False)
    
    # Train the model using the reusable training function.
    # Here we don't print intermediate results.
    best_val_acc = train_model(model, train_loader_trial, val_loader_trial,
                            device, criterion, optimizer, scheduler,
                            num_epochs=20, verbose=False)
    
    return best_val_acc
# -----------------------------------------------------------------
# -------------------------
# Main Pipeline
# -------------------------
if __name__ == "__main__":
    set_seed(42)
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
    test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), 
                             batch_size=64, shuffle=False)
    
    # -------- device ----------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Training on device:", device)
    # --------------------------
    
    # ----- class weights ------
    from sklearn.utils.class_weight import compute_class_weight
    y_train_np = y_train_tensor.numpy()
    classes = np.unique(y_train_np)
    class_weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train_np)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
    # --------------------------
    
    # ------- Hyperparameter Tuning with Optuna -------
    study = optuna.create_study(direction="maximize", 
                                pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials=15)
    
    print("Best trial:")
    best_trial = study.best_trial
    print("  Best Validation Accuracy:", best_trial.value)
    for key, value in best_trial.params.items():
        print(f"  {key}: {value}")
    # ----------------------------------
    
    # -------------------- Final Model and Training --------------------
    final_learning_rate = best_trial.params["learning_rate"]
    final_weight_decay = best_trial.params["weight_decay"]
    
    final_model = NewMultiModalAttentionFusion()
    final_model.to(device)
    
    final_optimizer = optim.Adam(final_model.parameters(), 
                                 lr=final_learning_rate, 
                                 weight_decay=final_weight_decay)
    final_criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(final_optimizer, patience=6)
    
    # Train the final model using the reusable training function.
    # Now we pass verbose=True to print metrics for each epoch.
    num_epochs = 20
    best_val_acc = train_model(final_model, train_loader, val_loader, device,
                            final_criterion, final_optimizer, scheduler,
                            num_epochs=num_epochs, verbose=True)
    # -----------------------------------------------------------
    # ------------------- Test Final Model ----------------------
    _, test_acc = evaluate_model(final_model, test_loader, device, final_criterion)
    print("Test Accuracy: {:.4f}".format(test_acc))
    # -----------------------------------------------------------
    # ------------------ Classification Report ------------------
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


