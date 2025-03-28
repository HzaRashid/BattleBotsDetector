import os
import ijson
import torch
import random
import optuna
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from hybrid import NewMultiModalAttentionFusion
from hybridv2 import GMUAttention
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer, models  # for text embeddings
from torch.utils.data import TensorDataset, DataLoader, Subset
from content_utils import generate_content_dna, encode_content_dna_batch_cnn
from time_utils import generate_time_dna, encode_time_dna_batch_cnn
from emoji_utils import generate_emoji_dna, encode_emoji_dna_batch_cnn
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, average_precision_score
from focal_loss import FocalLoss

# --- fixed batch size ---
BATCH_SIZE = 64
# -------------------------
# seeds for reproducability
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
# twhin-bert for sentence (user description) embeddings
# -------------------------
transformer_model = models.Transformer("Twitter/twhin-bert-base", model_args={'attn_implementation': 'eager'})
pooling_model = models.Pooling(transformer_model.get_word_embedding_dimension(), pooling_mode_mean_tokens=True)
st_model = SentenceTransformer(modules=[transformer_model, pooling_model])
# outputs 768-dim embeddings
# --------------------------------------------

# -------------------------
# Data Loading Function (returns separate modalities)
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

    # stratified train/test split
    user_info_df = pd.DataFrame(user_info_list)[['user_id', 'is_bot']].drop_duplicates()
    train_users, test_users = train_test_split(
        user_info_df, test_size=0.2, random_state=42, stratify=user_info_df['is_bot']
    )
    train_user_ids = set(train_users['user_id'])
    test_user_ids = set(test_users['user_id'])

    train_texts, test_texts = [], []
    train_dna_list, test_dna_list = [], []
    train_time_list, test_time_list = [], []
    train_emoji, test_emoji = [], []
    train_labels, test_labels = [], []

    for user in user_info_list:
        uid = user.get("user_id")
        if uid is None or uid not in user_posts_dict:
            continue
        description = user.get("description", "")
        # Get text embedding.
        desc_emb = st_model.encode(description)  # 768-dim
        # generate the DNA sequences
        content_dna = generate_content_dna(user_posts_dict[uid])
        time_dna = generate_time_dna(user_posts_dict[uid])
        emoji_dna = generate_emoji_dna(user_posts_dict[uid])
        # user's true label
        label = int(user.get("is_bot", 0))

        if uid in train_user_ids:
            train_texts.append(desc_emb)
            train_dna_list.append(content_dna)
            train_time_list.append(time_dna)
            train_emoji.append(emoji_dna)
            train_labels.append(label)
        elif uid in test_user_ids:
            test_texts.append(desc_emb)
            test_dna_list.append(content_dna)
            test_time_list.append(time_dna)
            test_emoji.append(emoji_dna)
            test_labels.append(label)
    
    # generate the CNN embeddings of the DNA sequences
    train_dna_embs = encode_content_dna_batch_cnn(train_dna_list)
    train_time_embs = encode_time_dna_batch_cnn(train_time_list)
    train_emoji_embs = encode_emoji_dna_batch_cnn(train_emoji)

    test_dna_embs = encode_content_dna_batch_cnn(test_dna_list)
    test_time_embs = encode_time_dna_batch_cnn(test_time_list)
    test_emoji_embs = encode_emoji_dna_batch_cnn(test_emoji)

    # return split with seperable modalities
    train = (np.array(x) 
             for x in [
                 train_texts,
                 train_dna_embs,
                 train_time_embs,
                 train_emoji_embs,
                 train_labels
                 ])
    test = (np.array(x) 
            for x in [
                test_texts,
                test_dna_embs,
                test_time_embs,
                test_emoji_embs,
                test_labels
                ])

    return (*train, *test)

# -------------------------
# Evaluation Function (updated for separate modalities)
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
        for batch in data_loader:
            inputs = [x.to(device) for x in batch[:-1]]
            labels = batch[-1].to(device)
            outputs = model(*inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * inputs[0].size(0)
            total_samples += inputs[0].size(0)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / total_samples
    acc = accuracy_score(all_labels, all_preds)
    return avg_loss, acc

# -------------------------
# Reusable Training Function (updated for separate modalities)
# -------------------------
def train_model(model, train_loader, val_loader, 
                device, criterion, optimizer, scheduler,
                num_epochs, verbose=False, trial=None):
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0.0
        total_samples = 0
        
        for batch in train_loader:
            inputs = [x.to(device) for x in batch[:-1]]
            labels = batch[-1].to(device)
            optimizer.zero_grad()
            outputs = model(*inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item() * inputs[0].size(0)
            total_samples += inputs[0].size(0)
        
        train_loss = total_train_loss / total_samples
        val_loss, val_acc = evaluate_model(model, val_loader, device, criterion)
        scheduler.step(val_loss)

        if trial:
            trial.report(val_acc, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
        
        if verbose:
            print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f} "
                  f"- Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f} "
                  f"- LR: {optimizer.param_groups[0]['lr']:.6f}")
            
    final_val_loss, final_val_acc = evaluate_model(model, val_loader, device, criterion)
    return final_val_acc, final_val_loss

# -------------------------
# Hyperparameter Tuning Objective Function using Optuna
# -------------------------
def objective(trial):
    model = GMUAttention()
    model.to(device)
    # hyperparameters for tuning
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)
    # optimizer and scheduler settings
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = FocalLoss(gamma=2.00, alpha=0.25)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.1)
    # stratified train/validation split
    indices = np.arange(len(X_train_text_tensor))
    train_idx, val_idx = train_test_split(
        indices, 
        test_size=0.2, 
        random_state=random.randint(1, 100), 
        stratify=y_train_tensor.numpy()
    )
    train_subset = Subset(TensorDataset(*train_tensors), train_idx)
    val_subset = Subset(TensorDataset(*train_tensors), val_idx)
    
    train_loader_trial = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader_trial = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False)
    
    val_acc, val_loss = train_model(model, train_loader_trial, val_loader_trial,
                                    device, criterion, optimizer, scheduler,
                                    num_epochs=20, verbose=False, trial=trial)
    
    return val_acc

# -------------------------
# Main Pipeline
# -------------------------
if __name__ == "__main__":
    set_seed(42)
    cur_dir = os.path.dirname(__file__)
    data_dir = os.path.join(cur_dir, "../data")
    
    # Load data. X_train_emoji X_test_emoji
    (X_train_text, 
     X_train_content, 
     X_train_time, 
     X_train_emoji,  
     y_train,
     X_test_text, 
     X_test_content, 
     X_test_time, 
     X_test_emoji, 
     y_test) = load_data(data_dir, 
                         session_numbers=[4, 10, 11, 12, 13, 14, 15, 16, 17], 
                         st_model=st_model, 
                         xnums=[]
                         )
    
    # Convert numpy arrays to torch tensors.
    X_train_text_tensor = torch.tensor(X_train_text, dtype=torch.float32)
    X_train_content_tensor = torch.tensor(X_train_content, dtype=torch.float32)
    X_train_time_tensor = torch.tensor(X_train_time, dtype=torch.float32)
    X_train_emoji_tensor = torch.tensor(X_train_emoji, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    
    X_test_text_tensor = torch.tensor(X_test_text, dtype=torch.float32)
    X_test_content_tensor = torch.tensor(X_test_content, dtype=torch.float32)
    X_test_time_tensor = torch.tensor(X_test_time, dtype=torch.float32)
    X_test_emoji_tensor = torch.tensor(X_test_emoji, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)
    
    # stratified train/validation split for final training.
    indices = np.arange(len(X_train_text_tensor))
    train_idx, val_idx = train_test_split(
        indices, test_size=0.1, random_state=42, stratify=y_train_tensor.numpy()
    )

    train_tensors = (X_train_text_tensor, 
                     X_train_content_tensor, 
                     X_train_time_tensor, 
                     X_train_emoji_tensor, 
                     y_train_tensor)
    test_tensors = (X_test_text_tensor, 
                    X_test_content_tensor, 
                    X_test_time_tensor, 
                    X_test_emoji_tensor,
                    y_test_tensor)
    train_dataset = Subset(TensorDataset(*train_tensors), train_idx)
    val_dataset = Subset(TensorDataset(*train_tensors), val_idx)
    test_dataset = TensorDataset(*test_tensors)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Training on device:", device)
    
    # ------- Hyperparameter Tuning with Optuna -------
    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials=30)
    
    print("Best trial:")
    best_trial = study.best_trial
    print("  Best Validation Accuracy:", best_trial.value)
    for key, value in best_trial.params.items():
        print(f"  {key}: {value}")
    
    # -------------------- Final Model and Training --------------------
    final_learning_rate = best_trial.params["learning_rate"]
    final_weight_decay = best_trial.params["weight_decay"]
    
    final_model = GMUAttention()
    final_model.to(device)
    
    final_optimizer = optim.Adam(final_model.parameters(), lr=final_learning_rate, weight_decay=final_weight_decay)
    final_criterion = FocalLoss(gamma=2.00, alpha=0.25)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(final_optimizer, patience=5, factor=0.1)
    
    num_epochs = 20
    val_acc, val_loss = train_model(final_model, train_loader, val_loader, device,
                                    final_criterion, final_optimizer, scheduler,
                                    num_epochs=num_epochs, verbose=True)
    
    # ------------------- Test Final Model ----------------------
    _, test_acc = evaluate_model(final_model, test_loader, device, final_criterion)
    print("Test Accuracy: {:.4f}".format(test_acc))
    
    # ------------------ Classification Report ------------------
    final_model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in test_loader:
            inputs = [x.to(device) for x in batch[:-1]]
            labels = batch[-1]
            outputs = final_model(*inputs)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
    
    print("Classification Report:\n", classification_report(all_labels, all_preds))
    print("Test ROC AUC: {:.4f}".format(roc_auc_score(all_labels, all_preds)))
    print("Test AUPR: {:.4f}".format(average_precision_score(all_labels, all_preds)))

    # ------------------- Save and Upload Model to Hugging Face ----------------------
    from upload_model import upload_model_to_hf
    model_save_path = "hybridv2_weights.bin"
    torch.save(final_model.state_dict(), model_save_path)
    
    # Set your Hugging Face repository ID in the format "<username>/<repo-name>"
    repo_id = "hzarashid/ForensiX"  # <-- CHANGE THIS to your repo id.
    
    upload_model_to_hf(model_save_path, repo_id, commit_message="Upload trained model weights")
    print(f"Model uploaded to Hugging Face repository: {repo_id}")

    # ----- optional class weights if using BCE loss ------
    # from sklearn.utils.class_weight import compute_class_weight
    # y_train_np = y_train_tensor.numpy()
    # classes = np.unique(y_train_np)
    # class_weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train_np)
    # class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
    # --------------------------
    # -------------------------
    # Helper function to downsample negatives in a given set of indices.
    # The target ratio is positive:negative = 1:10.
    # -------------------------
    # def downsample_indices(indices, labels, target_ratio=10):
    #     indices = np.array(indices)
    #     pos_idx = indices[labels[indices] == 1]
    #     neg_idx = indices[labels[indices] == 0]
    #     desired_neg_count = min(len(neg_idx), target_ratio * len(pos_idx))
    #     neg_idx_downsampled = np.random.choice(neg_idx, size=desired_neg_count, replace=False)
    #     combined = np.concatenate([pos_idx, neg_idx_downsampled])
    #     np.random.shuffle(combined)
    #     return combined
    # ----- optional downsampling for entire training set -----
    # train_idx_downsampled = downsample_indices(train_idx, y_train_tensor.numpy(), target_ratio=8)
    # ---------------------------------

    # ----- optional downsampling for optuna -----
    # train_idx_downsampled = downsample_indices(train_idx, y_train_tensor.numpy(), target_ratio=8)
    # rejected_idx = np.setdiff1d(train_idx, train_idx_downsampled)
    # val_idx_updated = np.concatenate([val_idx, rejected_idx])
    # -------------------------------




