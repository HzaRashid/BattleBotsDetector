import os
import torch
import random
import optuna
import numpy as np
import torch.optim as optim
from focal_loss import FocalLoss
from hybridv2 import GMUAttention
from load_tensors import load_data
from train_model import train_model, evaluate_model
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer, models
from torch.utils.data import TensorDataset, DataLoader, Subset
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score

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
    criterion = FocalLoss(gamma=2.7, alpha=0.025)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.25)
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
                                    num_epochs=20, verbose=False, trial=trial, optuna=optuna)
    
    return val_acc

# -------------------------
# Main Pipeline
# -------------------------
if __name__ == "__main__":
    set_seed(42)
    cur_dir = os.path.dirname(__file__)
    data_dir = os.path.join(cur_dir, "../data")
    
    # Load data
    (X_train_text, 
     X_train_content, 
     X_train_time, 
     y_train,
     X_test_text, 
     X_test_content, 
     X_test_time, 
     y_test) = load_data(data_dir, 
                         session_numbers=[], 
                         st_model=st_model, 
                         xnums=[0]
                         )
    
    # Convert numpy arrays to torch tensors.
    X_train_text_tensor = torch.tensor(X_train_text, dtype=torch.float32)
    X_train_content_tensor = torch.tensor(X_train_content, dtype=torch.float32)
    X_train_time_tensor = torch.tensor(X_train_time, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    
    X_test_text_tensor = torch.tensor(X_test_text, dtype=torch.float32)
    X_test_content_tensor = torch.tensor(X_test_content, dtype=torch.float32)
    X_test_time_tensor = torch.tensor(X_test_time, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)
    
    # stratified train/validation split for final training.
    indices = np.arange(len(X_train_text_tensor))
    train_idx, val_idx = train_test_split(
        indices, test_size=0.1, random_state=42, stratify=y_train_tensor.numpy()
    )

    train_tensors = (X_train_text_tensor, 
                     X_train_content_tensor, 
                     X_train_time_tensor, 
                     y_train_tensor)
    test_tensors = (X_test_text_tensor, 
                    X_test_content_tensor, 
                    X_test_time_tensor, 
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
    study.optimize(objective, n_trials=15)
    
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
    final_criterion = FocalLoss(gamma=2.7, alpha=0.025)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(final_optimizer, patience=5, factor=0.25)
    
    num_epochs = 20
    val_acc, val_loss = train_model(final_model, train_loader, val_loader, device,
                                    final_criterion, final_optimizer, scheduler,
                                    num_epochs=num_epochs, verbose=True, optuna=optuna)
    
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



