import os
import torch
import optuna
import random
import numpy as np
import torch.optim as optim
from focal_loss import FocalLoss
# from hybridv2 import GMUAttention
from hybridv3 import MOEAttention
from load_tensors import load_data
from train_model import train_model, evaluate_model
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer, models
from torch.utils.data import TensorDataset, DataLoader, Subset
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score

# --- fixed batch size ---
BATCH_SIZE = 64

def set_seed(seed):
    """seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)



MOE=True
scheduler_params = {"patience": 10, 
                    "factor": 0.8
                    }


WEIGHT_DECAY = 1e-6

# focal_loss_params = {"gamma": 0.1, 
                    #  "alpha": 0.45
                    #  }

# -------------------------
# Hyperparameter Tuning Objective Function using Optuna
# -------------------------
# def objective(trial):
#     # model = GMUAttention()
#     model = MOEAttention()
#     model.to(device)
#     # hyperparameters for tuning
#     learning_rate = trial.suggest_float("learning_rate", 1e-6, 1e-3, log=True)
#     # optimizer and scheduler settings
#     optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=WEIGHT_DECAY)
#     criterion = torch.nn.CrossEntropyLoss()
#     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **scheduler_params)
#     # stratified train/validation split
#     indices = train_idx
#     obj_train_idx, obj_val_idx = train_test_split(
#         indices, 
#         test_size=0.2, 
#         random_state=random.randint(1, 100), 
#         stratify=np.array([train_tensors[-1][i].item() for i in indices])
#     )
#     train_set = Subset(TensorDataset(*train_tensors), obj_train_idx)
#     val_set = Subset(TensorDataset(*train_tensors), obj_val_idx)
    
#     train_trial = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
#     val_trial = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)
    
#     val_acc, val_loss = train_model(model, train_trial, val_trial,
#                                     device, criterion, optimizer, scheduler,
#                                     num_epochs=20, verbose=False, trial=trial, 
#                                     optuna=optuna, torch=torch, moe=MOE)
    
#     return val_acc


# -------------------------
# Main Pipeline
# -------------------------
if __name__ == "__main__":
    set_seed(42)
    cur_dir = os.path.dirname(__file__)
    data_dir = os.path.join(cur_dir, "../data")
    # twhin-bert for sentence (user description) embeddings
    transformer_model = models.Transformer("Twitter/twhin-bert-base", model_args={'attn_implementation': 'eager'})
    pooling_model = models.Pooling(transformer_model.get_word_embedding_dimension(), pooling_mode_mean_tokens=True)
    st_model = SentenceTransformer(modules=[transformer_model, pooling_model])
    # outputs 768-dim embeddings
    # --------------------------------------------
    dtypes = [torch.float32, torch.float32, torch.float32, torch.float32, torch.long]
    
    # Load data
    (train, test) = load_data(data_dir, 
                              session_numbers=[10, 16, 17, 18, 19], 
                              st_model=st_model, 
                              xnums=[]
                              )
    train_tensors = [torch.tensor(train[i], dtype=dtypes[i]) for i in range(len(train))]
    test_tensors = [torch.tensor(test[i], dtype=dtypes[i]) for i in range(len(test))]
    
    # stratified train/validation split for final training.
    indices = np.arange(len(train_tensors[0]))
    train_idx, val_idx = train_test_split(
        indices, test_size=0.1, random_state=42, stratify=train_tensors[-1].numpy()
    )

    train_dataset = Subset(TensorDataset(*train_tensors), train_idx)
    val_dataset = Subset(TensorDataset(*train_tensors), val_idx)
    test_dataset = TensorDataset(*test_tensors)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Training on device:", device) 
    
    # ------- Hyperparameter Tuning with Optuna -------
    # study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner())
    # study.optimize(objective, n_trials=15)
    
    # print("Best trial:")
    # best_trial = study.best_trial
    # print("  Best Validation Accuracy:", best_trial.value)
    # for key, value in best_trial.params.items():
    #     print(f"  {key}: {value}")
    
    # -------------------- Final Model and Training --------------------
    final_learning_rate = 5e-4


    # final_model = GMUAttention()
    final_model = MOEAttention()
    final_model.to(device)
    
    final_optimizer = optim.Adam(final_model.parameters(), lr=final_learning_rate, weight_decay=WEIGHT_DECAY)
    final_criterion = torch.nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(final_optimizer, **scheduler_params)
    
    num_epochs = 20
    val_acc, val_loss = train_model(final_model, train_loader, val_loader, device,
                                    final_criterion, final_optimizer, scheduler,
                                    num_epochs=num_epochs, verbose=True, 
                                    optuna=None, torch=torch, moe=MOE)
    
    # ------------------- Test Final Model ----------------------
    _, test_acc, _, _ = evaluate_model(final_model, test_loader, device, 
                                 final_criterion, torch, moe=MOE)
    print("Test Accuracy: {:.4f}".format(test_acc))
    
    # ------------------ Classification Report ------------------
    final_model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in test_loader:
            inputs = [x.to(device) for x in batch[:-1]]
            labels = batch[-1]
            if MOE:
                outputs, aux_loss = final_model(*inputs)
            else:
                outputs, aux_loss = final_model(*inputs), 0
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
    
    print("Classification Report:\n", classification_report(all_labels, all_preds))
    print("Test ROC AUC: {:.4f}".format(roc_auc_score(all_labels, all_preds)))
    print("Test AUPR: {:.4f}".format(average_precision_score(all_labels, all_preds)))

    from upload_model import upload_model_to_hf
    model_save_path = "hybridv3_weights.bin"
    torch.save(final_model.state_dict(), model_save_path)
    
    # Set your Hugging Face repository ID in the format "<username>/<repo-name>"
    repo_id = "hzarashid/ForensiX"  # <-- CHANGE THIS to your repo id.
    
    upload_model_to_hf(model_save_path, repo_id, commit_message="Upload trained model weights")
    print(f"Model uploaded to Hugging Face repository: {repo_id}")

# # ------------------ Evaluation with Min-Max Scaled Positive-Class Confidences ------------------
# # Here we collect the positive class confidence (probabilities) from softmax for each test example.
# all_pos_confidences = []  # raw positive-class probabilities
# # (We already have the true labels in all_labels from above.)
# with torch.no_grad():
#     for batch in test_loader:
#         inputs = [x.to(device) for x in batch[:-1]]
#         if MOE:
#             outputs, aux_loss = final_model(*inputs)
#         else:
#             outputs, aux_loss = final_model(*inputs), 0
#         # Compute probabilities from logits
#         probs = torch.softmax(outputs, dim=1)
#         pos_conf = probs[:, 1].cpu().numpy()  # positive class is assumed to be index 1
#         all_pos_confidences.extend(pos_conf)

# all_pos_confidences = np.array(all_pos_confidences)

# # Apply min-max scaling across the test set
# min_val = all_pos_confidences.min()
# max_val = all_pos_confidences.max()
# scaled_confidences = (all_pos_confidences - min_val) / (max_val - min_val)

# # Derive binary predictions based on a threshold of 0.5 on the scaled confidences.
# scaled_preds = (scaled_confidences >= 0.1).astype(int)

# print("\nClassification Report (Min-Max Scaled Confidences):\n", classification_report(all_labels, scaled_preds))
# print("Test ROC AUC (Min-Max Scaled): {:.4f}".format(roc_auc_score(all_labels, scaled_confidences)))
# print("Test AUPR (Min-Max Scaled): {:.4f}".format(average_precision_score(all_labels, scaled_confidences)))