import os
import torch
import optuna
import random
import numpy as np
import torch.optim as optim
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
# scheduler_params = {"patience": 10, 
#                     "factor": 0.8
#                     }

scheduler_params = {"milestones": [10], 
                    "gamma": 0.95
                    }

WEIGHT_DECAY = 1e-6

# focal_loss_params = {"gamma": 0.1, 
                    #  "alpha": 0.45
                    #  }


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
                              session_numbers=[], 
                              st_model=st_model, 
                              xnums=[0]
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
    
    
    # -------------------- Final Model and Training --------------------
    final_learning_rate = 1e-4

    # final_model = GMUAttention()
    final_model = MOEAttention()
    final_model.to(device)
    
    final_optimizer = optim.Adam(final_model.parameters(), lr=final_learning_rate, weight_decay=WEIGHT_DECAY)
    final_criterion = torch.nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.MultiStepLR(final_optimizer, **scheduler_params)
    
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