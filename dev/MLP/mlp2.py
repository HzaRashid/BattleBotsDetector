import os
import torch
import optuna
import random
import numpy as np
import torch.optim as optim
from hybridv4 import MOEAttention
from load_tensors2 import load_data
from train_model2 import train_model, evaluate_model
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer, models
from torch.utils.data import TensorDataset, DataLoader, Subset
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score

# --- Fixed batch size ---
BATCH_SIZE = 64

def set_seed(seed):
    """Sets seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

MOE = True
scheduler_params = {"patience": 10, "factor": 0.8}
WEIGHT_DECAY = 1e-6

if __name__ == "__main__":
    set_seed(42)
    cur_dir = os.path.dirname(__file__)
    data_dir = os.path.join(cur_dir, "../data")
    
    # Initialize SentenceTransformer for generating description embeddings.
    transformer_model = models.Transformer("Twitter/twhin-bert-base", model_args={'attn_implementation': 'eager'})
    pooling_model = models.Pooling(transformer_model.get_word_embedding_dimension(), pooling_mode_mean_tokens=True)
    st_model = SentenceTransformer(modules=[transformer_model, pooling_model])
    
    # Load data – load_data returns a dictionary for train and test splits.
    train_data, test_data = load_data(data_dir, session_numbers=[], st_model=st_model, xnums=[0])
    
    # Convert numeric modalities (embeddings and labels) to torch tensors.
    train_desc_embs = torch.tensor(train_data["desc_embs"], dtype=torch.float32)
    train_dna_embs = torch.tensor(train_data["dna_embs"], dtype=torch.float32)
    train_time_embs = torch.tensor(train_data["time_embs"], dtype=torch.float32)
    train_labels = torch.tensor(train_data["labels"], dtype=torch.long)
    
    test_desc_embs = torch.tensor(test_data["desc_embs"], dtype=torch.float32)
    test_dna_embs = torch.tensor(test_data["dna_embs"], dtype=torch.float32)
    test_time_embs = torch.tensor(test_data["time_embs"], dtype=torch.float32)
    test_labels = torch.tensor(test_data["labels"], dtype=torch.long)
    
    # The tokenized prompts are already torch tensors.
    train_desc_input_ids = train_data["desc_tokens"]["input_ids"]
    train_desc_attention_mask = train_data["desc_tokens"]["attention_mask"]
    train_dna_input_ids = train_data["dna_tokens"]["input_ids"]
    train_dna_attention_mask = train_data["dna_tokens"]["attention_mask"]
    
    test_desc_input_ids = test_data["desc_tokens"]["input_ids"]
    test_desc_attention_mask = test_data["desc_tokens"]["attention_mask"]
    test_dna_input_ids = test_data["dna_tokens"]["input_ids"]
    test_dna_attention_mask = test_data["dna_tokens"]["attention_mask"]
    
    # Build the TensorDatasets.
    # Order:
    #   0: desc_embs, 1: dna_embs, 2: time_embs,
    #   3: desc_input_ids, 4: desc_attention_mask,
    #   5: dna_input_ids, 6: dna_attention_mask,
    #   7: labels
    train_dataset = TensorDataset(
        train_desc_embs, train_dna_embs, train_time_embs,
        train_desc_input_ids, train_desc_attention_mask,
        train_dna_input_ids, train_dna_attention_mask,
        train_labels
    )
    test_dataset = TensorDataset(
        test_desc_embs, test_dna_embs, test_time_embs,
        test_desc_input_ids, test_desc_attention_mask,
        test_dna_input_ids, test_dna_attention_mask,
        test_labels
    )
    
    # Create stratified train/validation splits from the train dataset.
    indices = np.arange(len(train_dataset))
    train_idx, val_idx = train_test_split(
        indices, test_size=0.1, random_state=42, stratify=train_labels.numpy()
    )
    train_subset = Subset(train_dataset, train_idx)
    val_subset = Subset(train_dataset, val_idx)
    
    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Training on device:", device)
    
    # --------------- Final Model and Training ---------------
    final_learning_rate = 1e-1
    final_model = MOEAttention()  # Your model expects:
    # desc_embs, dna_embs, time_embs, desc_tokenized, dna_tokenized.
    final_model.to(device)
    
    final_optimizer = optim.Adam(final_model.parameters(), lr=final_learning_rate, weight_decay=WEIGHT_DECAY)
    final_criterion = torch.nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.MultiStepLR(final_optimizer, milestones=[2], gamma=0.5)
    
    num_epochs = 2
    
    # --- Call the training routine ---
    best_val_acc, best_val_loss = train_model(
        final_model, train_loader, val_loader, device,
        final_criterion, final_optimizer, scheduler, num_epochs,
        verbose=True, torch=torch, moe=MOE
    )
    print(f"Training complete. Best Val Acc: {best_val_acc:.4f}, Best Val Loss: {best_val_loss:.4f}")
    
    # ------------------- Testing (Evaluation) ---------------------
    final_model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in test_loader:
            # Unpack the batch.
            desc_embs, dna_embs, time_embs, desc_input_ids, desc_attention_mask, dna_input_ids, dna_attention_mask, labels = batch
            
            # Move tensors to device.
            desc_embs = desc_embs.to(device)
            dna_embs = dna_embs.to(device)
            time_embs = time_embs.to(device)
            desc_input_ids = desc_input_ids.to(device)
            desc_attention_mask = desc_attention_mask.to(device)
            dna_input_ids = dna_input_ids.to(device)
            dna_attention_mask = dna_attention_mask.to(device)
            labels = labels.to(device)
            
            # Create tokenized dictionaries.
            desc_tokenized = {"input_ids": desc_input_ids, "attention_mask": desc_attention_mask}
            dna_tokenized = {"input_ids": dna_input_ids, "attention_mask": dna_attention_mask}
            
            # Forward pass.
            if MOE:
                outputs, aux_loss = final_model(desc_embs, dna_embs, time_embs, desc_tokenized, dna_tokenized)
            else:
                outputs, aux_loss = final_model(desc_embs, dna_embs, time_embs, desc_tokenized, dna_tokenized), 0
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
    
    print("Classification Report:\n", classification_report(all_labels, all_preds))
    print("Test ROC AUC: {:.4f}".format(roc_auc_score(all_labels, all_preds)))
    print("Test AUPR: {:.4f}".format(average_precision_score(all_labels, all_preds)))
