import os
import torch
import random
import numpy as np
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, models

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


# -------------------------
# Main Pipeline
# -------------------------
if __name__ == "__main__":
    from load_tensors import build_datasets
    from train_model import Trainer

    set_seed(42)
    cur_dir = os.path.dirname(__file__)
    data_dir = os.path.join(cur_dir, "../data")

    # twhin-bert for sentence (user description) embeddings
    transformer_model = models.Transformer("Twitter/twhin-bert-base", 
                                           model_args={'attn_implementation': 'eager'})
    pooling_model = models.Pooling(transformer_model.get_word_embedding_dimension(), 
                                   pooling_mode_mean_tokens=True)
    st_model = SentenceTransformer(modules=[transformer_model, pooling_model])

    # --------------------------------------------
    
    # Build datasets with train/val/test splits
    train_ds, val_ds, test_ds = build_datasets(
        data_dir, session_numbers=[], xnums=[1], st_model=st_model
    )

    # Create DataLoaders
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Training on device:", device) 
    
    
    # -------------------- Final Model and Training --------------------
    MoETrainer = Trainer()

    val_acc, val_loss = MoETrainer.train(train_loader=train_loader, val_loader=val_loader,
                                          verbose=True, unfreeze_threshold=1.0)
                                          
    _, test_acc, _, _, _ = MoETrainer.evaluate_model(test_loader, verbose=True)
    