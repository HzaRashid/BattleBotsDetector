import os
import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

# seeds for reproducability
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seed(42)

class MultiModalAttentionFusion(nn.Module):
    def __init__(self, 
                 text_dim=768,
                 content_dim=384,
                 time_dim=384,
                 hidden_dim=384,
                 ffn_hidden_dim=64,
                 num_classes=2,
                 dropout_rate=0.1
                 ):
        super(MultiModalAttentionFusion, self).__init__()

        # get dims
        self.text_dim = text_dim
        self.content_dim = content_dim
        self.time_dim = time_dim
        # -------------------------------------------------

        # Projection layer for text
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        # --------------------------------------------------
        # Multi-Head Attention layer.
        # MHA layer: use batch_first=True; input shape == (batch_size, tokens, hidden_dim)
        self.mha = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4, 
                                         dropout=dropout_rate, batch_first=True)
        self.input_layer_norm = nn.LayerNorm(hidden_dim)
        self.post_attn_layer_norm = nn.LayerNorm(hidden_dim)
        
        # ffn head for classification.
        self.clf_head = nn.Sequential(
            nn.Linear(hidden_dim, ffn_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(ffn_hidden_dim, num_classes)
        )
        
        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()


        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, 4*hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(4*hidden_dim, hidden_dim)
        )

    def forward(self, x):
        # Assume x is a concatenated input from three modalities:
        # first text_dim for text, then content_dim for content, and time_dim for time.
        text_end = self.text_dim  
        content_end = text_end + self.content_dim
        
        # Split input for each modality.
        text = x[:, :text_end]
        content = x[:, text_end:content_end]
        time = x[:, content_end:]
        
        # Project each modality into the shared hidden space.
        text_emb = self.relu(self.text_proj(text))
        content_emb = content
        time_emb = time
        
        # Stack embeddings into tokens with shape: (batch_size, 3, hidden_dim)
        tokens = torch.stack([text_emb, content_emb, time_emb], dim=1)
        normed_tokens = self.input_layer_norm(tokens)
        
        # Process tokens with multiheaded self-attention.
        # Here tokens serve as query, key, and value.
        torch.manual_seed(42)
        attn_out, _ = self.mha(normed_tokens, normed_tokens, normed_tokens)
        """ new """
        attn_out = self.relu(attn_out) 
        # Apply residual connection and layer normalization.
        torch.manual_seed(42)
        attn_out_plus_tokens = self.dropout(attn_out) + tokens
        # this would be the input to the optional sub_ffn
        normed_attn_out_plus_tokens = self.post_attn_layer_norm(attn_out_plus_tokens)

        # -------- ffn ----------
        # torch.manual_seed(42)
        # ffn_out = self.ffn(normed_attn_out_plus_tokens)
        # """ new """
        # ffn_out = self.relu(ffn_out)
        # torch.manual_seed(42)
        # ffn_out = self.dropout(ffn_out) + attn_out_plus_tokens
        # -----------------------

        # Aggregate tokens (using mean pooling) to form a fused representation.
        fused = normed_attn_out_plus_tokens.mean(dim=1)
        """ new """
        fused = self.relu(fused)
        torch.manual_seed(42)
        fused = self.dropout(fused)
        
        # Pass the fused representation through the FFN for classification.
        torch.manual_seed(42)
        logits = self.clf_head(fused)
        return logits
