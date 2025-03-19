import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiModalAttentionFusion(nn.Module):
    def __init__(self, 
                 text_dim=768,
                 content_dim=384,
                 time_dim=256,
                 hidden_dim=256,
                 ffn_hidden_dim=128,
                 num_classes=2,
                 dropout_rate=0.1,
                 num_heads=4):
        super(MultiModalAttentionFusion, self).__init__()
        
        # Projection layers for each modality.
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.content_proj = nn.Linear(content_dim, hidden_dim)
        self.time_proj = nn.Linear(time_dim, hidden_dim)
        
        # Multi-Head Attention layer.
        # We use batch_first=True so that inputs have shape (batch_size, tokens, hidden_dim)
        self.mha = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4, 
                                         dropout=dropout_rate, batch_first=True)
        self.attn_norm = nn.LayerNorm(hidden_dim)
        
        # FFN head for classification.
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(ffn_hidden_dim, num_classes)
        )
        
        self.dropout = nn.Dropout(dropout_rate)
        self.activation = nn.ReLU()

    def forward(self, x):
        # Assume x is a concatenated input from three modalities:
        # first text_dim for text, then content_dim for content, and time_dim for time.
        text_end = self.text_proj.in_features  # text_dim
        content_end = text_end + self.content_proj.in_features  # text_dim + content_dim
        
        # Split input for each modality.
        text = x[:, :text_end]
        content = x[:, text_end:content_end]
        time = x[:, content_end:]
        
        # Project each modality into the shared hidden space.
        text_emb = self.activation(self.text_proj(text))
        content_emb = self.activation(self.content_proj(content))
        time_emb = self.activation(self.time_proj(time))
        
        # Stack embeddings into tokens with shape: (batch_size, 3, hidden_dim)
        tokens = torch.stack([text_emb, content_emb, time_emb], dim=1)
        
        # Process tokens with multiheaded self-attention.
        # Here tokens serve as query, key, and value.
        attn_out, _ = self.mha(tokens, tokens, tokens)
        # Apply residual connection and layer normalization.
        tokens = self.attn_norm(self.dropout(attn_out) + tokens) 
        
        # Aggregate tokens (using mean pooling) to form a fused representation.
        fused = tokens.mean(dim=1)
        fused = self.dropout(fused)
        
        # Pass the fused representation through the FFN for classification.
        logits = self.ffn(fused)
        return logits
