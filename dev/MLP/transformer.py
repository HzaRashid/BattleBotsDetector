import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionFusion(nn.Module):
    def __init__(self, 
                 text_dim=768,
                 content_dim=384,
                 time_dim=384,
                 embed_dim=768,
                 dropout_rate_dna=0.1,
                 dropout_rate_all=0.1
                 ):
        super(AttentionFusion, self).__init__()
        
        # dims
        self.text_dim = text_dim
        self.content_dim = content_dim
        # -----------------------------------------------

        # DNA attention
        self.DNA_MHA = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=4, 
                                         dropout=dropout_rate_dna, batch_first=True)
        
        # DNA normalize
        self.layer_norm_DNA = nn.LayerNorm(embed_dim)
        # DNA dropout
        self.dropout_DNA = nn.Dropout(p=dropout_rate_dna)
        # -----------------------------------------------

        self.ALL_MHA = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=4, 
                                         dropout=dropout_rate_all, batch_first=True)
        
        self.layer_norm_ALL = nn.LayerNorm(embed_dim)
        self.dropout_ALL = nn.Dropout(p=dropout_rate_all)
        # ----------------------------------------------

        # activation
        self.relu = nn.ReLU()
        # ----------------------------------------------

        # dense 2nd last layer
        self.fc = nn.Linear(embed_dim, 128)
        self.out = nn.Linear(128, 2)


    def forward(self, x):
        text_end = self.text_dim
        content_end = text_end + self.content_dim
        
        # Split input for each modality.
        text = x[:, :text_end]
        content = x[:, text_end:content_end]
        time = x[:, content_end:]
        
        # Project each modality into the shared hidden space.
        
        # self attention on dna embeddings
        token1 = torch.cat([content, time], dim=1)
        token2 = torch.cat([time, content], dim=1)
        tokens_DNA = torch.stack([token1, token2], dim=1)  # shape: (batch_size, 2, 768)
        attn_out_DNA, _ = self.DNA_MHA(tokens_DNA, tokens_DNA, tokens_DNA)
        # residual connection and layer normalization
        tokens_DNA = self.layer_norm_DNA(attn_out_DNA + tokens_DNA) 
        # Aggregate tokens (using mean pooling) to form a fused representation.
        fused_DNA = tokens_DNA.mean(dim=1)
        fused_DNA = self.dropout_DNA(fused_DNA)
        # -------------------------------------------------------------------

        # self attention on text + attn_out of dna embeddings
        tokens_ALL = torch.stack([text, fused_DNA], dim=1)
        attn_out_ALL, _ = self.ALL_MHA(tokens_ALL, tokens_ALL, tokens_ALL)
        # residual connection and layer normalization
        tokens_ALL = self.layer_norm_ALL(attn_out_ALL + tokens_ALL)
        fused_ALL = tokens_ALL.mean(dim=1)
        fused_ALL = self.dropout_ALL(fused_ALL)
        # -------------------------------------------------------------------
        
        fc = self.fc(fused_ALL)
        fc = self.relu(fc)
        fc = self.dropout_ALL(fc)

        logits = self.out(fc)

        return logits
