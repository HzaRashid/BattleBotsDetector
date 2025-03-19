import torch
import torch.nn as nn
import torch.nn.functional as F
# -------------------------
# GMU-based Fusion Model
# -------------------------
class GMUFusionWithNorm(nn.Module):
    def __init__(self, 
                 text_dim=768,
                 content_dna_dim=384,
                 time_dna_dim=256,
                 hidden_dim=256,
                 ffn_hidden_dim=128,
                 num_classes=2,
                 ffn_dropout_rate=0.1,
                 fused_dropout_rate=0.1
                 ):
        super(GMUFusionWithNorm, self).__init__()

        self.text_dim = text_dim
        self.content_dna_dim = content_dna_dim
        self.time_dna_dim = time_dna_dim
        # Modality-specific projections.
        self.text_linear = nn.Linear(text_dim, hidden_dim)
        self.text_norm = nn.LayerNorm(hidden_dim)
        self.content_linear = nn.Linear(content_dna_dim, hidden_dim)
        self.content_norm = nn.LayerNorm(hidden_dim)
        self.time_linear = nn.Linear(time_dna_dim, hidden_dim)
        self.time_norm = nn.LayerNorm(hidden_dim)
        
        # Gating layers for each modality.
        self.text_gate = nn.Linear(text_dim, hidden_dim)
        self.content_gate = nn.Linear(content_dna_dim, hidden_dim)
        self.time_gate = nn.Linear(time_dna_dim, hidden_dim)
        
        # Normalization after fusion.
        self.fused_norm = nn.LayerNorm(hidden_dim)
        
        self.activation = nn.ReLU()
        self.fused_dropout = nn.Dropout(p=fused_dropout_rate)
        
        # Optional forget gate: takes the fused representation and computes a forget signal
        self.forget_gate = nn.Linear(hidden_dim, hidden_dim)
        # FFN head to produce logits.
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=ffn_dropout_rate),
            nn.Linear(ffn_hidden_dim, num_classes)
        )
    
    def forward(self, x):

        text_end = self.text_dim
        contend_end = self.text_dim + self.content_dna_dim
        # Split the concatenated input into three modalities.
        text_embedding = x[:, :text_end]
        content_dna_embedding = x[:, text_end:contend_end]
        time_dna_embedding = x[:, contend_end:]        
        
        # Compute modality-specific representations with normalization.
        h_text = self.activation(self.text_norm(self.text_linear(text_embedding)))
        h_content = self.activation(self.content_norm(self.content_linear(content_dna_embedding)))
        h_time = self.activation(self.time_norm(self.time_linear(time_dna_embedding)))
        
        # Compute gating scores.
        g_text = self.text_gate(text_embedding)
        g_content = self.content_gate(content_dna_embedding)
        g_time = self.time_gate(time_dna_embedding)
        
        # Stack gate scores and apply softmax.
        gates = torch.stack([g_text, g_content, g_time], dim=-1)
        gates = F.relu(gates)
        
        # Fuse modalities via a weighted sum.
        fused = gates[..., 0] * h_text + gates[..., 1] * h_content + gates[..., 2] * h_time
        
        # Apply dropout and normalization on the fused representation.
        # fused = self.fused_dropout(fused)
        fused = self.fused_norm(fused)

        # Pass through the FFN head.
        logits = self.ffn(fused)
        return logits
    



class GateMLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=None):
        super(GateMLP, self).__init__()
        if hidden_dim is None:
            hidden_dim = output_dim // 2  # you can adjust this as needed
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.net(x)
    

class GMUFusionWithGlobalSkip(nn.Module):
    def __init__(self, 
                 text_dim=768,
                 content_dna_dim=384,
                 time_dna_dim=256,
                 hidden_dim=256,
                 ffn_hidden_dim=128,
                 num_classes=2,
                 ffn_dropout_rate=0.1,
                 fused_dropout_rate=0.1,
                 dropout_text=0.25,
                 dropout_cdna=0.125,
                 dropout_tdna=0.125):
        super(GMUFusionWithGlobalSkip, self).__init__()

        # Modality-specific projections.
        self.text_linear = nn.Linear(text_dim, hidden_dim)
        self.text_norm = nn.LayerNorm(hidden_dim)

        self.content_linear = nn.Linear(content_dna_dim, hidden_dim)
        self.content_norm = nn.LayerNorm(hidden_dim)

        self.time_linear = nn.Linear(time_dna_dim, hidden_dim)
        self.time_norm = nn.LayerNorm(hidden_dim)
        # ----------------------------------------------------------

        # Gating layers replaced with GateMLP for more expressive power.
        self.text_gate = GateMLP(text_dim, hidden_dim)
        self.content_gate = GateMLP(content_dna_dim, hidden_dim)
        self.time_gate = GateMLP(time_dna_dim, hidden_dim)
        
        # Global skip: projects concatenated raw modality embeddings to hidden_dim.
        self.global_skip = nn.Linear(text_dim + content_dna_dim + time_dna_dim, hidden_dim)
        
        # Normalization and dropout.
        self.fused_norm = nn.LayerNorm(hidden_dim)
        self.activation = nn.ReLU()
        self.fused_dropout = nn.Dropout(p=fused_dropout_rate)

        self.text_dropout = nn.Dropout(p=dropout_text)
        self.cdna_dropout = nn.Dropout(p=dropout_cdna)
        self.tdna_dropout = nn.Dropout(p=dropout_tdna)
        self.dropout = nn.Dropout(p=0.1)
        
        # FFN head.
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=ffn_dropout_rate),
            nn.Linear(ffn_hidden_dim, num_classes)
        )

    def forward(self, x):
        # Split concatenated input into three modalities.
        text_end = self.text_linear.in_features
        content_end = text_end + self.content_linear.in_features
        
        text_embedding = x[:, :text_end]
        content_embedding = x[:, text_end:content_end]
        time_embedding = x[:, content_end:]
        
        # Compute modality-specific representations.
        h_text = self.dropout(self.text_norm(self.activation(self.text_linear(text_embedding))))
        h_content = self.dropout(self.content_norm(self.activation(self.content_linear(content_embedding))))
        h_time = self.dropout(self.time_norm(self.activation(self.time_linear(time_embedding))))
        
        # Compute gating scores using MLPs.
        g_text = self.text_gate(text_embedding)
        g_content = self.content_gate(content_embedding)
        g_time = self.time_gate(time_embedding)
        
        # Stack and apply softmax for proper weighting.
        gates = torch.stack([g_text, g_content, g_time], dim=-1)
        gates = torch.relu(gates)
        
        # Fuse modalities via weighted sum.
        fused = gates[..., 0] * h_text + gates[..., 1] * h_content + gates[..., 2] * h_time
        
        # Global skip: combine all raw modality embeddings.
        global_residual = self.global_skip(torch.cat([text_embedding, content_embedding, time_embedding], dim=1))
        
        # Add the global residual, apply normalization and dropout.
        fused = self.fused_norm(fused) + global_residual
        fused = self.fused_dropout(fused)
        
        # Pass through FFN head.
        logits = self.ffn(fused)
        return logits