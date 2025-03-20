import torch
import torch.nn as nn

class GatedMultimodalLayer(nn.Module):
    """Gated Multimodal Layer based on 'Gated multimodal networks, Arevalo et al.' (https://arxiv.org/abs/1702.01992)"""
    def __init__(self, size_in1, size_in2, size_out):
        super(GatedMultimodalLayer, self).__init__()
        self.size_in1, self.size_in2, self.size_out = size_in1, size_in2, size_out
        
        self.hidden1 = nn.Linear(size_in1, size_out, bias=False)
        self.hidden2 = nn.Linear(size_in2, size_out, bias=False)
        self.hidden_sigmoid = nn.Linear(size_out * 2, 1, bias=False)

        # Activation functions
        self.tanh_f = nn.Tanh()
        self.relu = nn.ReLU()
        self.sigmoid_f = nn.Sigmoid()

        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x1, x2):
        h1 = self.tanh_f(self.hidden1(x1))
        h2 = self.tanh_f(self.hidden2(x2))  # Use hidden2 for modality 2.
        combined = torch.cat((h1, h2), dim=1)
        z = self.sigmoid_f(self.hidden_sigmoid(combined))
        
        # Fuse the two modalities based on the gate.
        out = z * h1 + (1 - z) * h2
        out = self.relu(out)
        return out

class NewMultiModalAttentionFusion(nn.Module):
    def __init__(self, 
                 text_dim=768,
                 content_dim=384,
                 time_dim=384,
                 hidden_dim=384,
                 ffn_hidden_dim=64,
                 num_classes=2,
                 dropout_rate=0.1):
        super(NewMultiModalAttentionFusion, self).__init__()

        # Save modality dimensions for splitting the input.
        self.text_dim = text_dim
        self.content_dim = content_dim
        self.time_dim = time_dim
        
        # Projection layer for the text modality.
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

        # GMU to fuse content and time modalities.
        self.gmu = GatedMultimodalLayer(content_dim, time_dim, hidden_dim)
        
        # Cross-attention layer: text as query, GMU output as key/value.
        self.cross_attn_text_query = nn.MultiheadAttention(embed_dim=hidden_dim,
                                                             num_heads=6,
                                                             dropout=dropout_rate,
                                                             batch_first=True)
        # Cross-attention layer: GMU output as query, text as key/value.
        self.cross_attn_gmu_query = nn.MultiheadAttention(embed_dim=hidden_dim,
                                                            num_heads=6,
                                                            dropout=dropout_rate,
                                                            batch_first=True)
        # Layer normalizations for each cross-attention output.
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        
        # A feedforward network to further process the concatenated tokens.
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(4 * hidden_dim, hidden_dim)
        )
        self.layer_norm_ffn = nn.LayerNorm(hidden_dim)
        
        # Classification head.
        self.clf_head = nn.Sequential(
            nn.Linear(hidden_dim, ffn_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(ffn_hidden_dim, num_classes)
        )

    def forward(self, x):
        # Split the concatenated input into modalities.
        # x shape: (batch, text_dim + content_dim + time_dim)
        text = x[:, :self.text_dim]
        content = x[:, self.text_dim:self.text_dim + self.content_dim]
        time = x[:, self.text_dim + self.content_dim:]
        
        # Project the text modality.
        text_emb = self.relu(self.text_proj(text))
        
        # Fuse content and time via the GMU.
        gmu_out = self.gmu(content, time)
        
        # Prepare tokens for cross-attention by adding a sequence dimension.
        text_token = text_emb.unsqueeze(1)  # shape: (batch, 1, hidden_dim)
        gmu_token = gmu_out.unsqueeze(1)      # shape: (batch, 1, hidden_dim)
        
        # First cross-attention: text as query, GMU output as key/value.
        torch.manual_seed(42)
        attn1, _ = self.cross_attn_text_query(query=text_token,
                                              key=gmu_token,
                                              value=gmu_token)
        attn1 = self.layer_norm1(text_token + self.dropout(attn1))
        
        # Second cross-attention: GMU output as query, text as key/value.
        torch.manual_seed(42)
        attn2, _ = self.cross_attn_gmu_query(query=gmu_token,
                                             key=text_token,
                                             value=text_token)
        attn2 = self.layer_norm2(gmu_token + self.dropout(attn2))
        
        # Concatenate the outputs along the sequence dimension.
        combined = torch.cat([attn1, attn2], dim=1)  # shape: (batch, 2, hidden_dim)
        
        # Process with an FFN (with residual connection).
        ffn_out = self.ffn(combined)
        ffn_out = self.relu(ffn_out)
        ffn_out = self.layer_norm_ffn(combined + self.dropout(ffn_out))
        
        # Global average pooling over the sequence dimension.
        fused = ffn_out.mean(dim=1)
        fused = self.dropout(fused)
        
        # Classification head.
        torch.manual_seed(42)
        logits = self.clf_head(fused)
        return logits
