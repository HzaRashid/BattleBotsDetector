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

        # self.dropout = nn.Dropout(p=0.1)

    def forward(self, x1, x2):
        h1 = self.tanh_f(self.hidden1(x1))
        h2 = self.tanh_f(self.hidden2(x2))  # Use hidden2 for modality 2.
        combined = torch.cat((h1, h2), dim=1)
        z = self.sigmoid_f(self.hidden_sigmoid(combined))
        
        # Fuse the two modalities based on the gate.
        out = z * h1 + (1 - z) * h2
        out = self.relu(out)
        return out


class GMUAttention(nn.Module):
    def __init__(self, 
                 text_dim=768,
                 raw_content_channels=1280,  # Number of channels in raw content DNA features.
                 raw_time_channels=1280,     # Number of channels in raw time DNA features.
                 content_out_dim=384,        # Desired output dim for content modality.
                 time_out_dim=384,           # Desired output dim for time modality.
                 hidden_dim=384,
                 ffn_hidden_dim=64,
                 num_classes=2,
                 dropout_rate=0.1):
        super(GMUAttention, self).__init__()
        
        # Projection layer for the text modality.
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        
        # Learnable pooling & projection for the content DNA modality.
        self.content_pool_proj = nn.Sequential(
            nn.Conv2d(raw_content_channels, content_out_dim, kernel_size=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))  # Pool to (1,1) spatially.
        )
        
        # Learnable pooling & projection for the time DNA modality.
        self.time_pool_proj = nn.Sequential(
            nn.Conv2d(raw_time_channels, time_out_dim, kernel_size=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # GMU to fuse the processed content and time modalities.
        self.gmu = GatedMultimodalLayer(content_out_dim, time_out_dim, hidden_dim)
        
        # Cross-attention layers.
        self.cross_attn_text_query = nn.MultiheadAttention(embed_dim=hidden_dim,
                                                             num_heads=6,
                                                             dropout=dropout_rate,
                                                             batch_first=True)
        self.cross_attn_gmu_query = nn.MultiheadAttention(embed_dim=hidden_dim,
                                                            num_heads=6,
                                                            dropout=dropout_rate,
                                                            batch_first=True)
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.pre_ffn_ln = nn.LayerNorm(hidden_dim)
        
        # Feedforward network.
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(4 * hidden_dim, hidden_dim)
        )
        
        # Classification head.
        self.clf_head = nn.Sequential(
            nn.Linear(hidden_dim, ffn_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(ffn_hidden_dim, num_classes)
        )

    def forward(self, text, raw_content, raw_time):
        # Process the text modality.
        text_emb = self.relu(self.text_proj(text))  # (batch, hidden_dim)
        
        # Process content DNA modality.
        # raw_content: (batch, raw_content_channels, H, W)
        content_features = self.content_pool_proj(raw_content)  # -> (batch, content_out_dim, 1, 1)
        content_features = content_features.view(content_features.size(0), -1)  # (batch, content_out_dim)
        
        # Process time DNA modality.
        # raw_time: (batch, raw_time_channels, H, W)
        time_features = self.time_pool_proj(raw_time)  # -> (batch, time_out_dim, 1, 1)
        time_features = time_features.view(time_features.size(0), -1)  # (batch, time_out_dim)
        
        # Fuse content and time modalities using the GMU.
        gmu_out = self.gmu(content_features, time_features)
        
        # Prepare tokens for cross-attention.
        text_token = text_emb.unsqueeze(1)   # (batch, 1, hidden_dim)
        gmu_token = gmu_out.unsqueeze(1)       # (batch, 1, hidden_dim)
        text_tok_normed = self.layer_norm1(text_token)
        gmu_tok_normed = self.layer_norm2(gmu_token)
        
        # Cross-attention: text as query, GMU output as key/value.
        attn1, _ = self.cross_attn_text_query(query=text_tok_normed,
                                              key=gmu_tok_normed,
                                              value=gmu_tok_normed)
        attn1 = text_token + self.dropout(attn1)
        
        # Cross-attention: GMU output as query, text as key/value.
        attn2, _ = self.cross_attn_gmu_query(query=gmu_tok_normed,
                                             key=text_tok_normed,
                                             value=text_tok_normed)
        attn2 = gmu_token + self.dropout(attn2)
        
        # Concatenate the outputs along the sequence dimension.
        combined = torch.cat([attn1, attn2], dim=1)  # (batch, 2, hidden_dim)
        ffn_out = self.ffn(self.pre_ffn_ln(combined))
        ffn_out = self.relu(ffn_out)
        ffn_out = combined + self.dropout(ffn_out)
        
        # Global average pooling over the sequence dimension.
        fused = ffn_out.mean(dim=1)
        fused = self.dropout(fused)
        
        # Classification head.
        logits = self.clf_head(fused)
        return logits