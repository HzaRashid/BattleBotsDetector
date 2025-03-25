import torch
import torch.nn as nn

class GatedMultimodalLayer6(nn.Module):
    """
    A GMU variant that fuses six modality representations.
    Each input is first projected and activated with tanh.
    Then, the six projected representations are concatenated and passed
    through a gate layer (with softmax) to compute a weighted sum.
    """
    def __init__(self, size_in, size_out, num_modalities=6):
        super(GatedMultimodalLayer6, self).__init__()
        self.num_modalities = num_modalities
        self.linears = nn.ModuleList([
            nn.Linear(size_in, size_out, bias=False) for _ in range(num_modalities)
        ])
        self.gate_linear = nn.Linear(size_out * num_modalities, num_modalities, bias=False)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.ReLU()
        
    def forward(self, *inputs):
        hs = []
        for i, x in enumerate(inputs):
            h = self.tanh(self.linears[i](x))
            hs.append(h)
        # Concatenate along the feature dimension.
        combined = torch.cat(hs, dim=1)  # shape: (batch, num_modalities * size_out)
        gates = self.softmax(self.gate_linear(combined))  # shape: (batch, num_modalities)
        # Weighted sum of the modality projections.
        output = sum(gates[:, i].unsqueeze(1) * hs[i] for i in range(self.num_modalities))
        return self.relu(output)

class GCBAN(nn.Module):
    """
    Revised multi-modal fusion model with bi-directional cross-attention for each modality pair.
    
    Steps:
      1. Split the input into three modalities: text, content, and time.
      2. Project each modality to a shared hidden dimension.
      3. For each pair (text-content, text-time, content-time), compute cross-attention
         in both directions (6 attention layers total), with residual connections.
      4. Fuse the six outputs using a 6-input GMU.
      5. Pass the fused representation to the classification head.
    """
    def __init__(self, 
                 text_dim=768,
                 content_dim=384,
                 time_dim=384,
                 hidden_dim=256,
                 num_classes=2,
                 dropout_rate=0.1):
        super(GCBAN, self).__init__()


        self.text_dim = text_dim
        self.content_dim = content_dim
        self.time_dim = time_dim

        # Projection layers for each modality.
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.content_proj = nn.Linear(content_dim, hidden_dim)
        self.time_proj = nn.Linear(time_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        
        # Layer Normalizations.
        self.ln_text = nn.LayerNorm(hidden_dim)
        self.ln_content = nn.LayerNorm(hidden_dim)
        self.ln_time = nn.LayerNorm(hidden_dim)
        

        self.num_heads = 3
        # Define six cross-attention layers (bi-directional for each pair).
        # Text-Content pair.
        self.cross_attn_text_query_content = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=self.num_heads,
            dropout=dropout_rate,
            batch_first=True
        )
        self.cross_attn_content_query_text = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=self.num_heads,
            dropout=dropout_rate,
            batch_first=True
        )
        # Text-Time pair.
        self.cross_attn_text_query_time = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=self.num_heads,
            dropout=dropout_rate,
            batch_first=True
        )
        self.cross_attn_time_query_text = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=self.num_heads,
            dropout=dropout_rate,
            batch_first=True
        )
        # Content-Time pair.
        self.cross_attn_content_query_time = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=self.num_heads,
            dropout=dropout_rate,
            batch_first=True
        )
        self.cross_attn_time_query_content = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=self.num_heads,
            dropout=dropout_rate,
            batch_first=True
        )
        
        # GMU to fuse six cross-attention outputs.
        self.gmu6 = GatedMultimodalLayer6(hidden_dim, hidden_dim, num_modalities=6)
        
        # Classification head.
        self.clf_head = nn.Sequential(
            nn.Linear(hidden_dim, 3 * hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(3 * hidden_dim, num_classes)
        )
        
    def forward(self, x):
        # x shape: (batch, text_dim + content_dim + time_dim)
        # Split the input into modalities.
        text = x[:, :self.time_dim]
        content = x[:, self.time_dim:self.time_dim+self.content_dim]
        time = x[:, self.time_dim+self.content_dim:]
        
        # Project each modality.
        text_emb = self.relu(self.text_proj(text))
        content_emb = self.relu(self.content_proj(content))
        time_emb = self.relu(self.time_proj(time))
        
        # Add a sequence dimension (each modality becomes a single token).
        text_token = text_emb.unsqueeze(1)     # shape: (batch, 1, hidden_dim)
        content_token = content_emb.unsqueeze(1) # shape: (batch, 1, hidden_dim)
        time_token = time_emb.unsqueeze(1)       # shape: (batch, 1, hidden_dim)
        
        # Apply layer normalization.
        normed_text = self.ln_text(text_token)
        normed_content = self.ln_content(content_token)
        normed_time = self.ln_time(time_token)
        
        # -------------------------------
        # Bi-directional Cross-Attention:
        # -------------------------------
        # Text-Content pair.
        attn_text_content, _ = self.cross_attn_text_query_content(
            query=normed_text,
            key=normed_content,
            value=normed_content
        )
        attn_text_content = text_token + self.dropout(attn_text_content)
        
        attn_content_text, _ = self.cross_attn_content_query_text(
            query=normed_content,
            key=normed_text,
            value=normed_text
        )
        attn_content_text = content_token + self.dropout(attn_content_text)
        
        # Text-Time pair.
        attn_text_time, _ = self.cross_attn_text_query_time(
            query=normed_text,
            key=normed_time,
            value=normed_time
        )
        attn_text_time = text_token + self.dropout(attn_text_time)
        
        attn_time_text, _ = self.cross_attn_time_query_text(
            query=normed_time,
            key=normed_text,
            value=normed_text
        )
        attn_time_text = time_token + self.dropout(attn_time_text)
        
        # Content-Time pair.
        attn_content_time, _ = self.cross_attn_content_query_time(
            query=normed_content,
            key=normed_time,
            value=normed_time
        )
        attn_content_time = content_token + self.dropout(attn_content_time)
        
        attn_time_content, _ = self.cross_attn_time_query_content(
            query=normed_time,
            key=normed_content,
            value=normed_content
        )
        attn_time_content = time_token + self.dropout(attn_time_content)
        
        # Remove the sequence dimension (squeeze from (batch, 1, hidden_dim) to (batch, hidden_dim)).
        attn_text_content = attn_text_content.squeeze(1)
        attn_content_text = attn_content_text.squeeze(1)
        attn_text_time = attn_text_time.squeeze(1)
        attn_time_text = attn_time_text.squeeze(1)
        attn_content_time = attn_content_time.squeeze(1)
        attn_time_content = attn_time_content.squeeze(1)
        
        # Fuse the six attention outputs using the 6-input GMU.
        fused = self.gmu6(
            attn_text_content, 
            attn_content_text, 
            attn_text_time, 
            attn_time_text, 
            attn_content_time, 
            attn_time_content
        )
        fused = self.dropout(fused)
        
        # Classification head.
        logits = self.clf_head(fused)
        return logits
