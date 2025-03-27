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
    
class GatedMultimodalUnit(nn.Module):
    """
    A GMU variant that fuses three modalities.
    It projects each input, then computes gating coefficients over the three,
    and finally returns a gated sum (with a ReLU activation).
    """
    def __init__(self, size_in1, size_in2, size_in3, size_out):
        super(GatedMultimodalUnit, self).__init__()
        self.linear1 = nn.Linear(size_in1, size_out, bias=False)
        self.linear2 = nn.Linear(size_in2, size_out, bias=False)
        self.linear3 = nn.Linear(size_in3, size_out, bias=False)
        
        # Gate layer: maps the concatenated projections to three scalars
        self.gate_linear = nn.Linear(size_out * 3, 3, bias=False)
        
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x1, x2, x3):
        h1 = self.tanh(self.linear1(x1))
        h2 = self.tanh(self.linear2(x2))
        h3 = self.tanh(self.linear3(x3))
        
        # Concatenate along the feature dimension.
        combined = torch.cat((h1, h2, h3), dim=1)
        # Compute gates (one per modality) and normalize with softmax.
        gates = self.softmax(self.gate_linear(combined))  # shape: (batch, 3)
        g1, g2, g3 = gates[:, 0].unsqueeze(1), gates[:, 1].unsqueeze(1), gates[:, 2].unsqueeze(1)
        
        out = g1 * h1 + g2 * h2 + g3 * h3
        return out 

class GatedMultimodalUnitGeneral(nn.Module):
    """
    A GMU that fuses arbitrary number of modalities.
    It projects each input, then computes gating coefficients over the them,
    and finally returns a gated sum.
    """
    def __init__(self, size_ins=None, size_out=0):
        super(GatedMultimodalUnitGeneral, self).__init__()
        
        if not size_ins: 
            size_ins = []
        
        self.linears = nn.ModuleList([
            nn.Linear(size_in, size_out, bias=False) 
            for size_in in size_ins
        ])
        
        # Gate layer: maps the concatenated projections to three scalars
        self.gate_linear = nn.Linear(size_out * (len(size_ins)), len(size_ins), bias=False)
        
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, *x):
        n = len(x)
        hiddens = [
            self.tanh(self.linears[i](x[i]))
            for i in range(n)

        ]
        # Concatenate along the feature dimension.
        combined = torch.cat([hiddens[i] for i in range(n)], dim=1)
        # compute gates and normalize with softmax.
        gates = self.softmax(self.gate_linear(combined))
        weights = [gates[:, i].unsqueeze(1) for i in range(n)]
        
        out = sum(weights[i] * hiddens[i] for i in range(n))
        return out 
    
class GMUAttention(nn.Module):
    def __init__(self, 
                 text_dim=768,
                 content_dim=384,
                 time_dim=384,
                 emoji_dim=384,
                 hidden_dim=384,
                 clf_hidden_dim=64,
                 num_classes=2,
                 dropout_rate=0.1):
        super(GMUAttention, self).__init__()
        
        # Projection layer for the text modality.
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.relu = nn.ReLU()

        self.dropoutattn1 = nn.Dropout(dropout_rate)
        self.dropoutattn2 = nn.Dropout(dropout_rate)
        self.dropoutffn = nn.Dropout(dropout_rate)
        self.dropout = nn.Dropout(dropout_rate)
        
        self.dna_gmu = GatedMultimodalUnitGeneral(
            size_ins=[content_dim, time_dim, emoji_dim],
            size_out=hidden_dim
            )
        # GMU to fuse content and time modalities.
        # self.dna_gmu = GatedMultimodalUnit(content_dim, time_dim, emoji_dim, hidden_dim)
        # self.pooling_gmu = GatedMultimodalLayer(size_in1=hidden_dim, size_in2=hidden_dim, size_out=hidden_dim)
        
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
        # pre normalizations for each cross-attention input.
        self.text_preln = nn.LayerNorm(hidden_dim)
        self.gmu_out_preln = nn.LayerNorm(hidden_dim)
        self.pre_ffn_ln = nn.LayerNorm(hidden_dim)

        # A feedforward network to further process the concatenated tokens.
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(4 * hidden_dim, hidden_dim)
        )
        # self.layer_norm_ffn = nn.LayerNorm(hidden_dim)
        
        # Classification head.
        self.clf_head = nn.Sequential(
            nn.Linear(hidden_dim, clf_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(clf_hidden_dim, num_classes)
        )

    def forward(self, text, content, time, emoji):
        # Project the text modality.
        text_emb = self.relu(self.text_proj(text))
        # Fuse content and time via the GMU.
        gmu_out = self.dna_gmu(content, time, emoji)

        # Prepare tokens for cross-attention by adding a sequence dimension.
        text_token = text_emb.unsqueeze(1)      # shape: (batch, 1, hidden_dim)
        gmu_token = gmu_out.unsqueeze(1)        # shape: (batch, 1, hidden_dim)
        text_tok_normed = self.text_preln(text_token)    # text_preln
        gmu_tok_normed = self.gmu_out_preln(gmu_token)   # gmu_out_preln

        # First cross-attention: text as query, GMU output as key/value.
        attn1, _ = self.cross_attn_text_query(query=text_tok_normed,
                                              key=gmu_tok_normed,
                                              value=gmu_tok_normed)
        attn1 = text_token + self.dropoutattn1(attn1) 
        
        # Second cross-attention: GMU output as query, text as key/value.
        attn2, _ = self.cross_attn_gmu_query(query=gmu_tok_normed,
                                             key=text_tok_normed,
                                             value=text_tok_normed)
        attn2 = gmu_token + self.dropoutattn2(attn2)

        # Concatenate the outputs along the sequence dimension.
        combined = torch.cat([attn1, attn2], dim=1)  # shape: (batch, 2, hidden_dim)


        # Process with an FFN (with residual connection).
        ffn_out = self.ffn(
            self.pre_ffn_ln(combined)
        )
        ffn_out = self.relu(ffn_out)
        ffn_out = combined + self.dropout(ffn_out)

        # Global average pooling over the sequence dimension.
        fused = ffn_out.mean(dim=1)
        fused = self.dropout(fused)
        
        # Classification head.
        logits = self.clf_head(fused)
        return logits