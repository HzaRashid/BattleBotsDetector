import torch
import torch.nn as nn
import torch.nn.functional as F
from moe import MixtureOfExperts

class FeatureAlign(nn.Module):
    def __init__(self,align_size):
        super(FeatureAlign,self).__init__()

        self.linear_relu_des=nn.Sequential(
            nn.Linear(768,int(align_size)),
            nn.LeakyReLU(),
            # nn.Dropout(0.1),
            nn.Linear(int(align_size),int(align_size)),
        )
        self.linear_relu_tweet=nn.Sequential(
            nn.Linear(768,int(align_size)),
            nn.LeakyReLU(),
            # nn.Dropout(0.1),
            nn.Linear(int(align_size),int(align_size)),
        )
        self.linear_content_dna=nn.Sequential(
            nn.Linear(640,int(align_size)),
            nn.LeakyReLU(),
            # nn.Dropout(0.1),
            nn.Linear(int(align_size),int(align_size)),
        )
        self.linear_time_dna=nn.Sequential(
            nn.Linear(640,int(align_size)),
            nn.LeakyReLU(),
            # nn.Dropout(0.1),
            nn.Linear(int(align_size),int(align_size)),
        )



        self.linear_text=nn.Sequential(
            nn.Linear(768*2,int(align_size)),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(int(align_size),int(align_size)),
        )


        self.linear_dna=nn.Sequential(
            nn.Linear(2*640,int(align_size)),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(int(align_size),int(align_size)),
        )

    def forward(self, des_tensor,tweets_tensor,content_dna_tensor,time_dna_tensor):
        # des_tensor,tweets_tensor,content_dna_tensor,time_dna_tensor=self.linear_relu_des(des_tensor),self.linear_relu_tweet(tweets_tensor),\
        #                                                 self.linear_content_dna(content_dna_tensor),self.linear_time_dna(time_dna_tensor)

        text, dna = self.linear_text(torch.cat((des_tensor, tweets_tensor), dim=1)),\
        self.linear_dna(torch.cat((content_dna_tensor,time_dna_tensor), dim=1))
        return text,dna
    
# FixedPooling as provided.
class FixedPooling(nn.Module):
    def __init__(self, fixed_size):
        super().__init__()
        self.fixed_size = fixed_size

    def forward(self, x):
        b, w, h = x.shape
        p_w = self.fixed_size * ((w + self.fixed_size - 1) // self.fixed_size) - w
        p_h = self.fixed_size * ((h + self.fixed_size - 1) // self.fixed_size) - h
        x = F.pad(x, (0, p_h, 0, p_w))
        pool_size = (((w + self.fixed_size - 1) // self.fixed_size),
                     ((h + self.fixed_size - 1) // self.fixed_size))
        pool = nn.MaxPool2d(pool_size, stride=pool_size)
        return pool(x)

# LModel as provided.
class LModel(nn.Module):
    def __init__(self, embed_dim=768, num_heads=2, 
                 norm_first=True):
        super(LModel, self).__init__()
        self.multihead_attention = nn.MultiheadAttention(embed_dim=embed_dim, 
                                                         num_heads=num_heads,
                                                         dropout=0.1, 
                                                         batch_first=True)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(p=0.1)

    def forward(self, text_src):
        if self.norm_first:
            text, attention_weight = self._sa_block(self.norm1(text_src))
        else:
            text, attention_weight = self._sa_block(text_src)
            text = self.norm1(text_src + text)
        return text, attention_weight

    def _sa_block(self, text):
        text, attention_weight = self.multihead_attention(text, text, text)
        text = self.dropout1(text)
        return text, attention_weight
    

# MOEAttention model.
class MOEAttention(nn.Module):
    def __init__(self, 
                 num_classes=2,
                 expert_hidden_dim=128,
                 top_k=1):
        super(MOEAttention, self).__init__()
        # text.size(dim=1) to get text size, same for others
        self.align_size = 128
        self.align = FeatureAlign(align_size=self.align_size)
        self.expert_out_dim = 128

        # MOE for fusing text and tweet embeddings.
        self.desc_tweet_moe = MixtureOfExperts(
            input_dim=self.align_size, 
            num_experts=2,
            expert_hidden_dim=expert_hidden_dim,
            expert_output_dim=self.expert_out_dim,
            top_k=top_k,
            # tweet_desc=True
        )
        # MOE for fusing content and time embeddings.
        self.dna_moe = MixtureOfExperts(
            input_dim=self.align_size,
            num_experts=2,
            expert_hidden_dim=expert_hidden_dim,
            expert_output_dim=self.expert_out_dim,
            top_k=top_k,
            # tweet_desc=True
        )
        
        # Fusion module to combine MOE outputs.
        self.fusion = LModel(embed_dim=self.expert_out_dim, num_heads=2)
        
        # Fixed pooling as used in AllInOne.
        self.fixed_pooling = FixedPooling(fixed_size=4)
        # self.adaptive_pooling = nn.AdaptiveMaxPool2d((4, 4))
        self.consistency_conv = nn.Conv2d(
            in_channels=1,      
            out_channels=1,     
            kernel_size=(3,3),  
            padding=1           
        )
        
        # Batch normalization applied to the stacked outputs.
        self.bn1 = nn.BatchNorm1d(self.expert_out_dim)
        self.dropout1 = nn.Dropout(p=0.1)
        
        # The final feature dimension is the concatenation of:
        # - Fused tokens: here, we have 2 tokens each of dimension hidden_dim.
        # - Pooled attention: fixed pooling produces a (4 x 4) map → 16 features.
        final_feature_dim = self.expert_out_dim * 2 + 16
        self.bn2 = nn.BatchNorm1d(final_feature_dim)
        # self.dropout2 = nn.Dropout(p=0.1)
        self.mlp_classifier = nn.Linear(final_feature_dim, num_classes)

        self.apply(init_weights)

    def forward(self, desc, tweet, content, time):

        # desc, tweet, content, time = self.align(desc, tweet, content, time)
        # text_out, aux_loss_text = self.desc_tweet_moe(torch.cat((desc, tweet), dim=1))
        # # text_out, aux_loss_text = self.desc_tweet_moe.forward_roberta(tweets_tensor=tweet, des_tensor=desc)
        # dna_out, aux_loss_dna = self.dna_moe(torch.cat((content, time), dim=1))
        # # dna_out, aux_loss_dna = self.dna_moe.forward_roberta(tweets_tensor=time, des_tensor=content)
        # aux_loss = aux_loss_text + aux_loss_dna

        text, dna = self.align(desc, tweet, content, time)

        text_out, aux_loss_text = self.desc_tweet_moe(text)
        dna_out, aux_loss_dna = self.desc_tweet_moe(dna)
        aux_loss = aux_loss_text + aux_loss_dna

        
        # Stack the outputs along a new token dimension.
        outputs = [text_out, dna_out]  # Each: (batch, hidden_dim)
        out_tensor = torch.stack(outputs, dim=1)  # Shape: (batch, 2, hidden_dim)
        
        # Apply batch normalization along the feature dimension.
        out_tensor = self.bn1(out_tensor.transpose(1, 2)).transpose(1, 2)
        out_tensor = self.dropout1(out_tensor)
        
        # Fuse tokens using LModel.
        fused, attn = self.fusion(out_tensor)  # fused: (batch, 2, hidden_dim), attn: (batch, num_heads, 2, 2)

        # Apply fixed pooling to the attention map.
        attn = self.fixed_pooling(attn)  # Expected output shape: (batch, 4, 4)
        # print("fused shape:", fused.shape)
        # print("attention shape:", attn.shape)
        # ------* consistency *------
        attn = attn.unsqueeze(1)
        attn = self.consistency_conv(attn)
        # ------------------------
        # Flatten the fused tokens and attention map.
        fused_flat = fused.reshape(fused.size(0), -1)       # (batch, 2*hidden_dim)
        # attn_flat = attn.reshape(attn.size(0), -1)            # (batch, 16)
        # if consistency used:
        attn_flat = attn.flatten(start_dim=1)
        # Concatenate and classify.
        final_features = torch.cat([fused_flat, attn_flat], dim=1)  # (batch, 2*hidden_dim + 16)
        final_features = self.bn2(final_features)
        # final_features = self.dropout2(final_features)
        logits = self.mlp_classifier(final_features)
        
        return logits, aux_loss

def init_weights(m):
    if type(m)==nn.Linear:
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

# Example usage:
if __name__ == "__main__":
    batch_size = 8
    text_dim = 768
    tweet_dim = 768
    content_dim = 640
    time_dim = 640

    # Dummy inputs.
    # naive_input = torch.randn(batch_size, 20)       # Description embeddings.
    text_input = torch.randn(batch_size, text_dim)       # Description embeddings.
    tweet_input = torch.randn(batch_size, tweet_dim)       # Meta tweet embeddings.
    content_input = torch.randn(batch_size, content_dim)
    time_input = torch.randn(batch_size, time_dim)

    # Instantiate the classifier.
    model = MOEAttention(
        num_classes=2,
        expert_hidden_dim=256,
        top_k=1
    )
    
    logits, total_aux_loss = model(text_input, tweet_input, content_input, time_input)
    print("Logits shape:", logits.shape)         # Expected: (batch_size, num_classes)
    print("Total MOE auxiliary loss:", total_aux_loss.item())
