import torch
import torch.nn as nn
import torch.nn.functional as F
from moe2 import MixtureOfExperts  # your updated MOE module that accepts (gate_input, text_inputs)

class FeatureAlign(nn.Module):
    def __init__(self, align_size):
        super(FeatureAlign, self).__init__()
        self.linear_relu_des = nn.Sequential(
            nn.Linear(768, align_size),
            nn.LeakyReLU()
        )
        self.linear_content_dna = nn.Sequential(
            nn.Linear(640, align_size),
            nn.LeakyReLU()
        )
        self.linear_time_dna = nn.Sequential(
            nn.Linear(640, align_size),
            nn.LeakyReLU()
        )

    def forward(self, des_tensor, content_dna_tensor, time_dna_tensor):
        des_aligned = self.linear_relu_des(des_tensor)
        content_aligned = self.linear_content_dna(content_dna_tensor)
        time_aligned = self.linear_time_dna(time_dna_tensor)
        return des_aligned, content_aligned, time_aligned
    

class MOEAttention(nn.Module):
    def __init__(self, num_classes=2, expert_hidden_dim=256, top_k=1):
        super(MOEAttention, self).__init__()
        self.align = FeatureAlign(128)
        # MOE for the description branch:
        self.desc_moe = MixtureOfExperts(
            gate_input_dim=128,
            num_experts=2,
            expert_hidden_dim=expert_hidden_dim,
            top_k=top_k
        )
        # MOE for the DNA branch:
        self.dna_moe = MixtureOfExperts(
            gate_input_dim=128*2,
            num_experts=2,
            expert_hidden_dim=expert_hidden_dim,
            top_k=top_k
        )
        # Projection layer to convert concatenated 256-dim (content + time) to 128-dim gating vector.
        # self.dna_projection = nn.Linear(256, 128)
        self.num_classes = num_classes

    def forward(self, desc, content, time, desc_tokenized, content_time_tokenized):
        """
        Args:
          desc: Tensor of shape (batch_size, 768) with raw description embeddings.
          content: Tensor of shape (batch_size, 640) with raw content embeddings.
          time: Tensor of shape (batch_size, 640) with raw time embeddings.
          desc_tokenized: Dict of tokenized text for the description branch.
          content_time_tokenized: Dict of tokenized text for the DNA branch.
          
        Returns:
          final_logits: Tensor of shape (batch_size, 2) with final classification outputs.
          total_aux_loss: Sum of auxiliary load-balancing losses from both MOE modules.
        """
        # Obtain 128-dim aligned vectors for each modality.
        aligned_desc, aligned_content, aligned_time = self.align(desc, content, time)
        
        # Description branch uses the aligned description vector.
        desc_logits, desc_aux_loss = self.desc_moe(aligned_desc, desc_tokenized)
        
        # DNA branch: concatenate aligned content & time (results in 256 dims)
        # then project down to 128 dims.
        concatenated = torch.cat([aligned_content, aligned_time], dim=1)
        # aligned_dna = self.dna_projection(concatenated)
        dna_logits, dna_aux_loss = self.dna_moe(concatenated, content_time_tokenized)
        
        # Combine logits (for example, by averaging) and sum the auxiliary losses.
        final_logits = (desc_logits + dna_logits) / 2
        total_aux_loss = desc_aux_loss + dna_aux_loss
        
        return final_logits, total_aux_loss


###############################################################################
# Example usage:
###############################################################################
if __name__ == "__main__":
    batch_size = 8
    seq_length = 16  # Example sequence length for tokenized text.
    vocab_size = 32128  # T5's vocabulary size
    
    # Dummy tokenized inputs.
    dummy_desc_tokenized = {
        "input_ids": torch.randint(0, vocab_size, (batch_size, seq_length)),
        "attention_mask": torch.ones(batch_size, seq_length, dtype=torch.long)
    }
    dummy_content_time_tokenized = {
        "input_ids": torch.randint(0, vocab_size, (batch_size, seq_length)),
        "attention_mask": torch.ones(batch_size, seq_length, dtype=torch.long)
    }
    
    # Dummy raw embeddings for each modality.
    dummy_desc = torch.randn(batch_size, 768)
    dummy_content = torch.randn(batch_size, 640)
    dummy_time = torch.randn(batch_size, 640)
    
    # Instantiate and test the model.
    model = MOEAttention(num_classes=2, expert_hidden_dim=256, top_k=1)
    model.train()  # Enables training mode (e.g., for noisy gating if applicable)
    
    logits, total_aux_loss = model(dummy_desc, dummy_content, dummy_time,
                                   dummy_desc_tokenized, dummy_content_time_tokenized)
    print("Logits shape:", logits.shape)  # Expected: (batch_size, 2)
    print("Total auxiliary loss:", total_aux_loss.item())
