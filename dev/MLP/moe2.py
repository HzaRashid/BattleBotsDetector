import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions.normal import Normal
from transformers import AutoModel

class SparseDispatcher(object):
    """
    Helper for implementing a mixture of experts.
    Routes examples to experts and later combines expert outputs.
    This version is updated to also support inputs that are dictionaries
    (e.g. tokenized text inputs).
    """
    def __init__(self, num_experts, gates):
        self._gates = gates
        self._num_experts = num_experts
        # Find indices of nonzero entries (batch, expert) and sort by expert index.
        sorted_experts, index_sorted_experts = torch.nonzero(gates).sort(0)
        # Extract expert indices.
        _, self._expert_index = sorted_experts.split(1, dim=1)
        # Get corresponding batch indices.
        self._batch_index = torch.nonzero(gates)[index_sorted_experts[:, 1], 0]
        # Number of examples assigned per expert.
        self._part_sizes = (gates > 0).sum(0).tolist()
        # Expand gates to match with batch indices.
        gates_exp = gates[self._batch_index]
        self._nonzero_gates = torch.gather(gates_exp, 1, self._expert_index)

    def dispatch(self, inp):
        """
        Splits input into a list where each element contains the inputs
        for the corresponding expert.
        If inp is a dictionary (e.g., tokenized text), each key's tensor is split accordingly.
        """
        if isinstance(inp, dict):
            # For each key, split the tensor along the batch dimension.
            splitted = {}
            for key, value in inp.items():
                # Select samples using batch_index and split into parts.
                splitted[key] = torch.split(value[self._batch_index], self._part_sizes, dim=0)
            # Combine splits from all keys into a list of dictionaries (one per expert).
            expert_inputs = []
            for i in range(self._num_experts):
                expert_dict = {}
                for key in splitted:
                    expert_dict[key] = splitted[key][i]
                expert_inputs.append(expert_dict)
            return expert_inputs
        else:
            inp_exp = inp[self._batch_index]
            return torch.split(inp_exp, self._part_sizes, dim=0)

    def combine(self, expert_out, multiply_by_gates=True):
        """
        Combines expert outputs into a single tensor weighted by the gates.
        Clamps near-zero values to a small epsilon for numerical stability.
        """
        stitched = torch.cat(expert_out, 0)
        if multiply_by_gates:
            stitched = stitched.mul(self._nonzero_gates)
        zeros = torch.zeros(
            self._gates.size(0),
            expert_out[-1].size(1),
            requires_grad=True,
            device=stitched.device
        )
        combined = zeros.index_add(0, self._batch_index, stitched)
        combined = torch.clamp(combined, min=np.finfo(float).eps)
        return combined

    def expert_to_gates(self):
        """Splits nonzero gate values for each expert."""
        return torch.split(self._nonzero_gates, self._part_sizes, dim=0)

class TransformerExpert(nn.Module):
    """
    An expert that utilizes a pretrained transformer model.
    The transformer is loaded via AutoModel. All parameters are frozen except those in its last encoder block.
    Its forward method applies only the encoder on tokenized text input to produce token-level representations,
    then applies a simple FFN (classification head) on the first token's output to produce 2 logits.
    """
    def __init__(self, hidden_dim):
        super(TransformerExpert, self).__init__()
        self.transformer = AutoModel.from_pretrained("google/flan-t5-small")
        
        # Freeze all transformer parameters.
        for param in self.transformer.parameters():
            param.requires_grad = False
        
        # Unfreeze only the last encoder block.
        # T5 models store the encoder blocks in `encoder.block`.
        if hasattr(self.transformer, "encoder") and hasattr(self.transformer.encoder, "block"):
            for param in self.transformer.encoder.block[-1].parameters():
                param.requires_grad = True
        elif hasattr(self.transformer, "encoder") and hasattr(self.transformer.encoder, "layer"):
            # This branch is for models with encoder.layer (like BERT).
            for param in self.transformer.encoder.layer[-1].parameters():
                param.requires_grad = True
        else:
            print("Warning: Unable to determine the transformer structure to unfreeze its last layer.")
        
        # Set up a simple feed-forward network for classification.
        transformer_hidden_size = self.transformer.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(transformer_hidden_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)
        )

    def forward(self, text_inputs):
        # Check if the input_ids is empty (i.e. batch dimension is 0)
        if text_inputs["input_ids"].size(0) == 0:
            # Return a tensor of zeros with shape (0, 2) so that downstream operations succeed.
            return torch.ones(0, 2, device=text_inputs["input_ids"].device)
        # Otherwise, process normally.
        encoder_outputs = self.transformer.encoder(**text_inputs)
        cls_output = encoder_outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_output)
        return logits

class MixtureOfExperts(nn.Module):
    """
    A mixture of experts module with noisy top-k gating and an auxiliary load-balancing loss.
    The gating is computed from a 128-dimensional tensor input. The experts themselves are transformers
    that process tokenized text inputs and output 2 logits.
    """
    def __init__(self,
                 gate_input_dim,  # expected to be 128
                 num_experts,
                 expert_hidden_dim,
                 top_k=1,
                 noisy_gating=True):
        super(MixtureOfExperts, self).__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.noisy_gating = noisy_gating

        # Parameters for gating: initialize with Xavier uniform.
        self.w_gate = nn.Parameter(torch.empty(gate_input_dim, num_experts), requires_grad=False)
        nn.init.xavier_uniform_(self.w_gate)
        self.w_noise = nn.Parameter(torch.empty(gate_input_dim, num_experts), requires_grad=False)
        nn.init.xavier_uniform_(self.w_noise)

        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(dim=1)

        # Buffers for the noise distribution used in _prob_in_top_k.
        self.register_buffer("mean", torch.tensor([0.1]))
        self.register_buffer("std", torch.tensor([1.0]))

        # Instantiate experts: each expert is a TransformerExpert.
        self.experts = nn.ModuleList([
            TransformerExpert(expert_hidden_dim)
            for _ in range(num_experts)
        ])

    def _cv_squared(self, x):
        """
        Compute the squared coefficient of variation: var(x) / (mean(x)^2 + eps).
        Returns 0 if there is only one element.
        """
        eps = 1e-10
        if x.shape[0] == 1:
            return torch.tensor(0.0, device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean()**2 + eps)

    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
        """
        Computes the probability that each value is in the top k given random noise.
        This is used to compute the load balancing loss when noisy gating is enabled.
        """
        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()
        threshold_positions_if_in = torch.arange(batch, device=clean_values.device) * m + self.top_k
        threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)
        is_in = torch.gt(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_out), 1)
        normal = Normal(self.mean, self.std)
        prob_if_in = normal.cdf((clean_values - threshold_if_in) / noise_stddev)
        prob_if_out = normal.cdf((clean_values - threshold_if_out) / noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob

    def forward(self, gate_input, text_inputs, loss_coef=1e-2):
        """
        Args:
          gate_input: Tensor of shape (batch_size, gate_input_dim) for gating (e.g., 128-dim)
          text_inputs: Dictionary of tokenized text inputs for the transformer experts.
          loss_coef: Scalar coefficient for the auxiliary load-balancing loss.
        Returns:
          output: Combined expert outputs of shape (batch_size, 2)
          loss: Auxiliary load-balancing loss.
        """
        # Compute clean logits for gating.
        clean_logits = gate_input @ self.w_gate  # shape: (batch_size, num_experts)

        # Add noise if noisy gating is enabled and in training mode.
        if self.noisy_gating and self.training:
            raw_noise_stddev = gate_input @ self.w_noise
            noise_stddev = self.softplus(raw_noise_stddev) + 1e-2  # epsilon added for stability
            noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits

        # Select top-k experts.
        top_logits, top_indices = logits.topk(min(self.top_k + 1, self.num_experts), dim=1)
        top_k_logits = top_logits[:, :self.top_k]
        top_k_indices = top_indices[:, :self.top_k]
        top_k_gates = self.softmax(top_k_logits)

        # Build the full gates tensor.
        gates = torch.zeros_like(logits)
        gates = gates.scatter(1, top_k_indices, top_k_gates)

        # Compute load and importance for the auxiliary loss.
        if self.noisy_gating and self.top_k < self.num_experts and self.training:
            load = self._prob_in_top_k(clean_logits, logits, noise_stddev, top_logits).sum(0)
        else:
            load = (gates > 0).sum(0)
        importance = gates.sum(0)
        loss = loss_coef * (self._cv_squared(importance) + self._cv_squared(load))

        # Use SparseDispatcher to route text_inputs to experts.
        dispatcher = SparseDispatcher(self.num_experts, gates)
        expert_inputs = dispatcher.dispatch(text_inputs)
        expert_outputs = [
            self.experts[i](expert_inputs[i])
            for i in range(self.num_experts)
        ]
        output = dispatcher.combine(expert_outputs)

        return output, loss

# Example usage:
if __name__ == "__main__":
    batch_size = 8
    # For gating, we now expect a 128-dimensional tensor.
    gate_input_dim = 128

    # Dummy gating input.
    gating_input = torch.randn(batch_size, gate_input_dim)

    # Create a dummy text input simulating tokenization.
    # For example, input_ids and attention_mask with a sequence length of 16.
    seq_length = 16
    vocab_size = 32128  # T5's vocabulary size per the printed model.
    dummy_input_ids = torch.randint(0, vocab_size, (batch_size, seq_length))
    dummy_attention_mask = torch.ones(batch_size, seq_length, dtype=torch.long)
    text_inputs = {
        "input_ids": dummy_input_ids,
        "attention_mask": dummy_attention_mask
    }

    num_experts = 3
    expert_hidden_dim = 256  # Hidden dimension for the FFN classification head.

    moe = MixtureOfExperts(
        gate_input_dim, num_experts, expert_hidden_dim,
        top_k=1, noisy_gating=True,
    )

    moe.train()  # Set to training mode for noisy gating.
    output, aux_loss = moe(gating_input, text_inputs)
    print("Output shape:", output.shape)  # Expected: (batch_size, 2)
    print("Auxiliary loss:", aux_loss.item())
