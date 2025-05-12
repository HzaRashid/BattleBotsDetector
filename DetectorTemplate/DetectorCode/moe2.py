import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions.normal import Normal

class SparseDispatcher(object):
    """
    Helper for implementing a mixture of experts.
    Routes examples to experts and later combines expert outputs.
    """
    def __init__(self, num_experts, gates):
        self._gates = gates
        self._num_experts = num_experts
        # Find nonzero (batch, expert) indices and sort them.
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
        Splits input tensor into a list where each element contains
        the inputs for the corresponding expert.
        """
        inp_exp = inp[self._batch_index]
        return torch.split(inp_exp, self._part_sizes, dim=0)

    def combine(self, expert_out, multiply_by_gates=True):
        """
        Combines expert outputs into a single tensor weighted by the gates.
        Clamps zero values to a small epsilon to avoid numerical issues.
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
        # Clamp any zero values to a small constant (epsilon) to avoid division-by-zero issues.
        combined = torch.clamp(combined, min=np.finfo(float).eps)
        return combined

    def expert_to_gates(self):
        """Splits nonzero gate values for each expert."""
        return torch.split(self._nonzero_gates, self._part_sizes, dim=0)

class MOEExpert(nn.Module):
    """
    A simple feed-forward expert.
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MOEExpert, self).__init__()
        self.expert = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.expert(x)

class MixtureOfExperts(nn.Module):
    """
    A mixture of experts module that implements noisy top-k gating and 
    an auxiliary load-balancing loss. The gating logic (using w_gate and w_noise)
    is adapted from the reference implementation while preserving the overall 
    network shape and expert functionality.
    """
    def __init__(self, 
                 input_dim, 
                 num_experts, 
                 expert_hidden_dim, 
                 expert_output_dim,
                 top_k=1,
                 noisy_gating=True):
        super(MixtureOfExperts, self).__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.noisy_gating = noisy_gating
        
        # Parameters for gating: use Xavier uniform initialization instead of zeros.
        self.w_gate = nn.Parameter(torch.empty(input_dim, num_experts))
        nn.init.xavier_uniform_(self.w_gate)
        self.w_noise = nn.Parameter(torch.empty(input_dim, num_experts))
        nn.init.xavier_uniform_(self.w_noise)
        
        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(dim=1)
        
        # Buffers for the noise distribution used in _prob_in_top_k.
        self.register_buffer("mean", torch.tensor([0.1]))
        self.register_buffer("std", torch.tensor([1.0]))
        
        # Instantiate experts.
        self.experts = nn.ModuleList([
            MOEExpert(input_dim, expert_hidden_dim, expert_output_dim)
            for _ in range(num_experts)
        ])

    def _cv_squared(self, x):
        """
        Compute the squared coefficient of variation: var(x) / (mean(x)**2 + eps).
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

    def forward(self, x, loss_coef=1e-2):
        """
        Args:
          x: input tensor of shape (batch_size, input_dim)
          loss_coef: scalar multiplier for the auxiliary loss
        Returns:
          output: combined expert outputs of shape (batch_size, expert_output_dim)
          loss: auxiliary load-balancing loss
        """
        # Compute clean logits.
        clean_logits = x @ self.w_gate  # (batch_size, num_experts)
        
        # Add noise if noisy gating is enabled and in training mode.
        if self.noisy_gating and self.training:
            raw_noise_stddev = x @ self.w_noise
            noise_stddev = self.softplus(raw_noise_stddev) + 1e-2  # add epsilon for stability
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
        
        # Use SparseDispatcher to route inputs to experts.
        dispatcher = SparseDispatcher(self.num_experts, gates)
        expert_inputs = dispatcher.dispatch(x)
        expert_outputs = [self.experts[i](expert_inputs[i]) for i in range(self.num_experts)]
        output = dispatcher.combine(expert_outputs)
        
        return output, loss

# Example usage:
if __name__ == "__main__":
    batch_size = 8
    naive_dim=20
    content_dim = 640
    time_dim = 640
    input_dim = content_dim + time_dim  # e.g., 1280

    num_experts = 3
    expert_hidden_dim = 256
    expert_output_dim = 384
    top_k = 1  # Adjust as needed

    moe = MixtureOfExperts(
        input_dim, num_experts, expert_hidden_dim, expert_output_dim,
        top_k=top_k, noisy_gating=True,
    )
    
    dummy_input = torch.randn(batch_size, input_dim)
    output, aux_loss = moe(dummy_input)
    print("Output shape:", output.shape)         # Expected: (batch_size, expert_output_dim)
    print("Auxiliary loss:", aux_loss.item())
