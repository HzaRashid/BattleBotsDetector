import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.distributions.normal import Normal

class SparseDispatcher(object):
    """
    Helper for implementing a mixture of experts.
    Routes examples to experts and later combines expert outputs.
    """
    def __init__(self, num_experts, gates):
        self._gates = gates
        self._num_experts = num_experts
        # Find nonzero (batch, expert) indices, then sort them.
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
        """
        stitched = torch.cat(expert_out, 0)
        if multiply_by_gates:
            stitched = stitched.mul(self._nonzero_gates)
        zeros = torch.zeros(self._gates.size(0), expert_out[-1].size(1),
                            requires_grad=True, device=stitched.device)
        combined = zeros.index_add(0, self._batch_index, stitched)
        return combined

    def expert_to_gates(self):
        """Splits nonzero gate values for each expert."""
        return torch.split(self._nonzero_gates, self._part_sizes, dim=0)

class MOERouter(nn.Module):
    """
    A feed-forward network that acts as a router.
    It outputs raw logits and a softmax distribution over the experts.
    Optionally, it can add noise during training (noisy gating).
    """
    def __init__(self, input_dim, num_experts, hidden_dim, noisy_gating=False):
        super(MOERouter, self).__init__()
        self.noisy_gating = noisy_gating
        self.router = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_experts)
        )
        # self.router = nn.Linear(input_dim, num_experts)
        # If using noisy gating, create a noise layer.
        if self.noisy_gating:
            self.noise_layer = nn.Linear(input_dim, num_experts)
        # Allow the router parameters to be learned.

    def forward(self, x):
        # x: (batch_size, input_dim)
        logits = self.router(x)  # (batch_size, num_experts)
        if self.noisy_gating and self.training:
            noise_stddev = F.softplus(self.noise_layer(x)) + 1e-2
            noise = torch.randn_like(logits) * noise_stddev
            logits = logits + noise
        routing_weights = F.softmax(logits, dim=1)  # (batch_size, num_experts)
        return logits, routing_weights

class MOEExpert(nn.Module):
    """
    A simple feed-forward expert.
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MOEExpert, self).__init__()
        self.expert = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(hidden_dim, output_dim),
            # nn.ReLU(),
        )
    
    def forward(self, x):
        # x: (batch_size, input_dim)
        return self.expert(x)

class MixtureOfExperts(nn.Module):
    """
    Combines the MOERouter and multiple MOEExpert networks.
    Uses sparse routing (top-k) and a SparseDispatcher to assign
    inputs to experts and to combine their outputs.
    
    Computes the auxiliary loss in the same way as the original example:
    
      loss = loss_coef * (cv_squared(importance) + cv_squared(load))
    
    where:
      - importance = sum of routing weights for each expert, and
      - load = number of examples assigned to each expert.
    """
    def __init__(self, 
                 input_dim, 
                 num_experts, 
                 expert_hidden_dim, 
                 expert_output_dim,
                 router_hidden_dim=128, 
                 sparse_routing=True, 
                 top_k=1,
                 noisy_gating=True):
        super(MixtureOfExperts, self).__init__()
        self.router = MOERouter(input_dim, num_experts, router_hidden_dim, noisy_gating=noisy_gating)
        self.experts = nn.ModuleList([
            MOEExpert(input_dim, expert_hidden_dim, expert_output_dim)
            for _ in range(num_experts)
        ])
        self.sparse_routing = sparse_routing
        self.top_k = top_k

    def _cv_squared(self, x):
        """
        Compute the squared coefficient of variation:
            var(x) / (mean(x)**2 + eps)
        Returns 0 if there is only one expert.
        """
        eps = 1e-10
        if x.shape[0] == 1:
            return torch.tensor(0.0, device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean()**2 + eps)

    def forward(self, x, loss_coef=1e-2):
        """
        x: (batch_size, input_dim)
        loss_coef: a scalar multiplier for the auxiliary loss.
        
        Returns:
          - output: (batch_size, expert_output_dim)
          - loss: auxiliary loss computed using cv_squared on both importance and load.
        """
        # Get raw logits and routing weights from the router.
        logits, routing_weights = self.router(x)
        
        # If sparse routing is enabled, only keep top_k experts per sample.
        if self.sparse_routing:
            topk_values, topk_indices = routing_weights.topk(self.top_k, dim=1)
            mask = torch.zeros_like(routing_weights)
            mask.scatter_(1, topk_indices, 1.0)
            routing_weights = routing_weights * mask
            # Re-normalize the routing weights.
            routing_weights = routing_weights / (routing_weights.sum(dim=1, keepdim=True) + 1e-10)
        
        # Compute auxiliary loss.
        importance = routing_weights.sum(dim=0)  # shape: (num_experts)
        load = (routing_weights > 0).float().sum(dim=0)  # shape: (num_experts)
        loss = loss_coef * (self._cv_squared(importance) + self._cv_squared(load))
        
        # Use SparseDispatcher to route inputs to experts.
        dispatcher = SparseDispatcher(len(self.experts), routing_weights)
        expert_inputs = dispatcher.dispatch(x)
        # Compute outputs from each expert on its dispatched inputs.
        expert_outputs = [self.experts[i](expert_inputs[i]) for i in range(len(self.experts))]
        # Combine the expert outputs back into a single tensor.
        output = dispatcher.combine(expert_outputs)
        
        return output, loss

# Example usage:
if __name__ == "__main__":
    batch_size = 8
    content_dim = 640
    time_dim = 640
    input_dim = content_dim + time_dim  # e.g., 1280

    num_experts = 3
    expert_hidden_dim = 256
    expert_output_dim = 384
    router_hidden_dim = 128
    top_k = 1  # or adjust as needed

    moe = MixtureOfExperts(
        input_dim, num_experts, expert_hidden_dim, expert_output_dim,
        router_hidden_dim, sparse_routing=True, top_k=top_k, noisy_gating=True,
    )
    
    dummy_input = torch.randn(batch_size, input_dim)
    output, aux_loss = moe(dummy_input)
    print("Output shape:", output.shape)         # Expected: (batch_size, expert_output_dim)
    print("Auxiliary loss (lb + Z-loss):", aux_loss.item())
