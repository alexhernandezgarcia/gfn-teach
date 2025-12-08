"""
This script trains a minimal discrete diffusion model on the "Toy environment", which is
defined in Figure 2 of the GFlowNet Foundations paper, Bengio et al (JMLR, 2023):

    .. _a link: https://jmlr.org/papers/v24/22-0364.html

Key differences with gfn.py:
- Learns terminal state distribution directly instead of a sequential policy
- Does not use the DAG structure; treats terminals as an unstructured categorical
- Trains on samples from target distribution p(x) ∝ R(x), not on-policy trajectories
- Generates terminal state in one shot via denoising, not step-by-step navigation
"""

import torch
from torch.distributions import Categorical
from tqdm import tqdm

### COMMON VARIABLES ###

float_type = torch.float32
device = torch.device("cpu")
do_print = False

### ENVIRONMENT ###

# Terminal states and their rewards (same as gfn.py)
rewards_dict = {
    3: 30,
    4: 14,
    6: 23,
    8: 10,
    9: 30,
    10: 5,
}
terminal_states = list(rewards_dict.keys())
n_terminal = len(terminal_states)
state_to_idx = {s: i for i, s in enumerate(terminal_states)}
idx_to_state = {i: s for i, s in enumerate(terminal_states)}

# Target distribution: p(x) ∝ R(x)
z = sum(rewards_dict.values())
target_probs = torch.tensor(
    [rewards_dict[s] / z for s in terminal_states],
    dtype=float_type,
    device=device,
)

### DIFFUSION PARAMETERS ###

n_timesteps = 10
# Masking probability schedule: beta_t = probability of masking at step t
# Linear schedule from 0 to 1
betas = torch.linspace(0.1, 0.9, n_timesteps, dtype=float_type, device=device)

### DENOISING MODEL ###

# Input: (terminal_state_idx or MASK, timestep) -> logits over terminal states
# MASK is represented by n_terminal (index after all valid states)
MASK_IDX = n_terminal

class DenoisingModel(torch.nn.Module):
    def __init__(self, n_states, n_timesteps, hidden_dim=32):
        super().__init__()
        self.state_embed = torch.nn.Embedding(n_states + 1, hidden_dim)  # +1 for MASK
        self.time_embed = torch.nn.Embedding(n_timesteps, hidden_dim)
        self.net = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim * 2, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, n_states),
        )

    def forward(self, x, t):
        # x: (batch,) state indices or MASK_IDX
        # t: (batch,) timestep indices
        h_x = self.state_embed(x)
        h_t = self.time_embed(t)
        h = torch.cat([h_x, h_t], dim=-1)
        return self.net(h)

model = DenoisingModel(n_terminal, n_timesteps).to(device)

### OPTIMIZER ###

n_train_steps = 5000
batch_size = 64
learning_rate = 0.01
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

### TRAIN ###

if not do_print:
    pbar = tqdm(initial=0, total=n_train_steps)

for step in range(n_train_steps):
    # Sample x_0 from target distribution
    x_0 = Categorical(probs=target_probs).sample((batch_size,))

    # Sample random timestep
    t = torch.randint(0, n_timesteps, (batch_size,), device=device)

    # Forward process: corrupt x_0 by masking with probability beta_t
    mask_prob = betas[t]
    mask = torch.rand(batch_size, device=device) < mask_prob
    x_t = torch.where(mask, torch.full_like(x_0, MASK_IDX), x_0)

    # Predict x_0 from x_t
    logits = model(x_t, t)

    # Cross-entropy loss to predict original x_0
    loss = torch.nn.functional.cross_entropy(logits, x_0)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if do_print:
        print(f"Step {step}, Loss: {loss.item():.4f}")
    else:
        pbar.update(1)
        pbar.set_description(f"Loss: {loss.item():.4f}")

if not do_print:
    pbar.close()

### SAMPLING ###

@torch.no_grad()
def sample(n_samples):
    # Start from fully masked
    x = torch.full((n_samples,), MASK_IDX, dtype=torch.long, device=device)

    # Reverse diffusion: from t=T-1 to t=0
    for t_val in reversed(range(n_timesteps)):
        t = torch.full((n_samples,), t_val, dtype=torch.long, device=device)

        # Predict x_0
        logits = model(x, t)
        probs = torch.softmax(logits, dim=-1)

        # Sample from predicted distribution for masked positions
        x_pred = Categorical(probs=probs).sample()

        # For masked positions, use the prediction
        # For unmasked positions, keep the current value
        is_masked = x == MASK_IDX
        x = torch.where(is_masked, x_pred, x)

    return x

### EVALUATION ###

n_samples = 2000

samples = sample(n_samples)

# Convert indices back to states and count
samples_dict = {s: 0 for s in terminal_states}
for idx in samples.tolist():
    state = idx_to_state[idx]
    samples_dict[state] += 1

# Print results
print("\nEvaluation: \n")
absolute_error = 0.0
for state, count in samples_dict.items():
    p_sampled = count / n_samples
    p_true = rewards_dict[state] / z
    absolute_error += abs(p_sampled - p_true)
    print(
        "- Sample {:2d} was generated with probability {:.2f} and the "
        "actual probability is {:.2f}".format(state, p_sampled, p_true)
    )
mae = absolute_error / len(samples_dict)
print("Mean absolute error: {:.2f}".format(mae))
