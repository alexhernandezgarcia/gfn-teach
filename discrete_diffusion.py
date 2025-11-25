"""
This script trains a minimal version of a Discrete Diffusion model with the Flow Matching objective on
the "Toy environment", which is the environment defined in Figure 2 of the GFlowNet
Foundations paper, Bengio et al (JMLR, 2023):

    .. _a link: https://jmlr.org/papers/v24/22-0364.html

The flow matching objective is implementation follows the paper:
Campbell, A., Yim, J., Barzilay, R., Rainforth, T. &amp; Jaakkola, T.. (2024).
Generative Flows on Discrete State-Spaces: Enabling Multimodal Flows with Applications to Protein Co-Design. 
Proceedings of the 41st International Conference on Machine Learning, in Proceedings of Machine Learning Research
Available from https://proceedings.mlr.press/v235/campbell24a.html.

The following code implements masking based diffusion model for the toy environment.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from tqdm import tqdm

### COMMON VARIABLES ###

float_type = torch.float32
device = torch.device("cpu")
do_print = False

### ENVIRONMENT ###

# A dictionary of connections: the keys of the dictionary are the indices of the
# states, and the values are the indices of the states to which each state is
# connected.
connections_dict = {
    0: (1, 2),
    1: (3,),
    2: (3, 4),
    3: (5, -1),
    4: (6, -1),
    5: (7, 8),
    6: (8, 10, -1),
    7: (9,),
    8: (9, -1),
    9: (-1,),
    10: (-1,),
}
# A dictionary of rewards: the keys of the dictionary are the indices of the
# states, and the values are their rewards.
rewards_dict = {
    3: 30,
    4: 14,
    6: 23,
    8: 10,
    9: 30,
    10: 5,
}
n_states = len(connections_dict)

### DIFFUSION MODEL ###
# The "data" we want to generate are the terminal states (from masking based discrete diffusion), distributed according to rewards.
# We treat the state index as a single dimension (D=1) categorical variable.
# The vocabulary size S is n_states + 1 (for the mask token).

D = 1 # Number of dimensions
S = n_states + 1 # Valid states 0..10, Mask token 11

class Model(nn.Module):
    def __init__(self, D, S):
        super().__init__()
        self.embedding_dim = S
        self.embedding = nn.Embedding(S, self.embedding_dim)
        # Input to net is (Embedding (S) + Time (1)) * D
        input_dim = (self.embedding_dim + 1) * D
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, (S-1)*D),
        )

    def forward(self, x, t):
        B, D = x.shape
        x_emb = self.embedding(x) # (B, D, S)
        # Concatenate time t to each dimension's embedding
        # t is (B,), expand to (B, D, 1)
        t_expanded = t[:, None, None].repeat(1, D, 1)
        net_input = torch.cat([x_emb, t_expanded], dim=-1).reshape(B, -1) # (B, D * (S+1))
        return self.net(net_input).reshape(B, D, S-1) # (B, D, S-1)

policy = Model(D, S).to(device)

### OPTIMIZER ###

n_train_steps = 2000
learning_rate = 0.01
momentum = 0.9
optimizer = torch.optim.SGD(policy.parameters(), lr=learning_rate, momentum=momentum)

### TRAIN ###

# Progress bar
if not do_print:
    pbar = tqdm(
        initial=0,
        total=n_train_steps,
    )

# Prepare ground truth distribution for sampling
terminal_states = list(rewards_dict.keys())
terminal_rewards = list(rewards_dict.values())
total_reward = sum(terminal_rewards)
terminal_probs = torch.tensor([r / total_reward for r in terminal_rewards], dtype=float_type, device=device)

batch_size = 1

for step in range(n_train_steps):

    # 1. Sample x1 (data) from ground truth
    idx = Categorical(probs=terminal_probs).sample((batch_size,))
    x1 = torch.tensor(terminal_states, device=device)[idx].unsqueeze(1) # (B, D)

    # 2. Sample time t
    t = torch.rand((batch_size,), device=device) # ODE so t is in [0, 1]

    # 3. Corrupt to xt
    # Masking: 0, 1, ..., S-2 are valid. S-1 is MASK.
    # Mask probability is 1 - t
    xt = x1.clone()
    mask_prob = 1 - t[:, None] # (B, 1)
    # Expand mask_prob to (B, D)
    mask_prob = mask_prob.repeat(1, D)
    
    will_mask = torch.rand((batch_size, D), device=device) < mask_prob
    xt[will_mask] = S - 1 # Set to MASK token

    # 4. Model prediction
    logits = policy(xt, t) # (B, D, S-1)

    # 5. Loss
    # Only compute loss on masked tokens
    target = x1.clone()
    # If xt is not masked, we ignore the loss (set target to -1)
    target[xt != S - 1] = -1

    if (target != -1).any():
        loss = F.cross_entropy(logits.transpose(1, 2), target, reduction='mean', ignore_index=-1)
        
        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    else:
        loss = torch.tensor(0.0, device=device)

    loss_print = loss.item()
    if do_print:
        print("Loss: {:.4f}".format(loss_print))
    else:
        pbar.update(1)
        pbar.set_description("Loss: {:.4f}".format(loss_print))

### EVALUATE ###

n_samples = 2000

# A dictionary to count the number of times each terminal state is sampled
samples_dict = {
    3: 0,
    4: 0,
    6: 0,
    8: 0,
    9: 0,
    10: 0,
}

# Vectorized sampling
batch_size_eval = n_samples
xt = (S - 1) * torch.ones((batch_size_eval, D), dtype=torch.long, device=device)
t = 0.0
dt = 0.001
noise = 10 # noise * dt * D is the average number of dimensions that get re-masked each timestep

# Progress bar for sampling
pbar_eval = tqdm(total=int(1.0/dt), desc="Sampling")

while t < 1.0: # simulate ODE
    logits = policy(xt, t * torch.ones((batch_size_eval,), device=device))
    x1_probs = F.softmax(logits, dim=-1) # (B, D, S-1)
    x1_sample = Categorical(x1_probs).sample() # (B, D)
    
    denom = 1 - t
    if denom < 1e-5: denom = 1e-5
    prob_unmask = dt * (1 + noise * t) / denom
    
    # Vectorized mask generation
    will_unmask = (torch.rand((batch_size_eval, D), device=device) < prob_unmask) & (xt == (S - 1))
    
    prob_mask = dt * noise
    will_mask = (torch.rand((batch_size_eval, D), device=device) < prob_mask) & (xt != (S - 1))
    
    xt[will_unmask] = x1_sample[will_unmask]
    
    t += dt
    
    if t < 1.0:
        xt[will_mask] = S - 1
    
    pbar_eval.update(1)

pbar_eval.close()

# Count samples
for i in range(n_samples):
    sample_val = xt[i, 0].item()
    if sample_val in samples_dict:
        samples_dict[sample_val] += 1

# Print results
print("\nEvaluation: \n")
z = sum(rewards_dict.values())
absolute_error = 0.0
for sample, count in samples_dict.items():
    p_sampled = count / n_samples
    p_true = rewards_dict[sample] / z
    absolute_error += abs(p_sampled - p_true)
    print(
        "- Sample {:2d} was generated with probability {:.2f} and the "
        "actual probability is {:.2f}".format(sample, p_sampled, p_true)
    )
mae = absolute_error / len(samples_dict)
print("Mean absolute error: {:.2f}".format(mae))
