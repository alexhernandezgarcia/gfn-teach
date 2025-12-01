"""
This script trains a minimal version of a discrete diffusion model with the Flow Matching objective on
the "Toy environment", which is the environment defined in Figure 2 of the GFlowNet
Foundations paper, Bengio et al (JMLR, 2023):

    .. _a link: https://jmlr.org/papers/v24/22-0364.html

The discrete diffusion model is based on implementation from 
Campbell, A., Yim, J., Barzilay, R., Rainforth, T. &amp; Jaakkola, T.. (2024).
Generative Flows on Discrete State-Spaces: Enabling Multimodal Flows with Applications to Protein Co-Design. 
Proceedings of the 41st International Conference on Machine Learning, in Proceedings of Machine Learning Research
Available from https://proceedings.mlr.press/v235/campbell24a.html.
Refer eq. 18 for more details about the rate matrix.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from tqdm import tqdm

### CONFIGURATION ###

# NOISE_TYPE = 'masking' # Options: 'masking', 'uniform'
NOISE_TYPE = 'uniform'

### COMMON VARIABLES ###

float_type = torch.float32
device = torch.device("cpu")
do_print = False

### ENVIRONMENT ###

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
rewards_dict = {
    3: 30,
    4: 14,
    6: 23,
    8: 10,
    9: 30,
    10: 5,
}
n_states = len(connections_dict)

### MODEL PARAMETERS ###

D = 1 # Number of dimensions
if NOISE_TYPE == 'masking':
    S = n_states + 1 # Valid states 0..10, Mask token 11
elif NOISE_TYPE == 'uniform':
    S = n_states # Valid states 0..10
else:
    raise ValueError(f"Unknown NOISE_TYPE: {NOISE_TYPE}")

### MODEL ###

class Model(nn.Module):
    def __init__(self, D, S):
        super().__init__()
        self.embedding_dim = S
        self.embedding = nn.Embedding(S, self.embedding_dim)
        
        input_dim = (self.embedding_dim + 1) * D
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, S*D), # Output logits for all S states
        )

    def forward(self, x, t):
        B, D = x.shape
        x_emb = self.embedding(x) # (B, D, S)
        t_expanded = t[:, None, None].repeat(1, D, 1)
        net_input = torch.cat([x_emb, t_expanded], dim=-1).reshape(B, -1)
        return self.net(net_input).reshape(B, D, S)

policy = Model(D, S).to(device)

### OPTIMIZER ###

n_train_steps = 2000
learning_rate = 0.01
momentum = 0.9
optimizer = torch.optim.SGD(policy.parameters(), lr=learning_rate, momentum=momentum)

### HELPER FUNCTIONS ###

def sample_p_xt_g_x1(x1, t):
    # x1 (B, D)
    # t (B,)
    # Returns xt (B, D), corrupt_mask (B, D)
    B, D = x1.shape
    xt = x1.clone()
    
    if NOISE_TYPE == 'uniform':
        uniform_noise = torch.randint(0, S, (B, D), device=device)
        corrupt_mask = torch.rand((B, D), device=device) < (1 - t[:, None])
        xt[corrupt_mask] = uniform_noise[corrupt_mask]
    elif NOISE_TYPE == 'masking':
        mask_token = S - 1
        corrupt_mask = torch.rand((B, D), device=device) < (1 - t[:, None])
        xt[corrupt_mask] = mask_token
        
    return xt, corrupt_mask

def dt_p_xt_g_xt(x1, t):
    # x1 (B, D)
    # t float
    # returns (B, D, S)
    x1_onehot = F.one_hot(x1, num_classes=S).float()
    
    if NOISE_TYPE == 'uniform':
        return x1_onehot - (1/S)
    elif NOISE_TYPE == 'masking':
        mask_token = S - 1
        M_onehot = F.one_hot(torch.tensor([mask_token], device=device), num_classes=S).float() # (1, S)
        return x1_onehot - M_onehot

def p_xt_g_x1(x1, t):
    # x1 (B, D)
    # t float
    # returns (B, D, S)
    x1_onehot = F.one_hot(x1, num_classes=S).float()
    
    if NOISE_TYPE == 'uniform':
        return t * x1_onehot + (1-t) * (1/S)
    elif NOISE_TYPE == 'masking':
        mask_token = S - 1
        M_onehot = F.one_hot(torch.tensor([mask_token], device=device), num_classes=S).float()
        return t * x1_onehot + (1-t) * M_onehot

def sample_prior(num_samples, D):
    if NOISE_TYPE == 'uniform':
        return torch.randint(0, S, (num_samples, D), device=device)
    elif NOISE_TYPE == 'masking':
        return (S - 1) * torch.ones((num_samples, D), dtype=torch.long, device=device)

### TRAIN ###

if not do_print:
    pbar = tqdm(initial=0, total=n_train_steps, desc="Training")

terminal_states = list(rewards_dict.keys())
terminal_rewards = list(rewards_dict.values())
total_reward = sum(terminal_rewards)
terminal_probs = torch.tensor([r / total_reward for r in terminal_rewards], dtype=float_type, device=device)

batch_size = 1

for step in range(n_train_steps):
    
    # 1. Sample x1
    idx = Categorical(probs=terminal_probs).sample((batch_size,))
    x1 = torch.tensor(terminal_states, device=device)[idx].unsqueeze(1) # (B, D)
    
    # 2. Sample t
    t = torch.rand((batch_size,), device=device)
    
    # 3. Sample xt
    xt, corrupt_mask = sample_p_xt_g_x1(x1, t)
    
    # 4. Model prediction
    logits = policy(xt, t) # (B, D, S)
    
    # 5. Loss
    # Only compute loss on corrupted tokens
    target = x1.clone()
    target[~corrupt_mask] = -1
    if (target != -1).any():
        loss = F.cross_entropy(logits.transpose(1, 2), target, reduction='mean', ignore_index=-1)
            
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

if not do_print:
    pbar.close()

### EVALUATE ###

n_samples = 2000
samples_dict = {k: 0 for k in rewards_dict.keys()}

print("\nSampling...")

batch_size_eval = n_samples
xt = sample_prior(batch_size_eval, D)
t = 0.0
dt = 0.001

pbar_eval = tqdm(total=int(1.0/dt), desc="Sampling")
# This following implementation of rate matrix corresponds toEquation (18) in the paper
while t < 1.0:
    logits = policy(xt, t * torch.ones((batch_size_eval,), device=device))
    x1_probs = F.softmax(logits, dim=-1) # (B, D, S)
    x1_sample = Categorical(x1_probs).sample() # (B, D)
    
    # Calculate R_t^*
    dt_p_vals = dt_p_xt_g_xt(x1_sample, t) # (B, D, S)
    dt_p_vals_at_xt = dt_p_vals.gather(-1, xt[:, :, None]).squeeze(-1) # (B, D)
    
    R_t_numer = F.relu(dt_p_vals - dt_p_vals_at_xt[:, :, None]) # (B, D, S)
    
    pt_vals = p_xt_g_x1(x1_sample, t) # (B, D, S)
    Z_t = torch.count_nonzero(pt_vals, dim=-1) # (B, D)
    pt_vals_at_xt = pt_vals.gather(-1, xt[:, :, None]).squeeze(-1) # (B, D)
    
    R_t_denom = Z_t * pt_vals_at_xt # (B, D)
    
    # Avoid division by zero
    R_t_denom = R_t_denom + 1e-10
    
    R_t = R_t_numer / R_t_denom[:, :, None] # (B, D, S)
    
    # Zero out invalid transitions
    R_t[(pt_vals_at_xt < 1e-10)[:, :, None].repeat(1, 1, S)] = 0.0
    R_t[pt_vals < 1e-10] = 0.0
    
    step_probs = (R_t * dt).clamp(max=1.0)
    
    # Zero diagonal
    step_probs.scatter_(-1, xt[:, :, None], 0.0)
    # Fill diagonal
    step_probs.scatter_(-1, xt[:, :, None], (1.0 - step_probs.sum(dim=-1, keepdim=True)).clamp(min=0.0))
    
    xt = Categorical(step_probs).sample()
    
    t += dt
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
