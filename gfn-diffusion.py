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
n_states = len(connections_dict) # 11 (0..10)

# Mappings
# States: 0..10
# EOS: 11
# MASK: 12
EOS_TOKEN = 11
MASK_TOKEN = 12
S = 13 # Vocabulary size
D = 1  # We generate 1 token (next state) at a time

### DATASET GENERATION ###

def get_all_trajectories(current_state, current_path):
    # Returns a list of all valid trajectories starting from current_state
    # Trajectory includes EOS_TOKEN at the end
    
    if current_state == -1:
        return [current_path + [EOS_TOKEN]]
    
    trajs = []
    children = connections_dict[current_state]
    for child in children:
        if child == -1:
             trajs.append(current_path + [EOS_TOKEN])
        else:
            child_trajs = get_all_trajectories(child, current_path + [child])
            trajs.extend(child_trajs)
            
    return trajs

# Generate all valid paths starting from 0
all_paths_raw = get_all_trajectories(0, [0])

# Filter and weight paths by reward
valid_paths = []
path_rewards = []

for path in all_paths_raw:
    # Path ends with EOS. The terminal state is path[-2].
    term_state = path[-2]
    if term_state in rewards_dict:
        r = rewards_dict[term_state]
        valid_paths.append(path)
        path_rewards.append(r)

# Normalize rewards to get probabilities
total_reward = sum(path_rewards)
path_probs = torch.tensor([r / total_reward for r in path_rewards], dtype=float_type, device=device)

# We don't convert valid_paths to a single tensor because they have different lengths.
# We will sample indices and then pick the path.

### MODEL ###

class ConditionalModel(nn.Module):
    def __init__(self, S):
        super().__init__()
        self.embedding_dim = S
        self.embedding = nn.Embedding(S, self.embedding_dim)
        
        # Input: 
        # - Noisy next state embedding (S)
        # - Current state embedding (S) (Condition)
        # - Time embedding (1)
        input_dim = self.embedding_dim * 2 + 1
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, S), # Output logits for next state
        )

    def forward(self, x, t, cond):
        # x: (B, 1) or (B,) -> Noisy next state
        # t: (B,)
        # cond: (B, 1) or (B,) -> Current state
        
        if x.dim() == 1: x = x.unsqueeze(1)
        if cond.dim() == 1: cond = cond.unsqueeze(1)
        
        B = x.shape[0]
        
        x_emb = self.embedding(x).squeeze(1) # (B, S)
        cond_emb = self.embedding(cond).squeeze(1) # (B, S)
        t_emb = t.unsqueeze(1) # (B, 1)
        
        net_input = torch.cat([x_emb, cond_emb, t_emb], dim=-1) # (B, 2S+1)
        return self.net(net_input).unsqueeze(1) # (B, 1, S)

policy = ConditionalModel(S).to(device)

### OPTIMIZER ###

n_train_steps = 2000
learning_rate = 0.01
momentum = 0.9
optimizer = torch.optim.SGD(policy.parameters(), lr=learning_rate, momentum=momentum)

### TRAIN ###

if not do_print:
    pbar = tqdm(initial=0, total=n_train_steps, desc="Training")

batch_size = 32

for step in range(n_train_steps):
    
    # 1. Sample trajectories
    idx = Categorical(probs=path_probs).sample((batch_size,))
    
    # 2. Extract random transitions from sampled trajectories
    current_states = []
    next_states = []
    
    for i in idx:
        path = valid_paths[i.item()]
        # path is [0, s1, ..., EOS]
        # Pick a random transition
        # len(path) is at least 2 (0, EOS)
        # transitions are (path[k], path[k+1])
        # k goes from 0 to len(path)-2
        k = torch.randint(0, len(path)-1, (1,)).item()
        current_states.append(path[k])
        next_states.append(path[k+1])
        
    curr_s = torch.tensor(current_states, dtype=torch.long, device=device) # (B,)
    next_s = torch.tensor(next_states, dtype=torch.long, device=device)   # (B,) (Target x1)
    
    # 3. Sample t
    t = torch.rand((batch_size,), device=device)
    
    # 4. Corrupt next_s to xt (Random Masking)
    xt = next_s.clone()
    mask_prob = 1 - t # (B,)
    will_mask = torch.rand((batch_size,), device=device) < mask_prob
    xt[will_mask] = MASK_TOKEN
    
    # 5. Model prediction
    # Predict next_s given xt and curr_s
    logits = policy(xt, t, curr_s) # (B, 1, S)
    
    # 6. Loss
    # Only compute loss on masked tokens
    target = next_s.clone().unsqueeze(1) # (B, 1)
    target[~will_mask.unsqueeze(1)] = -1
    
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

### EVALUATE (SAMPLING) ###

n_samples = 2000
samples_dict = {k: 0 for k in rewards_dict.keys()}

print("\nSampling...")

# Sampling parameters
dt = 0.4 # Step size for diffusion generation of a single token, this is choosen to speed up
noise_scale = 10.0

# Precompute transition masks for efficiency
# trans_mask[u] gives a boolean mask of valid children for state u
valid_children_masks = torch.zeros((S, S), dtype=torch.bool, device=device)
for u in range(S):
    if u in connections_dict:
        children = connections_dict[u]
        for v in children:
            if v == -1:
                valid_children_masks[u, EOS_TOKEN] = True
            else:
                valid_children_masks[u, v] = True
    elif u == EOS_TOKEN:
        valid_children_masks[u, EOS_TOKEN] = True # Self-loop for EOS

pbar_eval = tqdm(total=n_samples, desc="Sampling Trajectories")

for i in range(n_samples):
    
    # Start at state 0
    curr_state = 0
    
    while curr_state != EOS_TOKEN:
        
        # Generate next state using diffusion
        # We want to generate 'x' (next state)
        # Start with x = MASK
        xt = torch.tensor([MASK_TOKEN], dtype=torch.long, device=device) # (1,)
        curr_s_tensor = torch.tensor([curr_state], dtype=torch.long, device=device) # (1,)
        
        t = 0.0
        # We simulate t from 1.0 down to 0.0? 
        # The training was: t ~ U[0, 1], mask_prob = 1-t.
        # So t=1 means mask_prob=0 (Clean). t=0 means mask_prob=1 (Masked).
        # Wait, usually t=0 is clean data in diffusion.
        # Let's check training:
        # mask_prob = 1 - t. 
        # If t=1, mask_prob=0 -> Clean.
        # If t=0, mask_prob=1 -> Masked.
        # So we should start at t=0 (Masked) and go to t=1 (Clean).
        
        t_curr = 0.0
        
        while t_curr < 1.0:
            t_tensor = torch.tensor([t_curr], dtype=float_type, device=device)
            
            # Predict x1 (Clean next state)
            logits = policy(xt, t_tensor, curr_s_tensor) # (1, 1, S)
            logits = logits.squeeze(1) # (1, S)
            
            # Mask invalid transitions
            # We must ensure the predicted x1 is a valid child of curr_state
            valid_mask = valid_children_masks[curr_state] # (S,)
            logits[0, ~valid_mask] = -torch.inf
            
            x1_probs = F.softmax(logits, dim=-1)
            x1_sample = Categorical(x1_probs).sample() # (1,)
            
            # Rate of change
            # From discrete_diffusion.py logic (adapted for t going 0->1)
            # In discrete_diffusion.py: t goes 0->1. mask_prob = 1-t.
            # So t=0 is Masked. t=1 is Clean.
            # That matches what I wrote in training.
            
            # Unmasking probability
            # prob_unmask = dt * (1 + noise * t) / (1 - t)
            # But here we are going 0->1 (Masked -> Clean)
            # The logic in discrete_diffusion.py was:
            # mask_prob = 1 - t.
            # So as t increases, mask_prob decreases (we unmask).
            
            denom = 1 - t_curr
            if denom < 1e-5: denom = 1e-5
            
            # Standard discrete diffusion rate
            prob_unmask = dt * (1 + noise_scale * t_curr) / denom
            
            # Vectorized mask generation
            # If currently masked, maybe unmask to x1_sample
            if xt.item() == MASK_TOKEN:
                if torch.rand(1).item() < prob_unmask:
                    xt = x1_sample
            
            # Re-masking (noise)
            # If currently unmasked, maybe mask again
            prob_mask = dt * noise_scale
            if xt.item() != MASK_TOKEN:
                if torch.rand(1).item() < prob_mask:
                    xt = torch.tensor([MASK_TOKEN], dtype=torch.long, device=device)
            
            t_curr += dt
            
            # Force unmask at the end if still masked?
            if t_curr >= 1.0 and xt.item() == MASK_TOKEN:
                 xt = x1_sample
        
        # Update state
        next_state = xt.item()
        
        # Safety check (should be handled by masking)
        if not valid_children_masks[curr_state, next_state]:
            # Fallback: just pick a valid child
            # This shouldn't happen if masking works and x1_sample is used at end
            # But if x1_sample was sampled when logits were bad?
            # We masked logits, so x1_sample should be valid.
            pass
            
        curr_state = next_state
    
    # Trajectory finished (reached EOS)
    # The terminal state is the one before EOS.
    # We don't store the full path here, just tracking curr_state.
    # Wait, if curr_state becomes EOS, the loop ends.
    # We need the state *before* EOS.
    # But we updated curr_state to next_state.
    # So we need to track `prev_state`.
    
    # Actually, let's just store the path.
    # But for evaluation we just need the terminal state.
    # The loop breaks when curr_state == EOS.
    # The `curr_state` variable holds EOS.
    # We need the state that transitioned TO EOS.
    # Let's restructure loop slightly.
    pass

# Re-implement loop for correct tracking
pbar_eval.close()
pbar_eval = tqdm(total=n_samples, desc="Sampling Trajectories")

for i in range(n_samples):
    curr_state = 0
    path = [0]
    
    while True:
        # Diffusion generation for next state
        xt = torch.tensor([MASK_TOKEN], dtype=torch.long, device=device)
        curr_s_tensor = torch.tensor([curr_state], dtype=torch.long, device=device)
        t_curr = 0.0
        
        while t_curr < 1.0:
            t_tensor = torch.tensor([t_curr], dtype=float_type, device=device)
            logits = policy(xt, t_tensor, curr_s_tensor).squeeze(1)
            valid_mask = valid_children_masks[curr_state]
            logits[0, ~valid_mask] = -torch.inf
            x1_probs = F.softmax(logits, dim=-1)
            x1_sample = Categorical(x1_probs).sample()
            
            denom = 1 - t_curr
            if denom < 1e-5: denom = 1e-5
            prob_unmask = dt * (1 + noise_scale * t_curr) / denom
            
            if xt.item() == MASK_TOKEN:
                if torch.rand(1).item() < prob_unmask:
                    xt = x1_sample
            
            prob_mask = dt * noise_scale
            if xt.item() != MASK_TOKEN:
                if torch.rand(1).item() < prob_mask:
                    xt = torch.tensor([MASK_TOKEN], dtype=torch.long, device=device)
            
            t_curr += dt
            
            if t_curr >= 1.0 and xt.item() == MASK_TOKEN:
                 xt = x1_sample
        
        next_state = xt.item()
        path.append(next_state)
        
        if next_state == EOS_TOKEN:
            break
        curr_state = next_state

    # Path ends with EOS. Terminal state is path[-2]
    term_state = path[-2]
    if term_state in samples_dict:
        samples_dict[term_state] += 1
    
    pbar_eval.update(1)

pbar_eval.close()

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
