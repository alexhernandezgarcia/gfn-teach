"""
This script trains a minimal REINFORCE agent on the "Toy environment", which is the
defined in Figure 2 of the GFlowNet Foundations paper, Bengio et al (JMLR, 2023):

.. _a link: https://jmlr.org/papers/v24/22-0364.html
"""

"""
Key differences with gfn.py:
- Uses REINFORCE framework instead of GFlowNet training
- Batches of episodes instead of single episodes to reduce variance
"""

import torch
from torch.distributions import Categorical
from tqdm import tqdm
import time


### COMMON VARIABLES ###

float_type = torch.float32
device = torch.device("cpu")
do_print = False

### ENVIRONMENT ###

discount_factor = 1.0 # No discounting

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

### POLICY MODEL ###

# Linear layer following an embedding layer. The inputs are the state indices and the
# outputs are tensors with dimensionality equal to the number of actions: the number of
# states plus one, for the end-of-sequence (EOS) action.
class Policy(torch.nn.Module):
    def __init__(self, n_states, float_type, device):
        super(Policy, self).__init__()
        self.embedding = torch.nn.Embedding(n_states, n_states)
        self.linear = torch.nn.Linear(n_states, n_states + 1, dtype=float_type, device=device)

    def forward(self, x):
        x = self.embedding(x)
        x = self.linear(x)
        return x

policy = Policy(n_states, float_type, device)

### OPTIMIZER ###

n_train_steps = 2000
learning_rate = 0.01
momentum = 0.9
optimizer = torch.optim.SGD(policy.parameters(), lr=learning_rate, momentum=momentum)

### LOSS FUNCTION ###

class ReinforceLoss(torch.nn.Module):
    def __init__(self, gamma=1.0):
        super(ReinforceLoss, self).__init__()
        self.gamma = gamma

    def forward(self, log_probs, returns):
        returns = torch.tensor(returns, dtype=float_type, device=device)
        # Why double flip: flip returns (e.g., [r0, r1, r2] -> [r2, r1, r0]), cumsum (prefix sums: [r2, r2+r1, r2+r1+r0]), flip back ([r2+r1+r0, r2+r1, r2])
        discounted_returns = torch.cumsum(returns.flip(0), dim=0).flip(0) 
        # REINFORCE Loss function: L(θ) = - J(θ), where J(θ) = sum over t log π_θ(a_t|s_t) * G_t (Advantage functions becomes the undiscounted cumulative reward since gamma=1.0 and there is no baseline)
        loss = - (log_probs * discounted_returns).sum()
        return loss

### GRAPH MASKS ###

mask_dict = {}
for state in range(n_states):
    mask_invalid = [False if s in connections_dict[state] else True for s in range(n_states)]
    mask_invalid += [-1 not in connections_dict[state]]
    mask_dict[state] = mask_invalid

### TRAIN ###

pg_loss = ReinforceLoss()

if not do_print:
    pbar = tqdm(
        initial=0,
        total=n_train_steps,
    )

start_time = time.time()

for step in range(n_train_steps):
    
    traj = {'states': [], 'actions': [], 'log_prob': [], 'rewards': []}
    
    state = 0
    traj_done = False
    
    if do_print:
        print(f"\nIteration {step}")
        print(f"\tTrajectory 0 -> ", end="")
    
    while not traj_done:
        
        mask_invalid = mask_dict[state]
        
        #with torch.no_grad(): # Commented out to keep gradients for log policy probability
        logits = policy(torch.tensor([state], device=device, dtype=torch.long))
        logits[0, mask_invalid] = -torch.inf
        action_dist = Categorical(logits=logits.squeeze())
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)

        traj['states'].append(state)
        traj['actions'].append(action)
        traj['log_prob'].append(log_prob)
        
        if action == n_states:
            traj_done = True
            reward = rewards_dict.get(state, 0)
            if do_print:
                print(f"DONE! Reward: {reward}")
        else:
            state = action.item()
            reward = 0
            if do_print:
                print(f"{state} -> ", end="")
        
        traj['rewards'].append(reward)

    # Compute REINFORCE loss and backpropagate
    log_probs = torch.stack(traj['log_prob'])
    rewards = traj['rewards']
    loss = pg_loss(log_probs, rewards)
    
    # Average loss by trajectory length (to match gfn.py)
    loss = loss / len(log_probs)
    
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    
    loss_print = loss.item()
    if do_print:
        print("Loss: {:.4f}".format(loss_print))
    else:
        pbar.update(1)
        pbar.set_description("Loss: {:.4f}".format(loss_print))

end_time = time.time()
print(f"Training time: {end_time - start_time:.2f}s")

### EVALUATION ###

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

for _ in range(n_samples):
    state = 0
    traj_done = False
    while not traj_done:
        mask_invalid = mask_dict[state]
        with torch.no_grad():
            logits = policy(torch.tensor([state], device=device, dtype=torch.long))
        logits[0, mask_invalid] = -torch.inf
        action_dist = Categorical(logits=logits.squeeze())
        action = action_dist.sample()
        
        if action == n_states:
            traj_done = True
            samples_dict[state] += 1
        else:
            state = action.item()

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