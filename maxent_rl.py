"""
This script trains a minimal Maximum-Entropy RL agent (soft Q-learning) on the
"Toy environment", which is the environment defined in Figure 2 of the GFlowNet
Foundations paper, Bengio et al (JMLR, 2023):

    https://jmlr.org/papers/v24/22-0364.html
"""
"""
Key differences with gfn.py and rl.py:
    - The agent uses trajectory-based soft Q-learning with a softmax(Q) policy and learns via soft Bellman updates
"""

### COMMON VARIABLES ###

import torch
from torch.distributions import Categorical
from tqdm import tqdm

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
eos = n_states
n_actions = n_states + 1

### POLICY ###

# Linear layer following an embedding layer. The inputs are the state indices and the
# outputs are tensors with dimensionality equal to the number of actions: the number of
# states plus one, for the end-of-sequence (EOS) action.
policy = torch.nn.Sequential(
    torch.nn.Embedding(n_states, n_states),
    torch.nn.Linear(
        n_states,
        n_actions,
        dtype=float_type,
        device=device,
    ),
)


### OPTIMIZER ###

optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

### GRAPH MASKS ###

def invalid_mask(state):
    mask = torch.ones(n_actions, dtype=torch.bool, device=device)
    for child in connections_dict[state]:
        if child == -1:
            mask[eos] = False
        else:
            mask[child] = False
    return mask

def masked_states(q_all, state):

    q = q_all[state].clone()
    m = invalid_mask(state)
    q[m] = -torch.inf
    return q

### TRAIN ###

n_episodes = 2000
gamma = 1.0 # No discounting

pbar = tqdm(range(n_episodes))

for ep in pbar:
    transitions = []

    state = 0
    done = False

    while not done:
        q_all = policy(torch.arange(n_states, dtype=torch.long, device=device))
        q_s = masked_states(q_all, state)

        dist = Categorical(logits=q_s)
        action = dist.sample()
        a = action.item()

        if a == eos:
            r = torch.log(torch.tensor(rewards_dict[state], dtype=float_type, device=device))
            next_state = None
            done = True
        else:
            r = torch.tensor(0.0, dtype=float_type, device=device)
            next_state = a

        transitions.append((state, a, r, next_state, done))
        state = next_state if next_state is not None else state

    loss_terms = []

    # Q(s,a) = r + gamma V(next_s)
    # Q(s,EOS) = log R(s)
    for (s, a, r, next_s, done) in transitions:
        q_sa = policy(torch.tensor(s, dtype=torch.long, device=device))[a]

        if done:
            target = r
        else:
            #V(next_s) = logsumexp_a' Q(next_s,a')
            with torch.no_grad():
                q_all_next = policy(torch.arange(n_states, dtype=torch.long, device=device))
                q_next = masked_states(q_all_next, next_s)
                V_next = torch.logsumexp(q_next, dim=0)
                target = r + gamma * V_next

        loss_terms.append((q_sa - target) ** 2)

    # Average loss by trajectory length (to match gfn.py)
    loss = torch.stack(loss_terms).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    pbar.set_description(f"Loss: {loss.item():.4f}")


### EVALUATION ###

n_samples = 2000

# A dictionary to count the number of times each terminal state is sampled
samples_dict = {s: 0 for s in rewards_dict.keys()}

for x in range(n_samples):
    state = 0
    done = False
    while not done:
        with torch.no_grad():
            q_all_eval = policy(torch.arange(n_states, dtype=torch.long, device=device))
            q_s = masked_states(q_all_eval, state)
        action = Categorical(logits=q_s).sample()
        a = action.item()
        if a == eos:
            samples_dict[state] += 1
            done = True
        else:
            state = a

print("\nEvaluation:\n")
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