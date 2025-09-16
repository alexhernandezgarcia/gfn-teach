"""
This script trains a minimal version of a GFlowNet with the Flow Matching objective on
the "Toy environment", which is the environment defined in Figure 2 of the GFlowNet
Foundations paper, Bengio et al (JMLR, 2023):

    .. _a link: https://jmlr.org/papers/v24/22-0364.html
"""

import torch
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

### POLICY MODEL ###

# MLP with one hidden layer and ReLU activations. The inputs are 1D and the outputs are
# tensors with dimensionality equal to the number of actions: the number of states plus
# one, for the end-of-sequence (EOS) action
input_dim = 1
n_units = 32
output_dim = n_states + 1
policy = torch.nn.Sequential(
    torch.nn.Linear(
        input_dim,
        n_units,
        dtype=float_type,
        device=device,
    ),
    torch.nn.ReLU(),
    torch.nn.Linear(
        n_units,
        output_dim,
        dtype=float_type,
        device=device,
    ),
    torch.nn.ReLU(),
)

### OPTIMIZER ###

n_train_steps = 10000
learning_rate = 1e-2
momentum = 0.9
optimizer = torch.optim.SGD(policy.parameters(), lr=learning_rate, momentum=momentum)

### TRAIN ###

# Progress bar
if not do_print:
    pbar = tqdm(
        initial=0,
        total=n_train_steps,
    )

for step in range(n_train_steps):

    # Initialize a trajectory with state 0 and trajectory not done
    state = 0
    traj_done = False

    # Initialize loss to zero
    loss = torch.tensor([0.0], dtype=float_type, device=device)

    if do_print:
        print(f"\nIteration {step}")
        print(f"\tTrajectory 0 -> ", end="")

    # Sample actions until trajectory is done
    while not traj_done:

        # Build the mask of invalid actions from the current state
        mask_invalid = [
            False if s in connections_dict[state] else True for s in range(n_states)
        ]
        mask_invalid += [-1 not in connections_dict[state]]

        # Obtain policy log-flows from the current state, mask invalid actions and
        # sample action
        with torch.no_grad():
            logits_sampled = policy(
                torch.tensor([state], dtype=float_type, device=device)
            )
        logits_sampled[mask_invalid] = -torch.inf
        action = Categorical(logits=logits_sampled).sample()

        # Update state, flag of done trajectory and get reward
        if action == n_states:
            traj_done = True
            reward = rewards_dict[state]
            if do_print:
                print(f"DONE! Reward: {reward}")
        else:
            state = action.item()
            reward = 0
            if do_print:
                print(f"{state} -> ", end="")

        # Obtain in-flows:
        # - Get parents of state
        # - Obtain log-flows from each parent to state
        # - Take the log of the sum of the exponential log-flows
        parents = torch.tensor(
            [s for s in range(n_states) if state in connections_dict[s]],
            dtype=float_type,
            device=device,
        ).unsqueeze(-1)
        inflow_logits = policy(parents)[:, action]
        loginflow = torch.logsumexp(inflow_logits, dim=0)

        # Obtain out-flows:
        # - Obtain children of state
        # - Obtain log-flows from the state and mask out transitions that are not
        # children
        # - Take the log of the sum of the exponential log-flows
        # - If the trajectory is done, the log-outflow is just the log-reward
        if traj_done:
            logoutflow = torch.log(
                torch.tensor(reward, dtype=float_type, device=device)
            )
        else:
            children = torch.tensor(
                [s for s in connections_dict[state]], dtype=torch.int, device=device
            )
            outflow_logits = torch.full(
                (output_dim,), -torch.inf, dtype=float_type, device=device
            )
            outflow_logits[children] = policy(
                (torch.tensor([state], dtype=float_type, device=device))
            )[children]
            logoutflow = torch.logsumexp(outflow_logits, dim=0)

        # Compute Flow Matching loss
        loss = loss + (loginflow - logoutflow).pow(2)

    # End of the trajectory: Back propagate and update parameters
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    loss_print = loss.item()
    if do_print:
        print("Loss: {:.4f}".format(loss_print))
    else:
        pbar.update(1)
        pbar.set_description("Loss: {:.4f}".format(loss_print))

### EVALUATE ###

n_samples = 1000

# A dictionary to count the number of times each terminal state is sampled
samples_dict = {
    3: 0,
    4: 0,
    6: 0,
    8: 0,
    9: 0,
    10: 0,
}

for step in range(n_samples):

    # Initialize a trajectory with state 0 and trajectory not done
    state = 0
    traj_done = False

    # Sample actions until trajectory is done
    while not traj_done:

        # Build the mask of invalid actions from the current state
        mask_invalid = [
            False if s in connections_dict[state] else True for s in range(n_states)
        ]
        mask_invalid += [-1 not in connections_dict[state]]

        # Obtain policy log-flows from the current state, mask invalid actions and
        # sample action
        with torch.no_grad():
            logits_sampled = policy(
                torch.tensor([state], dtype=float_type, device=device)
            )
        logits_sampled[mask_invalid] = -torch.inf
        action = Categorical(logits=logits_sampled).sample()

        # Update state, flag of done trajectory and get reward
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
