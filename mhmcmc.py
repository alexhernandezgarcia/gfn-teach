"""
This script runs the Metropolis-Hastings MCMC algorithm on the "Toy environment", which
is the environment defined in Figure 2 of the GFlowNet Foundations paper, Bengio et al
(JMLR, 2023):

    .. _a link: https://jmlr.org/papers/v24/22-0364.html
"""

import random
from tqdm import tqdm

### COMMON VARIABLES ###

transitions_distr = "uniform"

### ENVIRONMENT ###

# The states are only the terminating states of the Toy environment, since in standard
# MCMC there is no notion of intermediate states.
states = (3, 4, 6, 8, 9, 10)
state2index = {s: idx for idx, s in enumerate(states)}
n_states = len(states)

# A dictionary of uniform transition probabilities between states. For every state, the
# probability of transitioning to any other state including itself is the same and
# equal to 1 / n_states.
transitions_uniform_dict = {s: tuple([1.0 / n_states for _ in states]) for s in states}
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

### RUN ###

n_transitions = 2000
if transitions_distr == "uniform":
    transitions = transitions_uniform_dict
else:
    raise NotImplementedError("Valid transition distribution options are: uniform")

# A dictionary to count the number of times each state is visited
samples_dict = {
    3: 0,
    4: 0,
    6: 0,
    8: 0,
    9: 0,
    10: 0,
}

# Initialize the chain at a random state
state = random.choice(states)

for step in tqdm(range(n_transitions)):

    # Sample a new state
    state_proposed = random.choices(states, weights=transitions[state])[0]
    # Compute acceptance probability
    num = rewards_dict[state_proposed] * transitions[state_proposed][state2index[state]]
    den = rewards_dict[state] * transitions[state][state2index[state_proposed]]
    alpha = num / den
    # Compute acceptance rule
    acceptance = min(1, alpha)
    # Sample from uniform distribution
    u = random.random()
    # Transition according to rule
    if u <= acceptance:
        state = state_proposed
    # Record state
    samples_dict[state] += 1

### EVALUATE ###

# Print results
print("\nEvaluation: \n")
z = sum(rewards_dict.values())
absolute_error = 0.0
for sample, count in samples_dict.items():
    p_sampled = count / n_transitions
    p_true = rewards_dict[sample] / z
    absolute_error += abs(p_sampled - p_true)
    print(
        "- Sample {:2d} was generated with probability {:.2f} and the "
        "actual probability is {:.2f}".format(sample, p_sampled, p_true)
    )
mae = absolute_error / len(samples_dict)
print("Mean absolute error: {:.2f}".format(mae))
