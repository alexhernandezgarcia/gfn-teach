import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from tqdm import tqdm

seed = 0
random.seed(seed)
torch.manual_seed(seed)
device = torch.device("cpu")


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
EOS = n_states                    # End of sequence token
PAD = n_states + 1                # Padding token
V = n_states + 2                  # Vocabulary size


def children_non_eos(s):
    return [c for c in connections_dict[s] if c != -1]

def eos_allowed(s):
    return (-1 in connections_dict[s]) and (s in rewards_dict)


# --------------------------------------------------------------------------
# Get all valid trajectories
# --------------------------------------------------------------------------
trajs_by_terminal = {k: [] for k in rewards_dict.keys()}

def dfs(state, path):
    if eos_allowed(state):
        trajs_by_terminal[state].append(path + [EOS])
    for c in children_non_eos(state):
        dfs(c, path + [c])

dfs(0, [0])
all_trajs = [t for ts in trajs_by_terminal.values() for t in ts]
L = max(len(t) for t in all_trajs)

def pad_to_L(seq):
    return seq + [PAD] * (L - len(seq))

for k in trajs_by_terminal:
    trajs_by_terminal[k] = [pad_to_L(t) for t in trajs_by_terminal[k]]


terminals = sorted(trajs_by_terminal.keys())
rewards_vector = torch.tensor([rewards_dict[t] for t in terminals], dtype=torch.float32)
terminal_dist = Categorical(probs=rewards_vector / rewards_vector.sum())


def sample_x0_batch(B):
    idx = terminal_dist.sample((B,)).tolist()
    x0 = []
    for j in idx:
        term = terminals[j]
        x0.append(random.choice(trajs_by_terminal[term]))
    return torch.tensor(x0, dtype=torch.long, device=device)


# --------------------------------------------------------------------------
# Discrete diffusion forward process
# --------------------------------------------------------------------------
T = 5 #20
# betas = torch.linspace(0.02, 0.30, T, device=device)
betas = torch.linspace(0.02, 0.99, T, device=device)
alphas = 1.0 - betas
alpha_bar = torch.cumprod(alphas, dim=0)


def corrupt(x0, t):
    keep_p = alpha_bar[t - 1].view(-1, 1)
    keep = (torch.rand_like(x0.float()) < keep_p)
    noise = torch.randint(0, V, x0.shape, device=device)
    xt = torch.where(keep, x0, noise)

    # No corruption at the start of the sequence
    xt[:, 0] = 0
    keep[:, 0] = True

    corrupted = ~keep
    return xt, corrupted


# --------------------------------------------------------------------------
# Transformer denoiser
# --------------------------------------------------------------------------
class Denoiser(nn.Module):
    def __init__(self, V, L, T, d=128, nhead=4, nlayers=2):
        super().__init__()
        self.tok = nn.Embedding(V, d)
        self.pos = nn.Embedding(L, d)
        self.tim = nn.Embedding(T + 1, d)

        enc_layer = nn.TransformerEncoderLayer(d_model=d, nhead=nhead,
                                               batch_first=True)
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=nlayers)
        self.lin = nn.Linear(d, d)
        self.relu = nn.ReLU()
        self.out = nn.Linear(d, V)

    def forward(self, x, t):
        B, L_ = x.shape
        pos = torch.arange(L_, device=x.device).unsqueeze(0).expand(B, L_)
        h = self.tok(x) + self.pos(pos) + self.tim(t).unsqueeze(1)
        h = self.enc(h)
        h = self.lin(h)
        h = self.relu(h)
        return self.out(h)


model = Denoiser(V, L, T).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

t_dist = Categorical(probs=torch.ones(T, device=device) / T)


# --------------------------------------------------------------------------
# TRAINING LOOP
# --------------------------------------------------------------------------
train_steps = 5000
batch_size = 256

pbar = tqdm(range(train_steps))
for _ in pbar:
    x0 = sample_x0_batch(batch_size)

    t = t_dist.sample((batch_size,)) + 1
    xt, corrupted = corrupt(x0, t)

    logits = model(xt, t)

    # weighting the loss by the length of the sequence
    lengths = (x0 != PAD).sum(dim=1).float()
    weights = (1.0 / lengths)
    weights /= weights.mean()

    if corrupted.any():
        per_token_loss = F.cross_entropy(
            logits.reshape(-1, V), x0.reshape(-1),
            reduction="none"
        ).view(batch_size, L)

        mask_float = corrupted.float()
        weighted = per_token_loss * mask_float * weights.unsqueeze(1)
        loss = weighted.sum() / mask_float.sum()
    else:
        loss = torch.tensor(0.0, device=device)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    pbar.set_description(f"loss={loss.item():.4f}")


# --------------------------------------------------------------------------
# SAMPLER
# --------------------------------------------------------------------------


def allowed_actions_simple(s):
    """Only use children + EOS if allowed. No shortest path constraints."""
    opts = list(children_non_eos(s))
    if eos_allowed(s):
        opts.append(EOS)
    return opts
    
    
@torch.no_grad()
def sample_from_pure_noise_valid(model, n, batch=512, temp=1.5):
    out = []
    pbar = tqdm(total=n, desc="Sampling from Pure Noise")

    while len(out) < n:
        b = min(batch, n - len(out))

        xt = torch.randint(0, V, (b, L), device=device)
        xt[:, 0] = 0

        t_high = torch.full((b,), T, dtype=torch.long, device=device)
        logits_full = model(xt, t_high)

        seq = torch.full((b, L), PAD, dtype=torch.long, device=device)
        seq[:, 0] = 0

        for bi in range(b):
            s = 0
            done = False
            for i in range(1, L):
                if done:
                    seq[bi, i] = PAD
                    continue

                allowed = allowed_actions_simple(s)

                logits_pos = logits_full[bi, i] / temp
                masked = torch.full((V,), -torch.inf, device=device)
                masked[allowed] = logits_pos[allowed]

                tok = Categorical(logits=masked).sample().item()
                seq[bi, i] = tok

                if tok == EOS:
                    done = True
                else:
                    s = tok

        out.append(seq.cpu())
        pbar.update(b)

    pbar.close()
    return torch.cat(out)[:n]


def terminal_of_sequence(seq):
    s = 0
    for i in range(1, L):
        tok = int(seq[i])
        if tok == EOS:
            return s
        s = tok
    raise RuntimeError("Missing EOS")


# --------------------------------------------------------------------------
# EVALUATION
# --------------------------------------------------------------------------
n_samples = 2000
samples = sample_from_pure_noise_valid(model, n_samples, batch=256, temp=1.0)

counts = {k: 0 for k in rewards_dict}
for i in range(n_samples):
    term = terminal_of_sequence(samples[i].tolist())
    counts[term] += 1

print("\nEvaluation:")
Z = sum(rewards_dict.values())
abs_err = 0
for term in sorted(counts):
    p_samp = counts[term] / n_samples
    p_true = rewards_dict[term] / Z
    abs_err += abs(p_samp - p_true)
    print(f"- Sample {term:2d} prob {p_samp:.2f} | true {p_true:.2f}")

print("MAE =", abs_err / len(counts))
