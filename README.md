# RL-sim

Agent models for a **spatial combinatorial multi-armed bandit**: at every round an agent selects a subset of `k` arms out of `N`, collects the sum of their rewards and pays a penalty that depends on how far apart the chosen arms are in a `d`-dimensional space.

The agents differ in how much of the environment they know, which makes the repository a small testbed for studying the value of prior knowledge in this setting.

## Contents

| File | Description |
| --- | --- |
| `agent.py` | `AgentThing` base class (empirical arm means, exhaustive subset search over `combinations(N, k)`) and four subclasses: `AgentKnowsItAll`, `AgentGuessesXKnowsL`, `AgentKnowsXGuessesL`, `AgentGuessesAll`, which know or guess the arm locations `x` and the penalty scale `L`. |
| `type_agent.py` | `AgentStartGuesses` - generates initial guesses of arm means and locations (`naive`, `single_noisy`, `sep_noisy`, `dist_based`). |
| `env.py` | Reserved for the environment. Currently empty. |

## Requirements

Python 3.9+, `numpy`, `scipy`.

```bash
pip install numpy scipy
```

## Related work

See [`bandit`](https://github.com/XioDr/bandit) for a vectorised UCB / epsilon-greedy implementation of the same problem.
