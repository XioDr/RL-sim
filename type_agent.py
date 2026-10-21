import numpy as np

class AgentStartGuesses:

    def __init__(self, type, params):
      self.typ = type
      self.prms = params

    def get_em(self, t_mu, t_x, seed=None):
        if seed is not None:
            np.random.seed(seed)

        t_x = np.asarray(t_x)
        N = len(t_mu)
        e_mu = None
        e_x = None

        if self.typ == "naive":
            e_mu = np.zeros(N)
            e_x = np.zeros_like(t_x)

        elif self.typ == "single_noisy":
            s = self.prms.get("sigma", 0.3)
            e_mu = np.random.normal(loc=t_mu, scale=s, size=N)
            e_x = np.random.normal(loc=t_x, scale=s, size=t_x.shape)

        elif self.typ == "sep_noisy":
          s_m = self.prms.get("sigma_mu", 0.2)
          s_x = self.prms.get("sigma_x", 0.4)
          e_mu = np.random.normal(loc=t_mu, scale=s_m, size=N)
          e_x = np.random.normal(loc=t_x, scale=s_x, size=t_x.shape)

        elif self.typ == "dist_based":
            a_loc = np.asarray(self.prms["agent_loc"])
            n_min = self.prms.get("noise_min", 0.1)
            n_slp = self.prms.get("noise_slope", 0.2)
            diffs = t_x - a_loc
            dists = np.linalg.norm(diffs, axis=-1 if t_x.ndim > 1 else None)
            sigs = n_min + n_slp * dists
            e_mu = np.zeros(N)
            e_x = np.zeros_like(t_x)
            for i in range(N):
               current_s = sigs[i]
               e_mu[i] = np.random.normal(loc=t_mu[i], scale=current_s)
               e_x[i] = np.random.normal(loc=t_x[i], scale=current_s, size=t_x[i].shape)

        else:
            raise ValueError(f"dunno this type: {self.typ}")

        return e_mu, e_x