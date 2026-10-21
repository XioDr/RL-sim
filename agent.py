import numpy as np
from scipy.spatial.distance import euclidean
from itertools import combinations


class AgentThing:
    def __init__(self, n, k, d, start_mus, seed =None):
        if seed != None:
            np.random.seed(seed)
        self.N = n
        self.k = k
        self.d = d
        self.my_mus = np.array(start_mus, dtype=float)
        self.pulls = np.zeros(n, dtype=int)
        self.reward_sums = np.zeros(n, dtype=float)

    def whats_my_guess(self, arm):
        if self.pulls[arm] == 0:
            return self.my_mus[arm]
        return self.reward_sums[arm] / self.pulls[arm]

    # Subclasses need to make this work
    def calc_penalty(self, choice):
        pass

    def guess_reward(self, choice):
        if len(choice)!= self.k:
             return -999999 # Bad choice

        r_guess = sum(self.whats_my_guess(i) for i in choice)
        p_guess = self.calc_penalty(choice)
        if p_guess is None: p_guess=0 # Handle missing implementation maybe?
        return r_guess - p_guess

    def pick_next(self):
        best_one = None
        best_r = -np.inf
        candidates = []
        possible = combinations(range(self.N), self.k)

        for p in possible:
            current_choice = list(p)
            r = self.guess_reward(current_choice)
            if r> best_r:
                best_r = r
                candidates = [current_choice]
            elif r == best_r:
                 candidates.append(current_choice)

        if not candidates:
             best_one = list(np.random.choice(self.N, self.k, replace=False))
        else:
            idx = np.random.randint(len(candidates))
            best_one = candidates[idx]

        return best_one

    def learn(self, the_choice, got_reward):
         for i in the_choice:
             self.pulls[i]+=1
         # Base class doesn't update means or anything else much
         pass

#--- Specific Guys ---

class AgentKnowsItAll(AgentThing):
    def __init__(self, n, k, d, start_mus, real_x, real_L, seed=None):
        super().__init__(n, k, d, start_mus, seed)
        self.real_x = np.array(real_x)
        self.the_L = real_L

    def calc_penalty(self, choice):
        if len(choice) < 2: return 0.0
        pairs = 0; total_d = 0.0
        for i in range(len(choice)):
            for j in range(i + 1, len(choice)):
                a, b = choice[i], choice[j]
                loc_a = self.real_x[a].flatten()
                loc_b = self.real_x[b].flatten()
                dist = euclidean(loc_a, loc_b)
                total_d += dist
                pairs += 1
        avg_d = total_d / pairs if pairs > 0 else 0.0
        return self.the_L * avg_d

    def learn(self, the_choice, got_reward):
        super().learn(the_choice, got_reward)
        # Knows it all, learns nothing more here? Mu update maybe?
        pass


class AgentGuessesXKnowsL(AgentThing):
    def __init__(self, n, k, d, start_mus, my_x_guess, real_L, seed=None):
        super().__init__(n, k, d, start_mus, seed)
        self.my_x = np.array(my_x_guess) # Fixed guess
        self.the_L = real_L

    def calc_penalty(self, choice):
        # Use guessed locations
        if len(choice) < 2: return 0.0
        pairs = 0; total_d = 0.0
        for i in range(len(choice)):
            for j in range(i + 1, len(choice)):
                a, b = choice[i], choice[j]
                loc_a = self.my_x[a].flatten() # my guess
                loc_b = self.my_x[b].flatten() # my guess
                dist = euclidean(loc_a, loc_b)
                total_d += dist
                pairs += 1
        avg_d = total_d / pairs if pairs > 0 else 0.0
        return self.the_L * avg_d

    def learn(self, the_choice, got_reward):
        super().learn(the_choice, got_reward)
        # Doesn't learn X or L
        pass


class AgentKnowsXGuessesL(AgentThing):
    def __init__(self, n, k, d, start_mus, real_x, guess_L=1.0, L_rate=0.01, seed=None):
        super().__init__(n, k, d, start_mus, seed)
        self.real_x = np.array(real_x)
        self.my_L = float(guess_L)
        self.L_rate = L_rate

    def calc_penalty(self, choice):
        # Use real locations and guessed lambda
        if len(choice) < 2: return 0.0
        pairs = 0; total_d = 0.0
        for i in range(len(choice)):
            for j in range(i + 1, len(choice)):
                a, b = choice[i], choice[j]
                loc_a = self.real_x[a].flatten() # real one
                loc_b = self.real_x[b].flatten() # real one
                dist = euclidean(loc_a, loc_b)
                total_d += dist
                pairs += 1
        self.last_dist_calc = total_d / pairs if pairs > 0 else 0.0 # store for update
        return self.my_L * self.last_dist_calc

    def learn(self, the_choice, got_reward):
        super().learn(the_choice, got_reward)
        # Update lambda guess
        my_r_guess = sum(self.whats_my_guess(i) for i in the_choice)
        actual_dist = self.last_dist_calc # Retrieve distance used in penalty

        if actual_dist > 1e-9: # close enough to zero?
            L_sample = (my_r_guess - got_reward) / actual_dist
            self.my_L += self.L_rate * (L_sample - self.my_L)
            if self.my_L < 0: self.my_L = 0 # Cant be negative penalty?

class AgentGuessesAll(AgentThing):
    def __init__(self, n, k, d, start_mus, my_x_guess, guess_L=1.0, L_rate=0.01, seed=None):
        super().__init__(n, k, d, start_mus, seed)
        self.my_x = np.array(my_x_guess) # Fixed guess
        self.my_L = float(guess_L)
        self.L_rate = L_rate

    def calc_penalty(self, choice):
        # Use guessed locations and guessed lambda
        if len(choice) < 2: return 0.0
        pairs = 0; total_d = 0.0
        for i in range(len(choice)):
            for j in range(i + 1, len(choice)):
                a, b = choice[i], choice[j]
                loc_a = self.my_x[a].flatten() # my guess
                loc_b = self.my_x[b].flatten() # my guess
                dist = euclidean(loc_a, loc_b)
                total_d += dist
                pairs += 1
        self.last_dist_calc = total_d / pairs if pairs > 0 else 0.0 # store for update
        return self.my_L * self.last_dist_calc


    def learn(self, the_choice, got_reward):
        super().learn(the_choice, got_reward)
         # Update lambda guess based on guessed locations
        my_r_guess = sum(self.whats_my_guess(i) for i in the_choice)
        my_dist_guess = self.last_dist_calc # Use distance calculated with guessed locations

        if my_dist_guess > 1e-9: # avoid zero div
            L_sample = (my_r_guess - got_reward) / my_dist_guess
            self.my_L += self.L_rate * (L_sample - self.my_L)
            if self.my_L < 0: self.my_L = 0 # prevent negative L