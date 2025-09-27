from collections import defaultdict
import numpy as np
import csv


class OffPolicyMCAgent:
    def __init__(self, nS, nA):
        self.nS = nS
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(nA, dtype=float))
        self.C = defaultdict(lambda: np.zeros(nA, dtype=float))
        self.pi = defaultdict(lambda: 0)

    def greedy_action(self, s):
        return int(np.argmax(self.Q[s]))

    def act(self, s):
        return self.pi.get(s, self.greedy_action(s))

    def update_policy(self):
        self.pi = {s: self.greedy_action(s) for s in self.Q.keys()}

    def update(self, states, actions, rewards, gamma=1.0, epsilon=0.1):
        G = 0.0
        W = 1.0
        T = len(states)

        for t in reversed(range(T)):
            s, a = states[t], actions[t]
            r = rewards[t]
            G = gamma * G + r

            self.C[s][a] += W
            self.Q[s][a] += (W / self.C[s][a]) * (G - self.Q[s][a])

            self.pi[s] = self.greedy_action(s)

            if a != self.pi[s]:
                break

            b_prob = epsilon / self.nA + (1.0 - epsilon) if a == self.pi[s] else epsilon / self.nA
            W = W / b_prob

    def save_q_to_csv(self, filename="q_values.csv"):
        nA = next(iter(self.Q.values())).shape[0]
        with open(filename, mode="w", newline="") as f:
            writer = csv.writer(f)
            header = ["state"] + [f"Q(a={a})" for a in range(nA)]
            writer.writerow(header)
            for s in sorted(self.Q.keys()):
                writer.writerow([s] + self.Q[s].tolist())
        print(f"Saved Q to {filename}")
