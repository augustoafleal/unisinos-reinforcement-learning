from algos.DynamicProgrammingAgent import DynamicProgrammingAgent


class PolicyIterationAgent(DynamicProgrammingAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def train(self):
        for state in self.states:
            self.policy[state] = 0

        stable = False
        while not stable:
            while True:
                delta = 0
                V_new = self.V.copy()
                for state in self.states:
                    action = self.policy[state]
                    q = 0
                    for prob, next_state, reward in self.P[(state, action)]:
                        q += prob * (reward + self.gamma * self.V[next_state])
                    delta = max(delta, abs(self.V[state] - q))
                    V_new[state] = q
                    self.states_iteration += 1
                self.V = V_new
                if delta < self.epsilon:
                    break

            stable = True
            for state in self.states:
                old_action = self.policy[state]
                q_values = {}
                for action in self.actions:
                    q = 0
                    for prob, next_state, reward in self.P[(state, action)]:
                        q += prob * (reward + self.gamma * self.V[next_state])
                    q_values[action] = q
                best_action = max(q_values, key=q_values.get)
                self.policy[state] = best_action
                if old_action != best_action:
                    stable = False

            self.iteration += 1

    def act(self, state):
        return self.policy[state]
