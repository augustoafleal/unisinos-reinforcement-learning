from algos.DynamicProgrammingAgent import DynamicProgrammingAgent


class ValueIterationAgent(DynamicProgrammingAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def train(self):
        while True:
            delta = 0
            V_new = self.V.copy()
            for state in self.states:
                v = self.V[state]
                pos_index, visited = state
                x, y = self.position_from_index(pos_index)
                if (x, y) == self.goal and all(visited):
                    continue

                q_values = []
                for action in self.actions:
                    q = 0
                    for prob, next_state, reward in self.P[(state, action)]:
                        q += prob * (reward + self.gamma * self.V[next_state])
                    q_values.append(q)

                V_new[state] = max(q_values)
                delta = max(delta, abs(v - V_new[state]))
                self.states_iteration += 1

            self.V = V_new
            self.iteration += 1
            if delta < self.epsilon:
                break

        for state in self.states:
            pos_index, visited = state
            x, y = self.position_from_index(pos_index)
            if (x, y) == self.goal and all(visited):
                self.policy[state] = None
                continue

            q_values = {}
            for action in self.actions:
                q = 0
                for prob, next_state, reward in self.P[(state, action)]:
                    q += prob * (reward + self.gamma * self.V[next_state])
                q_values[action] = q

            self.policy[state] = max(q_values, key=q_values.get)

    def act(self, state):
        return self.policy[state]
