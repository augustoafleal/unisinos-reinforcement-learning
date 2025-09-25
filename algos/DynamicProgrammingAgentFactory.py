from algos.ValueIterationAgent import ValueIterationAgent
from algos.PolicyIterationAgent import PolicyIterationAgent


class DynamicProgrammingAgentFactory:
    @staticmethod
    def create(agent_type, grid_size, islands, goal, enemies=None, gamma=0.9, epsilon=1e-4, stochastic=False):
        agent_type = agent_type.lower()

        if agent_type == "value_iteration":
            return ValueIterationAgent(
                grid_size=grid_size,
                islands=islands,
                goal=goal,
                enemies=enemies,
                gamma=gamma,
                epsilon=epsilon,
                stochastic=stochastic,
            )
        elif agent_type == "policy_iteration":
            return PolicyIterationAgent(
                grid_size=grid_size,
                islands=islands,
                goal=goal,
                enemies=enemies,
                gamma=gamma,
                epsilon=epsilon,
                stochastic=stochastic,
            )
        else:
            raise ValueError(f"Unknown agent: {agent_type}")
