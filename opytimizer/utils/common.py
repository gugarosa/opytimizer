import numpy as np

def check_bound_limits(agents, lower_bound, upper_bound):
    """Checks bounds limists of all agents and variables.

    Args:
        agents (list): List of agents.
        lower_bound (np.array): Array holding lower bounds.
        upper_bound (np.array): Array holding upper bounds.

    """

    # Iterate through all agents
    for agent in agents:
        # Iterate through all decision variables
        for j, (lb, ub) in enumerate(zip(lower_bound, upper_bound)):
            # Clip the array based on variables' lower and upper bounds
            agent.position[j] = np.clip(agent.position[j], lb, ub)