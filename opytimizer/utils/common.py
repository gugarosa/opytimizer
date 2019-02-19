def check_bound_limits(agents, lower_bound, upper_bound):
    """Checks bounds limists of all agents and variables.

    Args:
        agents (list): List of agents.
        lower_bound (np.array): Array holding lower bounds.
        upper_bound (np.array): Array holding upper bounds.

    """

    # Iterate through all agents
    for agent in agents:
        # Iterating through all variables
        for v in range(agent.n_variables):
            # If current position is lower than lower bound
            if agent.position[v] < lower_bound[v]:
                # Bring it back to minimum bound value
                agent.position[v] = lower_bound[v]
            # If current position is greater than upper bound
            elif agent.position[v] > upper_bound[v]:
                # Bring it back to maximum bound value
                agent.position[v] = upper_bound[v]