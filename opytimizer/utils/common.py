def check_bound_limits(agents, lower_bound, upper_bound):
    """
    """
    
    for agent in agents:
        for v in range(agent.n_variables):
            if agent.position[v] < lower_bound[v]:
                agent.position[v] = lower_bound[v]
            elif agent.position[v] > upper_bound[v]:
                agent.position[v] = upper_bound[v]