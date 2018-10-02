def is_best_agent(space, agent):
    """
    """
    
    if agent.fit < space.best_agent.fit:
        space.best_agent = agent 