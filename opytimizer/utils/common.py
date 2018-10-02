def _is_best_agent(self, agent):
        """
        """
        
        if agent.fit < self.space.best_agent.fit:
            self.space.best_agent = agent