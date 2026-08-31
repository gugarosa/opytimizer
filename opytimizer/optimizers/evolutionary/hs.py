"""Harmony Search-based algorithms."""

import copy
from typing import Any, Callable, Dict, List, Optional

import numpy as np

import opytimizer.utils.constant as c
from opytimizer.core import Optimizer
from opytimizer.core.agent import Agent
from opytimizer.core.space import Space


class HS(Optimizer):
    """A HS class, inherited from Optimizer.

    This is the designed class to define HS-related
    variables and methods.

    References:
        Z. W. Geem, J. H. Kim, and G. V. Loganathan.
        A new heuristic optimization algorithm: Harmony search. Simulation (2001).

    """

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        """Initialization method.

        Args:
            params: Contains key-value parameters to the meta-heuristics.

        """

        super(HS, self).__init__()

        self.HMCR = 0.7
        self.PAR = 0.7
        self.bw = 1.0

        self.build(params)

    def _generate_new_harmony(self, agents: List[Agent]) -> Agent:
        """It generates a new harmony.

        Args:
            agents: List of agents.

        Returns:
            (Agent): A new agent (harmony) based on music generation process.

        """

        a = copy.deepcopy(agents[0])

        for j, (lb, ub) in enumerate(zip(a.lb, a.ub)):
            r1 = np.random.uniform(0.0, 1.0, 1)
            if r1 <= self.HMCR:
                k = np.random.randint(0, len(agents), None)
                a.position[j] = agents[k].position[j]

                r2 = np.random.uniform(0.0, 1.0, 1)
                if r2 <= self.PAR:
                    r3 = np.random.uniform(-1, 1, 1)
                    a.position[j] += r3 * self.bw
            else:
                a.position[j] = np.random.uniform(lb, ub, a.n_dimensions)

        return a

    def update(self, space: Space, function: Callable) -> None:
        """Wraps Harmony Search over all agents and variables.

        Args:
            space: Space containing agents and update-related information.
            function: A callable that will be used as the objective function.

        """

        agent = self._generate_new_harmony(space.agents)
        agent.clip_by_bound()

        agent.fit = function(agent.position)

        space.agents.sort(key=lambda x: x.fit)

        if agent.fit < space.agents[-1].fit:
            space.agents[-1].position = copy.deepcopy(agent.position)
            space.agents[-1].fit = copy.deepcopy(agent.fit)


class IHS(HS):
    """An IHS class, inherited from HS.

    This is the designed class to define IHS-related
    variables and methods.

    References:
        M. Mahdavi, M. Fesanghary, and E. Damangir.
        An improved harmony search algorithm for solving optimization problems.
        Applied Mathematics and Computation (2007).

    """

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        """Initialization method.

        Args:
            params: Contains key-value parameters to the meta-heuristics.

        """

        self.PAR_min = 0
        self.PAR_max = 1

        self.bw_min = 1
        self.bw_max = 10

        super(IHS, self).__init__(params)

    def update(
        self, space: Space, function: Callable, iteration: int, n_iterations: int
    ) -> None:
        """Wraps Improved Harmony Search over all agents and variables.

        Args:
            space: Space containing agents and update-related information.
            function: A callable that will be used as the objective function.
            iteration: Current iteration.
            n_iterations: Maximum number of iterations.

        """

        self.PAR = self.PAR_min + (
            ((self.PAR_max - self.PAR_min) / n_iterations) * iteration
        )

        self.bw = self.bw_max * np.exp(
            (np.log(self.bw_min / self.bw_max) / n_iterations) * iteration
        )

        agent = self._generate_new_harmony(space.agents)
        agent.clip_by_bound()

        agent.fit = function(agent.position)

        space.agents.sort(key=lambda x: x.fit)

        if agent.fit < space.agents[-1].fit:
            space.agents[-1].position = copy.deepcopy(agent.position)
            space.agents[-1].fit = copy.deepcopy(agent.fit)


class GHS(IHS):
    """A GHS class, inherited from IHS.

    This is the designed class to define GHS-related
    variables and methods.

    References:
        M. Omran and M. Mahdavi. Global-best harmony search.
        Applied Mathematics and Computation (2008).

    """

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        """Initialization method.

        Args:
            params: Contains key-value parameters to the meta-heuristics.

        """

        super(GHS, self).__init__(params)

    def _generate_new_harmony(self, agents: List[Agent]) -> Agent:
        """It generates a new harmony.

        Args:
            agents: List of agents.

        Returns:
            (Agent): A new agent (harmony) based on music generation process.

        """

        a = copy.deepcopy(agents[0])

        for j, (lb, ub) in enumerate(zip(a.lb, a.ub)):
            r1 = np.random.uniform(0.0, 1.0, 1)
            if r1 <= self.HMCR:
                k = np.random.randint(0, len(agents), None)
                a.position[j] = agents[k].position[j]

                r2 = np.random.uniform(0.0, 1.0, 1)
                if r2 <= self.PAR:
                    z = np.random.randint(0, a.n_variables, None)
                    a.position[j] = agents[0].position[z]
            else:
                a.position[j] = np.random.uniform(lb, ub, a.n_dimensions)

        return a


class SGHS(HS):
    """A SGHS class, inherited from HS.

    This is the designed class to define SGHS-related
    variables and methods.

    References:
        Q.-K. Pan, P. Suganthan, M. Tasgetiren and J. Liang.
        A self-adaptive global best harmony search algorithm for continuous optimization problems.
        Applied Mathematics and Computation (2010).

    """

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        """Initialization method.

        Args:
            params: Contains key-value parameters to the meta-heuristics.

        """

        self.LP = 100

        self.HMCRm = 0.98
        self.PARm = 0.9

        self.bw_min = 1
        self.bw_max = 10

        super(SGHS, self).__init__(params)

    def compile(self, space: Space) -> None:
        """Compiles additional information that is used by this optimizer.

        Args:
            space: A Space object containing meta-information.

        """

        self.lp = 1

        self.HMCR_history = []
        self.PAR_history = []

    def _generate_new_harmony(self, agents: List[Agent]) -> Agent:
        """It generates a new harmony.

        Args:
            agents: List of agents.

        Returns:
            (Agent): A new agent (harmony) based on music generation process.

        """

        a = copy.deepcopy(agents[0])

        for j, (lb, ub) in enumerate(zip(a.lb, a.ub)):
            r1 = np.random.uniform(0.0, 1.0, 1)
            if r1 <= self.HMCR:
                r2 = np.random.uniform(-1, 1, 1)
                a.position[j] += r2 * self.bw

                r3 = np.random.uniform(0.0, 1.0, 1)
                if r3 <= self.PAR:
                    a.position[j] = agents[0].position[j]
            else:
                a.position[j] = np.random.uniform(lb, ub, a.n_dimensions)

        return a

    def update(
        self, space: Space, function: Callable, iteration: int, n_iterations: int
    ) -> None:
        """Wraps Self-Adaptive Global-Best Harmony Search over all agents and variables.

        Args:
            space: Space containing agents and update-related information.
            function: A callable that will be used as the objective function.
            iteration: Current iteration.
            n_iterations: Maximum number of iterations.

        """

        self.HMCR = np.random.normal(self.HMCRm, 0.01, 1)[0]
        self.PAR = np.random.normal(self.PARm, 0.05, 1)[0]

        self.HMCR_history.append(self.HMCR)
        self.PAR_history.append(self.PAR)

        if iteration < n_iterations // 2:
            self.bw = (
                self.bw_max
                - ((self.bw_max - self.bw_min) / n_iterations) * 2 * iteration
            )
        else:
            self.bw = self.bw_min

        agent = self._generate_new_harmony(space.agents)
        agent.clip_by_bound()

        agent.fit = function(agent.position)

        space.agents.sort(key=lambda x: x.fit)

        if agent.fit < space.agents[-1].fit:
            space.agents[-1].position = copy.deepcopy(agent.position)
            space.agents[-1].fit = copy.deepcopy(agent.fit)

        if self.lp == self.LP:
            self.HMCRm = np.mean(self.HMCR_history)
            self.PARm = np.mean(self.PAR_history)
            self.lp = 1
        else:
            self.lp += 1


class NGHS(HS):
    """A NGHS class, inherited from HS.

    This is the designed class to define NGHS-related
    variables and methods.

    References:
        D. Zou, L. Gao, J. Wu and S. Li.
        Novel global harmony search algorithm for unconstrained problems.
        Neurocomputing (2010).

    """

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        """Initialization method.

        Args:
            params: Contains key-value parameters to the meta-heuristics.

        """

        self.pm = 0.1

        super(NGHS, self).__init__(params)

    def _generate_new_harmony(self, best: Agent, worst: Agent) -> Agent:
        """It generates a new harmony.

        Args:
            best: Best agent.
            worst: Worst agent.

        Returns:
            (Agent): A new agent (harmony) based on music generation process.

        """

        a = copy.deepcopy(best)

        for j, (lb, ub) in enumerate(zip(a.lb, a.ub)):
            new_position = 2 * (best.position[j] - worst.position[j])
            new_position = np.clip(new_position, lb, ub)

            r1 = np.random.uniform(0.0, 1.0, 1)

            a.position[j] = worst.position[j] + r1 * (new_position - worst.position[j])

            r2 = np.random.uniform(0.0, 1.0, 1)
            if r2 <= self.pm:
                a.position[j] = np.random.uniform(lb, ub, a.n_dimensions)

        return a

    def update(self, space: Space, function: Callable) -> None:
        """Wraps Novel Global Harmony Search over all agents and variables.

        Args:
            space: Space containing agents and update-related information.
            function: A callable that will be used as the objective function.

        """

        agent = self._generate_new_harmony(space.agents[0], space.agents[-1])
        agent.clip_by_bound()

        agent.fit = function(agent.position)

        space.agents.sort(key=lambda x: x.fit)

        space.agents[-1].position = copy.deepcopy(agent.position)
        space.agents[-1].fit = copy.deepcopy(agent.fit)


class GOGHS(NGHS):
    """A GOGHS class, inherited from NGHS.

    This is the designed class to define GOGHS-related
    variables and methods.

    References:
        Z. Guo, S. Wang, X. Yue and H. Yang.
        Global harmony search with generalized opposition-based learning.
        Soft Computing (2017).

    """

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        """Initialization method.

        Args:
            params: Contains key-value parameters to the meta-heuristics.

        """

        super(GOGHS, self).__init__(params)

    def _generate_opposition_harmony(
        self, new_agent: Agent, agents: List[Agent]
    ) -> Agent:
        """It generates a new opposition-based harmony.

        Args:
            new_agent: Newly created agent.
            agents: List of agents.

        Returns:
            (Agent): A new agent (harmony) based on opposition generation process.

        """

        a = copy.deepcopy(agents[0])

        A = np.zeros((a.n_variables))
        B = np.zeros((a.n_variables))

        k = np.random.uniform(0.0, 1.0, 1)

        for j in range(a.n_variables):
            A[j], B[j] = c.FLOAT_MAX, -c.FLOAT_MAX

            for agent in agents:
                position = agent.position[j].item()
                if A[j] > position:
                    A[j] = position
                elif B[j] < position:
                    B[j] = position

            a.position[j] = k * (A[j] + B[j]) - new_agent.position[j]

        return a

    def update(self, space: Space, function: Callable) -> None:
        """Wraps Generalized Opposition Global-Best Harmony Search over all agents and variables.

        Args:
            space: Space containing agents and update-related information.
            function: A callable that will be used as the objective function.

        """

        agent = self._generate_new_harmony(space.agents[0], space.agents[-1])
        opp_agent = self._generate_opposition_harmony(agent, space.agents)

        agent.clip_by_bound()
        opp_agent.clip_by_bound()

        agent.fit = function(agent.position)
        opp_agent.fit = function(opp_agent.position)
        if opp_agent.fit < agent.fit:
            agent = copy.deepcopy(opp_agent)

        space.agents.sort(key=lambda x: x.fit)

        if agent.fit < space.agents[-1].fit:
            space.agents[-1].position = copy.deepcopy(agent.position)
            space.agents[-1].fit = copy.deepcopy(agent.fit)
