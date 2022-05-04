"""Harmony Search-based algorithms.
"""

import copy
from typing import Any, Dict, List, Optional

import numpy as np

import opytimizer.math.random as r
import opytimizer.utils.constant as c
import opytimizer.utils.exception as e
from opytimizer.core import Optimizer
from opytimizer.core.agent import Agent
from opytimizer.core.function import Function
from opytimizer.core.space import Space
from opytimizer.utils import logging

logger = logging.get_logger(__name__)


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

        logger.info("Overriding class: Optimizer -> HS.")

        # Overrides its parent class with the receiving params
        super(HS, self).__init__()

        # Harmony memory considering rate
        self.HMCR = 0.7

        # Pitch adjusting rate
        self.PAR = 0.7

        # Bandwidth parameter
        self.bw = 1.0

        # Builds the class
        self.build(params)

        logger.info("Class overrided.")

    @property
    def HMCR(self) -> float:
        """Harmony memory considering rate."""

        return self._HMCR

    @HMCR.setter
    def HMCR(self, HMCR: float) -> None:
        if not isinstance(HMCR, (float, int)):
            raise e.TypeError("`HMCR` should be a float or integer")
        if HMCR < 0 or HMCR > 1:
            raise e.ValueError("`HMCR` should be between 0 and 1")

        self._HMCR = HMCR

    @property
    def PAR(self) -> float:
        """Pitch adjusting rate."""

        return self._PAR

    @PAR.setter
    def PAR(self, PAR: float) -> None:
        if not isinstance(PAR, (float, int)):
            raise e.TypeError("`PAR` should be a float or integer")
        if PAR < 0 or PAR > 1:
            raise e.ValueError("`PAR` should be between 0 and 1")

        self._PAR = PAR

    @property
    def bw(self) -> float:
        """Bandwidth parameter."""

        return self._bw

    @bw.setter
    def bw(self, bw: float) -> None:
        if not isinstance(bw, (float, int)):
            raise e.TypeError("`bw` should be a float or integer")
        if bw < 0:
            raise e.ValueError("`bw` should be >= 0")

        self._bw = bw

    def _generate_new_harmony(self, agents: List[Agent]) -> Agent:
        """It generates a new harmony.

        Args:
            agents: List of agents.

        Returns:
            (Agent): A new agent (harmony) based on music generation process.

        """

        # Mimics an agent position
        a = copy.deepcopy(agents[0])

        # For every decision variable
        for j, (lb, ub) in enumerate(zip(a.lb, a.ub)):
            # Generates an uniform random number
            r1 = r.generate_uniform_random_number()

            # Using the harmony memory
            if r1 <= self.HMCR:
                # Generates a random index
                k = r.generate_integer_random_number(0, len(agents))

                # Replaces the position with agent `k`
                a.position[j] = agents[k].position[j]

                # Generates a new uniform random number
                r2 = r.generate_uniform_random_number()

                # Checks if it needs a pitch adjusting
                if r2 <= self.PAR:
                    # Generates a final random number
                    r3 = r.generate_uniform_random_number(-1, 1)

                    # Updates harmony position
                    a.position[j] += r3 * self.bw

            # If harmony memory is not used
            else:
                # Generate a uniform random number
                a.position[j] = r.generate_uniform_random_number(
                    lb, ub, size=a.n_dimensions
                )

        return a

    def update(self, space: Space, function: Function) -> None:
        """Wraps Harmony Search over all agents and variables.

        Args:
            space: Space containing agents and update-related information.
            function: A Function object that will be used as the objective function.

        """

        # Generates a new harmony
        agent = self._generate_new_harmony(space.agents)

        # Checks agent limits
        agent.clip_by_bound()

        # Calculates the new harmony fitness
        agent.fit = function(agent.position)

        # Sorts agents
        space.agents.sort(key=lambda x: x.fit)

        # If newly generated agent fitness is better
        if agent.fit < space.agents[-1].fit:
            # Updates the corresponding agent's position and fitness
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

        logger.info("Overriding class: HS -> IHS.")

        # Minimum pitch adjusting rate
        self.PAR_min = 0

        # Maximum pitch adjusting rate
        self.PAR_max = 1

        # Minimum bandwidth parameter
        self.bw_min = 1

        # Maximum bandwidth parameter
        self.bw_max = 10

        # Overrides its parent class with the receiving params
        super(IHS, self).__init__(params)

        logger.info("Class overrided.")

    @property
    def PAR_min(self) -> float:
        """Minimum pitch adjusting rate."""

        return self._PAR_min

    @PAR_min.setter
    def PAR_min(self, PAR_min: float) -> None:
        if not isinstance(PAR_min, (float, int)):
            raise e.TypeError("`PAR_min` should be a float or integer")
        if PAR_min < 0 or PAR_min > 1:
            raise e.ValueError("`PAR_min` should be between 0 and 1")

        self._PAR_min = PAR_min

    @property
    def PAR_max(self) -> float:
        """Maximum pitch adjusting rate."""

        return self._PAR_max

    @PAR_max.setter
    def PAR_max(self, PAR_max: float) -> None:
        if not isinstance(PAR_max, (float, int)):
            raise e.TypeError("`PAR_max` should be a float or integer")
        if PAR_max < 0 or PAR_max > 1:
            raise e.ValueError("`PAR_max` should be between 0 and 1")
        if PAR_max < self.PAR_min:
            raise e.ValueError("`PAR_max` should be >= `PAR_min`")

        self._PAR_max = PAR_max

    @property
    def bw_min(self) -> float:
        """Minimum bandwidth parameter."""

        return self._bw_min

    @bw_min.setter
    def bw_min(self, bw_min: float) -> None:
        if not isinstance(bw_min, (float, int)):
            raise e.TypeError("`bw_min` should be a float or integer")
        if bw_min < 0:
            raise e.ValueError("`bw_min` should be >= 0")

        self._bw_min = bw_min

    @property
    def bw_max(self) -> float:
        """Maximum bandwidth parameter."""

        return self._bw_max

    @bw_max.setter
    def bw_max(self, bw_max: float) -> None:
        if not isinstance(bw_max, (float, int)):
            raise e.TypeError("`bw_max` should be a float or integer")
        if bw_max < 0:
            raise e.ValueError("`bw_max` should be >= 0")
        if bw_max < self.bw_min:
            raise e.ValueError("`bw_max` should be >= `bw_min`")

        self._bw_max = bw_max

    def update(
        self, space: Space, function: Function, iteration: int, n_iterations: int
    ) -> None:
        """Wraps Improved Harmony Search over all agents and variables.

        Args:
            space: Space containing agents and update-related information.
            function: A Function object that will be used as the objective function.
            iteration: Current iteration.
            n_iterations: Maximum number of iterations.

        """

        # Updates pitch adjusting rate
        self.PAR = self.PAR_min + (
            ((self.PAR_max - self.PAR_min) / n_iterations) * iteration
        )

        # Updates bandwidth parameter
        self.bw = self.bw_max * np.exp(
            (np.log(self.bw_min / self.bw_max) / n_iterations) * iteration
        )

        # Generates a new harmony
        agent = self._generate_new_harmony(space.agents)

        # Checks agent limits
        agent.clip_by_bound()

        # Calculates the new harmony fitness
        agent.fit = function(agent.position)

        # Sorts agents
        space.agents.sort(key=lambda x: x.fit)

        # If newly generated agent fitness is better
        if agent.fit < space.agents[-1].fit:
            # Updates the corresponding agent's position and fitness
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

        logger.info("Overriding class: IHS -> GHS.")

        # Overrides its parent class with the receiving params
        super(GHS, self).__init__(params)

        logger.info("Class overrided.")

    def _generate_new_harmony(self, agents: List[Agent]) -> Agent:
        """It generates a new harmony.

        Args:
            agents: List of agents.

        Returns:
            (Agent): A new agent (harmony) based on music generation process.

        """

        # Mimics an agent position
        a = copy.deepcopy(agents[0])

        # For every decision variable
        for j, (lb, ub) in enumerate(zip(a.lb, a.ub)):
            # Generates an uniform random number
            r1 = r.generate_uniform_random_number()

            # Using the harmony memory
            if r1 <= self.HMCR:
                # Generates a random index
                k = r.generate_integer_random_number(0, len(agents))

                # Replaces the position with agent `k`
                a.position[j] = agents[k].position[j]

                # Generates a new uniform random number
                r2 = r.generate_uniform_random_number()

                # Checks if it needs a pitch adjusting
                if r2 <= self.PAR:
                    # Generates a random index
                    z = r.generate_integer_random_number(0, a.n_variables)

                    # Updates harmony position
                    a.position[j] = agents[0].position[z]

            # If harmony memory is not used
            else:
                # Generate a uniform random number
                a.position[j] = r.generate_uniform_random_number(
                    lb, ub, size=a.n_dimensions
                )

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

        logger.info("Overriding class: HS -> SGHS.")

        # Learning period
        self.LP = 100

        # Mean harmony memory considering rate
        self.HMCRm = 0.98

        # Mean pitch adjusting rate
        self.PARm = 0.9

        # Minimum bandwidth parameter
        self.bw_min = 1

        # Maximum bandwidth parameter
        self.bw_max = 10

        # Overrides its parent class with the receiving params
        super(SGHS, self).__init__(params)

        logger.info("Class overrided.")

    @property
    def HMCR(self) -> float:
        """Harmony memory considering rate."""

        return self._HMCR

    @HMCR.setter
    def HMCR(self, HMCR: float) -> None:
        if not isinstance(HMCR, (float, int)):
            raise e.TypeError("`HMCR` should be a float or integer")

        self._HMCR = HMCR

    @property
    def PAR(self) -> float:
        """Pitch adjusting rate."""

        return self._PAR

    @PAR.setter
    def PAR(self, PAR: float) -> None:
        if not isinstance(PAR, (float, int)):
            raise e.TypeError("`PAR` should be a float or integer")

        self._PAR = PAR

    @property
    def LP(self) -> int:
        """Learning period."""

        return self._LP

    @LP.setter
    def LP(self, LP: int) -> None:
        if not isinstance(LP, int):
            raise e.TypeError("`LP` should be a integer")
        if LP <= 0:
            raise e.ValueError("`LP` should be > 0")

        self._LP = LP

    @property
    def HMCRm(self) -> float:
        """Mean harmony memory considering rate."""

        return self._HMCRm

    @HMCRm.setter
    def HMCRm(self, HMCRm: float) -> None:
        if not isinstance(HMCRm, (float, int)):
            raise e.TypeError("`HMCRm` should be a float or integer")
        if HMCRm < 0 or HMCRm > 1:
            raise e.ValueError("`HMCRm` should be between 0 and 1")

        self._HMCRm = HMCRm

    @property
    def PARm(self) -> float:
        """Mean pitch adjusting rate."""

        return self._PARm

    @PARm.setter
    def PARm(self, PARm: float) -> None:
        if not isinstance(PARm, (float, int)):
            raise e.TypeError("`PARm` should be a float or integer")
        if PARm < 0 or PARm > 1:
            raise e.ValueError("`PARm` should be between 0 and 1")

        self._PARm = PARm

    @property
    def bw_min(self) -> float:
        """Minimum bandwidth parameter."""

        return self._bw_min

    @bw_min.setter
    def bw_min(self, bw_min: float) -> None:
        if not isinstance(bw_min, (float, int)):
            raise e.TypeError("`bw_min` should be a float or integer")
        if bw_min < 0:
            raise e.ValueError("`bw_min` should be >= 0")

        self._bw_min = bw_min

    @property
    def bw_max(self) -> float:
        """Maximum bandwidth parameter."""

        return self._bw_max

    @bw_max.setter
    def bw_max(self, bw_max: float) -> None:
        if not isinstance(bw_max, (float, int)):
            raise e.TypeError("`bw_max` should be a float or integer")
        if bw_max < 0:
            raise e.ValueError("`bw_max` should be >= 0")
        if bw_max < self.bw_min:
            raise e.ValueError("`bw_max` should be >= `bw_min`")

        self._bw_max = bw_max

    @property
    def lp(self) -> int:
        """Current learning period."""

        return self._lp

    @lp.setter
    def lp(self, lp: int) -> None:
        if not isinstance(lp, int):
            raise e.TypeError("`lp` should be a integer")
        if lp <= 0:
            raise e.ValueError("`lp` should be > 0")

        self._lp = lp

    @property
    def HMCR_history(self) -> List[float]:
        """Historical harmony memory considering rates."""

        return self._HMCR_history

    @HMCR_history.setter
    def HMCR_history(self, HMCR_history: List[float]) -> None:
        if not isinstance(HMCR_history, list):
            raise e.TypeError("`HMCR_history` should be a list")

        self._HMCR_history = HMCR_history

    @property
    def PAR_history(self) -> List[float]:
        """Historical pitch adjusting rates."""

        return self._PAR_history

    @PAR_history.setter
    def PAR_history(self, PAR_history: List[float]) -> None:
        if not isinstance(PAR_history, list):
            raise e.TypeError("`PAR_history` should be a list")

        self._PAR_history = PAR_history

    def compile(self, space: Space) -> None:
        """Compiles additional information that is used by this optimizer.

        Args:
            space: A Space object containing meta-information.

        """

        # Current learning period
        self.lp = 1

        # Historical HMCRs and PARs
        self.HMCR_history = []
        self.PAR_history = []

    def _generate_new_harmony(self, agents: List[Agent]) -> Agent:
        """It generates a new harmony.

        Args:
            agents: List of agents.

        Returns:
            (Agent): A new agent (harmony) based on music generation process.

        """

        # Mimics an agent position
        a = copy.deepcopy(agents[0])

        # For every decision variable
        for j, (lb, ub) in enumerate(zip(a.lb, a.ub)):
            # Generates an uniform random number
            r1 = r.generate_uniform_random_number()

            # Using the harmony memory
            if r1 <= self.HMCR:
                # Generates a uniform random number
                r2 = r.generate_uniform_random_number(-1, 1)

                # Updates harmony position
                a.position[j] += r2 * self.bw

                # Generates a new uniform random number
                r3 = r.generate_uniform_random_number()

                # Checks if it needs a pitch adjusting
                if r3 <= self.PAR:
                    # Updates harmony position
                    a.position[j] = agents[0].position[j]

            # If harmony memory is not used
            else:
                # Generate a uniform random number
                a.position[j] = r.generate_uniform_random_number(
                    lb, ub, size=a.n_dimensions
                )

        return a

    def update(
        self, space: Space, function: Function, iteration: int, n_iterations: int
    ) -> None:
        """Wraps Self-Adaptive Global-Best Harmony Search over all agents and variables.

        Args:
            space: Space containing agents and update-related information.
            function: A Function object that will be used as the objective function.
            iteration: Current iteration.
            n_iterations: Maximum number of iterations.

        """

        # Updates harmony memory considering and pitch adjusting rates
        self.HMCR = r.generate_gaussian_random_number(self.HMCRm, 0.01)[0]
        self.PAR = r.generate_gaussian_random_number(self.PARm, 0.05)[0]

        # Stores updates values to lists
        self.HMCR_history.append(self.HMCR)
        self.PAR_history.append(self.PAR)

        # If current iteration is smaller than half
        if iteration < n_iterations // 2:
            # Updates the bandwidth parameter
            self.bw = (
                self.bw_max
                - ((self.bw_max - self.bw_min) / n_iterations) * 2 * iteration
            )
        else:
            # Replaces by the minimum bandwidth
            self.bw = self.bw_min

        # Generates a new harmony
        agent = self._generate_new_harmony(space.agents)

        # Checks agent limits
        agent.clip_by_bound()

        # Calculates the new harmony fitness
        agent.fit = function(agent.position)

        # Sorts agents
        space.agents.sort(key=lambda x: x.fit)

        # If newly generated agent fitness is better
        if agent.fit < space.agents[-1].fit:
            # Updates the corresponding agent's position and fitness
            space.agents[-1].position = copy.deepcopy(agent.position)
            space.agents[-1].fit = copy.deepcopy(agent.fit)

        # Checks if learning period has reached its maximum
        if self.lp == self.LP:
            # Re-calculates the mean HMCR and PAR, and resets learning period
            self.HMCRm = np.mean(self.HMCR_history)
            self.PARm = np.mean(self.PAR_history)
            self.lp = 1
        else:
            # Increases learning period
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

        logger.info("Overriding class: HS -> NGHS.")

        # Mutation probability
        self.pm = 0.1

        # Overrides its parent class with the receiving params
        super(NGHS, self).__init__(params)

        logger.info("Class overrided.")

    @property
    def pm(self) -> float:
        """Mutation probability."""

        return self._pm

    @pm.setter
    def pm(self, pm: float) -> None:
        if not isinstance(pm, (float, int)):
            raise e.TypeError("`pm` should be a float or integer")
        if pm < 0 or pm > 1:
            raise e.ValueError("`pm` should be between 0 and 1")

        self._pm = pm

    def _generate_new_harmony(self, best: Agent, worst: Agent) -> Agent:
        """It generates a new harmony.

        Args:
            best: Best agent.
            worst: Worst agent.

        Returns:
            (Agent): A new agent (harmony) based on music generation process.

        """

        # Mimics an agent position
        a = copy.deepcopy(best)

        # For every decision variable
        for j, (lb, ub) in enumerate(zip(a.lb, a.ub)):
            # Updates the harmony position
            new_position = 2 * (best.position[j] - worst.position[j])

            # Clips the harmony position between lower and upper bounds
            new_position = np.clip(new_position, lb, ub)

            # Generates a uniform random number
            r1 = r.generate_uniform_random_number()

            # Updates current agent's position
            a.position[j] = worst.position[j] + r1 * (new_position - worst.position[j])

            # Generates another uniform random number
            r2 = r.generate_uniform_random_number()

            # Checks if is supposed to be mutated
            if r2 <= self.pm:
                # Mutates the position
                a.position[j] = r.generate_uniform_random_number(
                    lb, ub, size=a.n_dimensions
                )

        return a

    def update(self, space: Space, function: Function) -> None:
        """Wraps Novel Global Harmony Search over all agents and variables.

        Args:
            space: Space containing agents and update-related information.
            function: A Function object that will be used as the objective function.

        """

        # Generates a new harmony
        agent = self._generate_new_harmony(space.agents[0], space.agents[-1])

        # Checks agent limits
        agent.clip_by_bound()

        # Calculates the new harmony fitness
        agent.fit = function(agent.position)

        # Sorts agents
        space.agents.sort(key=lambda x: x.fit)

        # Updates the worst agent's position and fitness
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

        logger.info("Overriding class: NGHS -> GOGHS.")

        # Overrides its parent class with the receiving params
        super(GOGHS, self).__init__(params)

        logger.info("Class overrided.")

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

        # Mimics an agent position
        a = copy.deepcopy(agents[0])

        # Creates pseudo-harmonies
        A = np.zeros((a.n_variables))
        B = np.zeros((a.n_variables))

        # Generates a new uniform random number
        k = r.generate_uniform_random_number()

        # Iterates over every variable
        for j in range(a.n_variables):
            # Defines to `A` and `B` maximum and minimum values, respectively
            A[j], B[j] = c.FLOAT_MAX, -c.FLOAT_MAX

            # Iterates over every agent
            for agent in agents:
                # If `A` is bigger than agent's position
                if A[j] > agent.position[j]:
                    # Replaces its value
                    A[j] = agent.position[j]

                # If `B` is smaller than agent's position
                elif B[j] < agent.position[j]:
                    # Replaces its value
                    B[j] = agent.position[j]

            # Calculates new agent's position
            a.position[j] = k * (A[j] + B[j]) - new_agent.position[j]

        return a

    def update(self, space: Space, function: Function) -> None:
        """Wraps Generalized Opposition Global-Best Harmony Search over all agents and variables.

        Args:
            space: Space containing agents and update-related information.
            function: A Function object that will be used as the objective function.

        """

        # Generates new harmonies
        agent = self._generate_new_harmony(space.agents[0], space.agents[-1])
        opp_agent = self._generate_opposition_harmony(agent, space.agents)

        # Checks agents limits
        agent.clip_by_bound()
        opp_agent.clip_by_bound()

        # Calculates harmonies fitness
        agent.fit = function(agent.position)
        opp_agent.fit = function(opp_agent.position)

        # Checks if oppisition-based is better than agent
        if opp_agent.fit < agent.fit:
            # Copies the agent
            agent = copy.deepcopy(opp_agent)

        # Sorts agents
        space.agents.sort(key=lambda x: x.fit)

        # If generated agent fitness is better
        if agent.fit < space.agents[-1].fit:
            # Updates the corresponding agent's position and fitness
            space.agents[-1].position = copy.deepcopy(agent.position)
            space.agents[-1].fit = copy.deepcopy(agent.fit)
