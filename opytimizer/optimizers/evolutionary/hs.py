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

        super(HS, self).__init__()

        self.HMCR = 0.7
        self.PAR = 0.7
        self.bw = 1.0

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

        a = copy.deepcopy(agents[0])

        for j, (lb, ub) in enumerate(zip(a.lb, a.ub)):
            r1 = r.generate_uniform_random_number()
            if r1 <= self.HMCR:
                k = r.generate_integer_random_number(0, len(agents))
                a.position[j] = agents[k].position[j]

                r2 = r.generate_uniform_random_number()
                if r2 <= self.PAR:
                    r3 = r.generate_uniform_random_number(-1, 1)
                    a.position[j] += r3 * self.bw
            else:
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

        logger.info("Overriding class: HS -> IHS.")

        self.PAR_min = 0
        self.PAR_max = 1

        self.bw_min = 1
        self.bw_max = 10

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

        logger.info("Overriding class: IHS -> GHS.")

        super(GHS, self).__init__(params)

        logger.info("Class overrided.")

    def _generate_new_harmony(self, agents: List[Agent]) -> Agent:
        """It generates a new harmony.

        Args:
            agents: List of agents.

        Returns:
            (Agent): A new agent (harmony) based on music generation process.

        """

        a = copy.deepcopy(agents[0])

        for j, (lb, ub) in enumerate(zip(a.lb, a.ub)):
            r1 = r.generate_uniform_random_number()
            if r1 <= self.HMCR:
                k = r.generate_integer_random_number(0, len(agents))
                a.position[j] = agents[k].position[j]

                r2 = r.generate_uniform_random_number()
                if r2 <= self.PAR:
                    z = r.generate_integer_random_number(0, a.n_variables)
                    a.position[j] = agents[0].position[z]
            else:
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

        self.LP = 100

        self.HMCRm = 0.98
        self.PARm = 0.9

        self.bw_min = 1
        self.bw_max = 10

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
            r1 = r.generate_uniform_random_number()
            if r1 <= self.HMCR:
                r2 = r.generate_uniform_random_number(-1, 1)
                a.position[j] += r2 * self.bw

                r3 = r.generate_uniform_random_number()
                if r3 <= self.PAR:
                    a.position[j] = agents[0].position[j]
            else:
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

        self.HMCR = r.generate_gaussian_random_number(self.HMCRm, 0.01)[0]
        self.PAR = r.generate_gaussian_random_number(self.PARm, 0.05)[0]

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

        logger.info("Overriding class: HS -> NGHS.")

        self.pm = 0.1

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

        a = copy.deepcopy(best)

        for j, (lb, ub) in enumerate(zip(a.lb, a.ub)):
            new_position = 2 * (best.position[j] - worst.position[j])
            new_position = np.clip(new_position, lb, ub)

            r1 = r.generate_uniform_random_number()

            a.position[j] = worst.position[j] + r1 * (new_position - worst.position[j])

            r2 = r.generate_uniform_random_number()
            if r2 <= self.pm:
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

        logger.info("Overriding class: NGHS -> GOGHS.")

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

        a = copy.deepcopy(agents[0])

        A = np.zeros((a.n_variables))
        B = np.zeros((a.n_variables))

        k = r.generate_uniform_random_number()

        for j in range(a.n_variables):
            A[j], B[j] = c.FLOAT_MAX, -c.FLOAT_MAX

            for agent in agents:
                if A[j] > agent.position[j]:
                    A[j] = agent.position[j]
                elif B[j] < agent.position[j]:
                    B[j] = agent.position[j]

            a.position[j] = k * (A[j] + B[j]) - new_agent.position[j]

        return a

    def update(self, space: Space, function: Function) -> None:
        """Wraps Generalized Opposition Global-Best Harmony Search over all agents and variables.

        Args:
            space: Space containing agents and update-related information.
            function: A Function object that will be used as the objective function.

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
