# Copyright (c) 2019-2026 Opytimizer contributors.
# Licensed under the Apache License, Version 2.0.

"""Grid-Search.

References:
    J. Bergstra and Y. Bengio. Random search for hyper-parameter optimization.
    Journal of machine learning research (2012).

"""

from typing import Any

from opytimizer.core import Optimizer


class GS(Optimizer):
    """Evaluate a fixed grid with the base optimizer lifecycle.

    """

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """Initialize a grid search optimizer.

        Args:
            params: Base optimizer overrides without grid-search-specific parameters.

        """

        super(GS, self).__init__()

        self.build(params)
