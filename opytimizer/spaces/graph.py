"""Graph-based search space.
"""


from opytimizer.utils import logging

logger = logging.get_logger(__name__)


class GraphSpace:
    """ """

    def __init__(self, n_blocks: int) -> None:
        """Initialization method.

        Args:
            n_blocks: Number of blocks.

        """

        logger.info("Creating class: GraphSpace.")

        self.n_blocks = n_blocks

        self.build()

        logger.debug("Blocks: %d | Built: %s.", self.n_blocks, self.built)
        logger.info("Class created.")
