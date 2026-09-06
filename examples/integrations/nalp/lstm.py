# Copyright (c) 2019-2026 Opytimizer contributors.
# Licensed under the Apache License, Version 2.0.

import numpy as np
import tensorflow as tf
from nalp.corpus import TextCorpus
from nalp.datasets import LanguageModelingDataset
from nalp.encoders import IntegerEncoder
from nalp.models.generators import LSTMGenerator

from opytimizer import Opytimizer
from opytimizer.optimizers.swarm import PSO
from opytimizer.spaces import SearchSpace

corpus = TextCorpus(from_file="examples/integrations/nalp/chapter1_harry.txt", corpus_type="char")

encoder = IntegerEncoder()
encoder.learn(corpus.vocab_index, corpus.index_vocab)
encoded_tokens = encoder.encode(corpus.tokens)

dataset = LanguageModelingDataset(encoded_tokens, max_contiguous_pad_length=10, batch_size=64)


def lstm(opytimizer: np.ndarray) -> float:
    """Train a fresh character-level LSTM using the shared encoded corpus.

    Args:
        opytimizer: One-row position array containing the Adam learning rate.

    Returns:
        One minus the final training accuracy after one hundred epochs.

    """

    learning_rate = opytimizer[0][0]

    lstm = LSTMGenerator(vocab_size=corpus.vocab_size, embedding_size=256, hidden_size=512)

    # As NALP's LSTMs are stateful, we need to build it with a fixed batch size
    lstm.build((64, None))

    lstm.compile(
        optimizer=tf.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )

    history = lstm.fit(dataset.batches, epochs=100)

    acc = history.history["accuracy"][-1]

    return 1 - acc


n_agents = 5
n_variables = 1

lower_bound = [0]
upper_bound = [1]

space = SearchSpace(n_agents, n_variables, lower_bound, upper_bound)
optimizer = PSO()

opt = Opytimizer(space, optimizer, lstm)

opt.start(n_iterations=3)
