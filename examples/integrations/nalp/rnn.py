import tensorflow as tf
from nalp.corpus.text import TextCorpus
from nalp.datasets.language_modeling import LanguageModelingDataset
from nalp.encoders.integer import IntegerEncoder
from nalp.models.generators.rnn import RNNGenerator

from opytimizer import Opytimizer
from opytimizer.core import Function
from opytimizer.optimizers.swarm import PSO
from opytimizer.spaces import SearchSpace

# Creates a character TextCorpus from file
corpus = TextCorpus(from_file='examples/integrations/nalp/chapter1_harry.txt', corpus_type='char')

# Creates an IntegerEncoder
encoder = IntegerEncoder()

# Learns the encoding based on the TextCorpus dictionary and reverse dictionary
encoder.learn(corpus.vocab_index, corpus.index_vocab)

# Applies the encoding on new data
encoded_tokens = encoder.encode(corpus.tokens)

# Creates Language Modeling Dataset
dataset = LanguageModelingDataset(encoded_tokens, max_length=10, batch_size=64)


def rnn(opytimizer):
    # Gathers parameters from Opytimizer
    # Pay extremely attention to their order when declaring due to their bounds
    learning_rate = opytimizer[0][0]

    # Creates the RNN
    rnn = RNNGenerator(vocab_size=corpus.vocab_size, embedding_size=256, hidden_size=512)

    # As NALP's RNNs are stateful, we need to build it with a fixed batch size
    rnn.build((64, None))

    # Compiling the RNN
    rnn.compile(optimizer=tf.optimizers.Adam(learning_rate=learning_rate),
                loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=[tf.metrics.SparseCategoricalAccuracy(name='accuracy')])

    # Fitting the RNN
    history = rnn.fit(dataset.batches, epochs=100)

    # Gathers last iteration's accuracy
    acc = history.history['accuracy'][-1]

    return 1 - acc


# Number of agents and decision variables
n_agents = 5
n_variables = 1

# Lower and upper bounds (has to be the same size as `n_variables`)
lower_bound = [0]
upper_bound = [1]

# Creates the space, optimizer and function
space = SearchSpace(n_agents, n_variables, lower_bound, upper_bound)
optimizer = PSO()
function = Function(rnn)

# Bundles every piece into Opytimizer class
opt = Opytimizer(space, optimizer, function)

# Runs the optimization task
opt.start(n_iterations=3)
