import tensorflow as tf
from nalp.corpus.text import TextCorpus
from nalp.datasets.language_modeling import LanguageModelingDataset
from nalp.encoders.integer import IntegerEncoder
from nalp.models.generators.rnn import RNNGenerator
from opytimizer import Opytimizer
from opytimizer.core.function import Function
from opytimizer.optimizers.swarm.pso import PSO
from opytimizer.spaces.search import SearchSpace

# Creating a character TextCorpus from file
corpus = TextCorpus(from_file='examples/integrations/nalp/chapter1_harry.txt', type='char')

# Creating an IntegerEncoder
encoder = IntegerEncoder()

# Learns the encoding based on the TextCorpus dictionary and reverse dictionary
encoder.learn(corpus.vocab_index, corpus.index_vocab)

# Applies the encoding on new data
encoded_tokens = encoder.encode(corpus.tokens)

# Creating Language Modeling Dataset
dataset = LanguageModelingDataset(encoded_tokens, max_length=10, batch_size=64)


def rnn(opytimizer):
    # Gathering parameters from Opytimizer
    # Pay extremely attention to their order when declaring due to their bounds
    learning_rate = opytimizer[0][0]

    # Creating the RNN
    rnn = RNNGenerator(vocab_size=corpus.vocab_size, embedding_size=256, hidden_size=512)

    # As NALP's RNNs are stateful, we need to build it with a fixed batch size
    rnn.build((64, None))

    # Compiling the RNN
    rnn.compile(optimizer=tf.optimizers.Adam(learning_rate=learning_rate),
                loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=[tf.metrics.SparseCategoricalAccuracy(name='accuracy')])

    # Fitting the RNN
    history = rnn.fit(dataset.batches, epochs=100)

    # Gathering last iteration's accuracy
    acc = history.history['accuracy'][-1]

    return 1 - acc


# Creating Function's object
f = Function(pointer=rnn)

# Number of agents
n_agents = 5

# Number of decision variables
n_variables = 1

# Number of running iterations
n_iterations = 3

# Lower and upper bounds (has to be the same size as n_variables)
lower_bound = [0]
upper_bound = [1]

# Creating the SearchSpace class
s = SearchSpace(n_agents=n_agents, n_iterations=n_iterations,
                n_variables=n_variables, lower_bound=lower_bound,
                upper_bound=upper_bound)

# Hyperparameters for the optimizer
hyperparams = {
    'w': 0.7,
    'c1': 1.7,
    'c2': 1.7
}

# Creating PSO's optimizer
p = PSO(hyperparams=hyperparams)

# Finally, we can create an Opytimizer class
o = Opytimizer(space=s, optimizer=p, function=f)

# Running the optimization task
history = o.start()
