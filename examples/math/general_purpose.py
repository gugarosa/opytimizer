import opytimizer.math.general as g

# Creating a list for pairwising
individuals = [1, 2, 3, 4]

# Creating pairwise from list
for pair in g.n_wise(individuals, 2):
    # Outputting pairs
    print(f'Pair: {pair}')

# Performing a tournmanet selection over list
selected = g.tournament_selection(individuals, 2)

# Outputting selected individuals
print(f'Selected: {selected}')
