def pairwise(it):
    it = iter(it)
    while True:
        yield next(it), next(it)