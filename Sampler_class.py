import random


class Sampler():
    def __init__(self, train, batch_size):
        self.train = train
        self.batch_size = batch_size

    def __iter__(self):
        indices = [(i, len((s[0])))
                   for i, s in enumerate(self.train)]
        # indices = [(i, len(self.train[i][0])) for i in range(len(self.train))]
        random.shuffle(indices)
        pooled_indices = []
        # create pool of indices with similar lengths
        for i in range(0, len(indices), self.batch_size * 100):
            pooled_indices.extend(
                sorted(indices[i:i + self.batch_size * 100], key=lambda x: x[1]))

        pooled_indices = [x[0] for x in pooled_indices]

        # yield indices for current batch
        for i in range(0, len(pooled_indices), self.batch_size):
            yield pooled_indices[i:i + self.batch_size]
        return pooled_indices
