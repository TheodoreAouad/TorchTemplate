import random

class ComposeIterator:

    def __init__(self, iterators, shuffle=False):
        
        self.iterators = iterators
        self.shuffle = shuffle
        self._length = sum([len(it) for it in iterators])



    def __iter__(self):
        self.current_iterators = [
            iter(it) for it in self.iterators
        ]
        return self

    def __next__(self):

        while len(self.current_iterators) != 0:
            if self.shuffle:
                idx = random.choice(range(len(self.current_iterators)))
            else:
                idx = 0
            iterator = self.current_iterators[idx]
            try:
                return next(iterator)
            except StopIteration:
                del self.current_iterators[idx]
        raise StopIteration

    def __len__(self):
        return self._length

    def __getitem__(self, idx):
        return self.iterators[idx]