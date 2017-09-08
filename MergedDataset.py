import torch.utils.data as data
import numpy as np

class TestDataset1(data.Dataset):
    def __init__(self):
        pass
    def __getitem__(self, index):
        return 'TEST 1'
    def __len__(self):
        return 3

class TestDataset2(data.Dataset):
    def __init__(self):
        pass
    def __getitem__(self, index):
        return 'TEST 2'
    def __len__(self):
        return 3

class MergedDataset(data.Dataset):
    def __init__(self, datasets, prob):
        # prob should be a probability distribution eg. (0.3, 0.5, 0.2)
        self.datasets = datasets
        self.progress = np.zeros(len(datasets)).astype(int)
        self.indicies = list(range(len(datasets)))

        self.prob = list(prob)
        mul_const = (1./sum(self.prob))
        self.prob = [i * mul_const for i in self.prob]

        self.length = sum(len(i) for i in datasets)
        self.counter = 0
    def __getitem__(self, index):
        i = np.random.choice(self.indicies, p=self.prob)
        ret_val = self.datasets[i][self.progress[i]]
        self.progress[i] += 1
        self.counter += 1

        if (not self.counter == self.length) and self.progress[i] >= len(self.datasets[i]):
            print self.prob, i
            print self.indicies, i
            self.prob.pop(i)
            self.indicies.pop(i)
            mul_const = (1./sum(self.prob))
            self.prob = [i * mul_const for i in self.prob]

        return ret_val
    def __len__(self):
        return self.length

if __name__ == '__main__':
    for j in range(100):
        a = MergedDataset([TestDataset1(), TestDataset2()], [0.5, 0.5])
        for i in range(len(a)):
            print a[i]
