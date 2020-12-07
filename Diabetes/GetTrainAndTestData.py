from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from pandas import read_csv
from sklearn.preprocessing import LabelEncoder

class CSVDataset(Dataset):
    # load the dataset
    def __init__(self, path):
        # load the csv file as a dataframe
        df = read_csv(path, header=None)
        # store the inputs and outputs
        self.X = df.values[1:, :-1]
        self.y = df.values[1:, -1]
        # ensure input data is floats
        self.X = self.X.astype('float32')

        # label encode target and ensure the values are floats
        self.y = LabelEncoder().fit_transform(self.y)
        self.y = self.y.astype('float32')
        self.y = self.y.reshape((len(self.y), 1))

    # number of rows in the dataset
    def __len__(self):
        return len(self.X)

    # get a row at an index
    def __getitem__(self, idx):
        return [self.X[idx], self.y[idx]]
    def get__inputs(self, idx):
        return self.X[idx]
    # get indexes for train and test rows
    def get_splits(self, n_test=0.3):
        # determine sizes
        test_size = round(n_test * len(self.X))
        train_size = len(self.X) - test_size
        # calculate the split
        return random_split(self, [train_size, test_size])

# prepare the dataset
def prepare_data(path):
    # load the dataset
    dataset = CSVDataset(path)
    # calculate split
    train, test = dataset.get_splits()
    # prepare data loaders
    train_dl = DataLoader(train, batch_size=1, shuffle=True)
    test_dl = DataLoader(test, batch_size=1, shuffle=False)
    return train_dl, test_dl