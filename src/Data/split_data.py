import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class SplitData:
    def __init__(self, non_alz, alz):
        self.non_alz = non_alz
        self.alz = alz
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def  train_data_test(self):
        all_data = np.concatenate((np.array(self.non_alz), np.array(self.alz)))
        all_data = all_data.reshape(all_data.shape[0], np.product(all_data.shape[1:]))
        scaler = StandardScaler()
        X = scaler.fit_transform(all_data)
        y = [0] * len(self.non_alz) + [1] * len(self.alz)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2)
        return self.X_train, self.X_test, self.y_train, self.y_test
