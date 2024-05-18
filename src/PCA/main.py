import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn import decomposition
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from utils.directory_manager import DirectoryManager

class PCA:
    def __init__(self, non_alz, alz):
        self.directory_path = r'C:\Users\ocuen\Developer\Repos\NeuroNet-Engine\assets\Models\detection\PCA'
        self.directory = DirectoryManager(self.directory_path)
        self.non_alz = non_alz
        self.alz = alz
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.pca_model = None
        self.k = None
        self.var_cumu = None

    def begin(self):
        print('Inicia el proceso')
        self.__create_directory()
        self.__prepare_model()
        self.__find_optimal_pca_components()
        self.__save_model()
        self.__save_figure()
        return self.X_train, self.X_test, self.y_train, self.y_test

    def __create_directory(self):
        print('Creamos el directorio')
        self.directory.create_directory()

    def __prepare_model(self):
        print('Preparamos el modelo')
        all_data = np.concatenate((np.array(self.non_alz), np.array(self.alz)))
        all_data = all_data.reshape(all_data.shape[0], np.product(all_data.shape[1:]))
        scaler = StandardScaler()
        X = scaler.fit_transform(all_data)
        y = [0] * len(self.non_alz) + [1] * len(self.alz)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2)

    def __find_optimal_pca_components(self):
        print('Encontramos los componentes')
        self.pca_model = decomposition.PCA(600)
        self.pca_model.fit_transform(self.X_train)
        self.var_cumu = np.cumsum(self.pca_model.explained_variance_ratio_) * 100
        self.k = np.argmax(self.var_cumu > 80)
        print('Veamos si me da el valor correcto: ', self.k)

    def __save_model(self):
        print('Guardamos el modelo')
        model_path = os.path.join(self.directory_path, 'PCA.pkl')
        joblib.dump(self.pca_model, model_path)
        print(f"Modelo PCA guardado en: {model_path}")

    def __save_figure(self):
        print('Guardamos la figura')
        plt.figure(figsize=[10,5])
        plt.title('Cumulative Explained Variance by Principal Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.xlabel('Principal Components (n={})'.format(self.k))
        plt.axvline(x=self.k, color="k", linestyle="--")
        plt.axhline(y=80, color="r", linestyle="--")
        plt.plot(self.var_cumu)
        variance_graph_path = os.path.join(self.directory_path, 'PCA.jpg')
        plt.savefig(variance_graph_path)
        print(f"Explained variance graph saved at: {variance_graph_path}")
