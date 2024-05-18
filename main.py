from src.Data.base_data import BaseData
from utils.result_saver import ResultSaver
from src.PCA.main import PCA
from src.LDA.main import LDA

class AlzheimerIA:
    # Este es la clase principal, el cual se hara el llamado a cada uno de los modelos respectivos
    def __init__(self):
        self.safe_files = False # Cambiar a True en caso de querer Generar nuevamente los archivos npy
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.display = BaseData()
        self.images = dict()
        self.__prepare_images()
        self.result_saver = ResultSaver(self.images)
        self.extracted_data = None
        self.__generate_files_npy_and_extract_feature()
        self.begin()

    def begin(self):
        self.X_train, self.X_test, self.y_train, self.y_test = PCA(self.extracted_data['non_ALZ'], self.extracted_data['ALZ']).begin()
        LDA(self.X_train, self.X_test, self.y_train, self.y_test).begin()

    def __prepare_images(self):
        self.images = self.display.load_and_display_images()  # Obtengo las imagenes originales categorizadas en un diccionario

    def __generate_files_npy_and_extract_feature(self):
        self.extracted_data = self.result_saver.process_features(self.safe_files)  # Guardo las variables en archivos .npy y retorno las categorias trabajadas.
        print('De la data que traemos tenemos: ', self.extracted_data.keys())

# Ahora preparamos las imagenes







if __name__ == "__main__":
    AlzheimerIA()



