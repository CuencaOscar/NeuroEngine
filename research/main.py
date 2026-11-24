from src.Data.base_data import BaseData
from src.Data.split_data import SplitData
from utils.result_saver import ResultSaver
from src.models.detection.PCA.main import PCA
from src.models.detection.LDA.main import LDA
from src.models.detection.SVM_Linear.main import SVMLinear
from src.models.detection.SVM_Poly.main import SVMPoly
from src.models.detection.SVM_Rbf.main import SVMRbf
from src.models.detection.CNNVGG16.main import CNNVGG16
from src.models.classification.SVM_Linear_Classification.main import SVMLinearClassification


class AlzheimerIA:
    # Este es la clase principal, el cual se hara el llamado a cada uno de los modelos respectivos
    def __init__(self):
        self.safe_files = False  # Cambiar a True en caso de querer Generar nuevamente los archivos npy
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.display = BaseData()
        self.images = dict()
        self.result_saver = None
        self.extracted_data = None
        self.begin()

    def begin(self):
        self.__prepare_images()
        self.__generate_files_npy_and_extract_feature()
        self.__train_test_data()
        ## Seleccionar los modelos a trabajar, hasta ese punto ya tenemos el split de los datos para los entrenamientos
        # PCA(self.extracted_data['non_ALZ'],
        #     self.extracted_data['ALZ'],
        #     self.X_train,
        #     self.X_test,
        #     self.y_train,
        #     self.y_test).begin()
        # LDA(self.X_train,
        #     self.X_test,
        #     self.y_train,
        #     self.y_test).begin()
        # SVMLinear(self.images).begin()
        # SVMPoly(self.images).begin()
        # SVMRbf(self.images).begin()
        # CNNVGG16(self.X_train,
        #     self.X_test,
        #     self.y_train,
        #     self.y_test).begin()
        SVMLinearClassification().begin()

    def __prepare_images(self):
        print("Entra aqui ?")
        self.images = (
            self.display.load_and_display_images()
        )  # Obtengo las imagenes originales categorizadas en un diccionario

    def __generate_files_npy_and_extract_feature(self):
        print("Veamos...")
        self.result_saver = ResultSaver(self.images)
        self.extracted_data = self.result_saver.process_features(
            self.safe_files
        )  # Guardo las variables en archivos .npy y retorno las categorias trabajadas.
        print("De la data que traemos tenemos: ", self.extracted_data.keys())

    def __train_test_data(self):
        self.X_train, self.X_test, self.y_train, self.y_test = SplitData(
            self.extracted_data["non_ALZ"], self.extracted_data["ALZ"]
        ).train_data_test()


# Ahora preparamos las imagenes


if __name__ == "__main__":
    AlzheimerIA()
