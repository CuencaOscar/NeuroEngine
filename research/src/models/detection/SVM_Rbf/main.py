from paths import DATASET_DIR_NAME, RESULTS_ROOT, ROOT  # noqa: F401
import os
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import svm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split
from utils.directory_manager import DirectoryManager
from utils.image_featured_extractor import ImageFeatureExtractor

FILE_PATH = [str(ROOT.joinpath(DATASET_DIR_NAME, 'Mild_Demented')),
             str(ROOT.joinpath(DATASET_DIR_NAME, 'Moderate_Demented')),
             str(ROOT.joinpath(DATASET_DIR_NAME, 'Non_Demented')),
             str(ROOT.joinpath(DATASET_DIR_NAME, 'Very_Mild_Demented'))]

class SVMRbf:
    def __init__(self, images):
        self.directory_path_info = str(RESULTS_ROOT.joinpath('Models', 'detection', 'SVM_Rbf', 'info.txt'))
        self.directory_path = str(RESULTS_ROOT.joinpath('Models', 'detection', 'SVM_Rbf'))
        self.directory = DirectoryManager(self.directory_path_info)
        self.datasets_directory = FILE_PATH
        self.resized_image_array = list()
        self.resized_image_array_label = list()
        self.resized_image_array_np = None
        self.resized_image_array_label_np = None
        self.clf = None
        self.train_x = None
        self.test_x = None
        self.train_y = None
        self.test_y = None
        self.y_pred_test = None
        self.fpr = None
        self.tpr = None
        self.roc_auc = None
        self.width = 256
        self.height = 256
        self.new_size = (self.width,
                         self.height)
        self.resizer = ImageFeatureExtractor()
        self.images_categorized = images

    def begin(self):
        print('Inicia el proceso')
        print('Las cosas que recbimos son: ', self.images_categorized.keys())
        self.__create_directory()
        self.__process_image_without_alz()
        self.__process_image_with_alz()
        self.__converter_list_to_np_array()
        self.__split_train_test_data()
        self.__verify_size_array()
        self.__create_and_fit_model()
        self.__evaluate_model_train()
        self.__evaluate_model_test()
        self.__save_model()
        self.__create_confusion_matrix()
        self.__save_confusion_matrix()
        self.__prepare_roc_curve()
        self.__save_roc_figure()

    def __create_directory(self):
        print('Creamos el directorio')
        self.directory.create_directory()

    def __process_image_without_alz(self):
        print('Procesamos las imagenes de las carpetas que no contienen Alzheimer')
        self.resized_image_array, self.resized_image_array_label = self.resizer.resizer(self.datasets_directory[-2],
                                                                                        0,
                                                                                        self.new_size,
                                                                                        self.resized_image_array,
                                                                                        self.resized_image_array_label)

    def __process_image_with_alz(self):
        print('Procesamos las imagenes de las carpetas que contienen Alzheimer')
        for path in [self.datasets_directory[-1], self.datasets_directory[0], self.datasets_directory[1]]:
            self.resized_image_array, self.resized_image_array_label = self.resizer.resizer(path,
                                                                                            1,
                                                                                            self.new_size,
                                                                                            self.resized_image_array,
                                                                                            self.resized_image_array_label)

    def __converter_list_to_np_array(self):
        self.resized_image_array_np = np.array(self.resized_image_array)
        self.resized_image_array_label_np = np.array(self.resized_image_array_label)

    def __split_train_test_data(self):
        self.train_x, self.test_x, self.train_y, self.test_y = train_test_split(
            self.resized_image_array_np,
            self.resized_image_array_label_np,
            test_size=0.2,
            stratify=self.resized_image_array_label_np
        )

    def __verify_size_array(self):
        print('Guardar lso size de el resize y cantidad de imagenes')
        with open(self.directory_path_info, 'w') as file:
            file.write(f"Total processed images: {len(self.resized_image_array)}\n")
            file.write(f"Label distribution: {np.unique(self.resized_image_array_label, return_counts=True)}\n")
            file.write(f"Number of images in 'non': {len(self.images_categorized['Non_Demented'])}\n")
            file.write(f"Number of images in 'ALZ' (very_mild + mild + moderate): "
                    f"{len(self.images_categorized['Very_Mild_Demented']) + len(self.images_categorized['Mild_Demented']) + len(self.images_categorized['Moderate_Demented'])}\n")
            file.write(f"Total processed images: {len(self.resized_image_array)}\n")
            file.write(f"Example data from one image (flattened array): {self.resized_image_array[1][:10]}\n")
            file.write(f"Training set size: {len(self.train_x)}\n")
            file.write(f"Test set size: {len(self.test_x)}\n")


    def __create_and_fit_model(self):
        self.clf = svm.SVC(kernel='rbf', probability=True)
        self.clf.fit(self.train_x, self.train_y)

    def __evaluate_model_train(self):
        y_pred_train = self.clf.predict(self.train_x)
        self.__evaluate_model(y_pred_train, self.train_y, "Training")

    def __evaluate_model_test(self):
        self.y_pred_test = self.clf.predict(self.test_x)
        self.__evaluate_model(self.y_pred_test, self.test_y, "Testing")

    def __evaluate_model(self, predictions, truths, dataset_type="Training"):
        with open(self.directory_path_info, 'a') as file:
            file.write(f'--- {dataset_type} Data ---\n')
            file.write(f'Accuracy : { accuracy_score(truths, predictions)}\n')
            file.write(f'Precision: { precision_score(truths, predictions)}\n')
            file.write(f'Recall   : { recall_score(truths, predictions)}\n')
            file.write(f'F1 Score : { f1_score(truths, predictions)}\n')
            file.write(f'Confusion Matrix: { confusion_matrix(truths, predictions)}\n')


    def __save_model(self):
        print('Guardamos el modelo')
        model_path = os.path.join(self.directory_path, 'SVM_rbf.pkl')
        joblib.dump(self.clf, model_path)
        print(f"Modelo SVM Rbf guardado en: {model_path}")

    def __create_confusion_matrix(self):
        self.cm_test = confusion_matrix(self.test_y, self.y_pred_test)

    def __save_confusion_matrix(self):
        plt.figure(figsize=(8, 6))
        sns.heatmap(self.cm_test, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Labels')
        plt.xlabel('Predicted Labels')
        path_matrix = os.path.join(self.directory_path, 'SVM_rbf_matrix.jpg')
        plt.savefig(path_matrix, format='jpg')
        plt.close()
        print(f"Matriz de confusión guardada como imagen en '{self.directory_path}'.")
        
    def __prepare_roc_curve(self):
        y_scores = self.clf.predict_proba(self.test_x)[:, 1]
        self.fpr, self.tpr, thresholds = roc_curve(self.test_y, y_scores)
        self.roc_auc = auc(self.fpr, self.tpr) 
 
    def __save_roc_figure(self):
        plt.figure(figsize=(8, 6))
        plt.plot(self.fpr, self.tpr, color='darkorange', lw=2, label=f'ROC curve (area = {self.roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC)')
        plt.legend(loc="lower right")
        roc_image_path = os.path.join(self.directory_path, 'SVM_rbf_roc.jpg')
        plt.savefig(roc_image_path, format='jpg')