import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from utils.directory_manager import DirectoryManager
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDAModel

class LDA:
    def __init__(self, x_train, x_test, y_train, y_test):
        self.directory = DirectoryManager(r'C:\Users\ocuen\Developer\Repos\NeuroNet-Engine\assets\Models\detection\LDA\info.txt')
        self.directory_path = r'C:\Users\ocuen\Developer\Repos\NeuroNet-Engine\assets\Models\detection\LDA'
        self.directory_path_info = r'C:\Users\ocuen\Developer\Repos\NeuroNet-Engine\assets\Models\detection\LDA\info.txt'
        self.lda_model = None
        self.X_train_LDA = None
        self.X_test_LDA = None
        self.y_pred_test = None
        self.cm_test = None
        self.fpr = None
        self.tpr = None
        self.roc_auc = None
        self.X_train = x_train
        self.y_train = y_train
        self.X_test = x_test
        self.y_test = y_test

    def begin(self):
        print('Inicia el proceso')
        self.__create_directory()
        self.__lda_configure()
        self.__train_lda_model()
        self.__evaluate_model()
        self.__result_predict()
        self.__save_predictions()
        self.__save_model()
        self.__create_confusion_matrix()
        self.__save_confusion_matrix()
        self.__prepare_roc_curve()
        self.__save_roc_figure()

    def __create_directory(self):
        print('Creamos el directorio')
        self.directory.create_directory()

    def __lda_configure(self):
        print('Configramos LDA')
        self.lda_model = LDAModel(n_components=1)

    def __train_lda_model(self):
        print('Entrenamos con datos de entrenamiento')
        self.X_train_LDA = self.lda_model.fit_transform(self.X_train, self.y_train)
        self.X_test_LDA = self.lda_model.transform(self.X_test)

    def __evaluate_model(self):
        with open(self.directory_path_info, 'w') as file:
            accuracy_train = self.lda_model.score(self.X_train, self.y_train)
            file.write(f'{accuracy_train*100:.2f}% accuracy (training data)\n')
            accuracy = self.lda_model.score(self.X_test, self.y_test)
            file.write(f'{accuracy*100:.2f}% accuracy (testing data)\n')

    def __result_predict(self):
        self.y_pred_test = self.lda_model.predict(self.X_test)

    def __save_predictions(self):
        with open(self.directory_path_info, 'a') as file:
            file.write("Test set predictions:\n")
            for pred in self.y_pred_test:
                file.write(f"{pred}\n")
            print(f"Predictions appended to {self.directory_path_info}")

    def __save_model(self):
        print('Guardamos el modelo')
        model_path = os.path.join(self.directory_path, 'LDA.pkl')
        joblib.dump(self.lda_model, model_path)
        print(f"Modelo LDA guardado en: {model_path}")

    def __create_confusion_matrix(self):
        self.cm_test = confusion_matrix(self.y_test, self.y_pred_test)

    def __save_confusion_matrix(self):
        plt.figure(figsize=(8, 6))
        sns.heatmap(self.cm_test, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Labels')
        plt.xlabel('Predicted Labels')
        path_matrix = os.path.join(self.directory_path, 'LDA_matrix.jpg')
        plt.savefig(path_matrix, format='jpg')
        plt.savefig(self.directory_path)
        plt.close()
        print(f"Matriz de confusión guardada como imagen en '{self.directory_path}'.")
        
    def __prepare_roc_curve(self):
        y_scores = self.lda_model.predict_proba(self.X_test)[:, 1]
        self.fpr, self.tpr, thresholds = roc_curve(self.y_test, y_scores)
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
        roc_image_path = os.path.join(self.directory_path, 'LDA_roc.jpg')
        plt.savefig(roc_image_path, format='jpg')