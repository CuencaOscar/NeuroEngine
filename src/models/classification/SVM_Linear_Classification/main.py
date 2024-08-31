import os
import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from PIL import Image
from sklearn import svm
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
import joblib

DIRECTORIES = {
    'non': r'C:\Users\ocuen\Developer\Repos\NeuroEngine\Dataset\Non_Demented',
    'very_mild': r'C:\Users\ocuen\Developer\Repos\NeuroEngine\Dataset\Very_Mild_Demented',
    'mild': r'C:\Users\ocuen\Developer\Repos\NeuroEngine\Dataset\Mild_Demented',
    'moderate': r'C:\Users\ocuen\Developer\Repos\NeuroEngine\Dataset\Moderate_Demented'
}

class SVMLinearClassification:
    def __init__(self, directories=DIRECTORIES):
        self.class_names = ['Non Demented', 'Very Mild Demented', 'Mild Demented', 'Moderate Demented']
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.directories = directories
        self.resized_image_array = []
        self.resized_image_array_label = []
        self.width = 256
        self.height = 256
        self.new_size = (self.width, self.height)
        self.clf = svm.SVC(kernel='linear', probability=True)
        self.train_x = None
        self.train_y = None
        self.test_x = None
        self.test_y = None
        self.directory_path = r'C:\Users\ocuen\Developer\Repos\NeuroEngine\assets\Models\classification\SVM_Linear_Classification'
        self.directory = self.create_directory(self.directory_path)
        self.n_classes = len(np.unique(self.train_y))

    def begin(self):
        self.resizer()
        self.split_data()
        self.load_model()
        self.train_model()
        self.evaluate_model()
        self.save_model()
        self.create_confusion_matrix()
        self.create_roc_curve()
        self.save_performance()

    def load_model(self):
        """Carga un modelo entrenado desde un archivo .pkl"""
        self.clf = joblib.load(r'C:\Users\ocuen\Developer\Repos\NeuroEngine\assets\Models\classification\SVM_Linear_Classification\SVM_Linear_model.pkl')
        print("Model loaded from: ", r'C:\Users\ocuen\Developer\Repos\NeuroEngine\assets\Models\classification\SVM_Linear_Classification\SVM_Linear_model.pkl')

    def create_directory(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        return path

    def resizer(self):
        for label, directory in enumerate(self.directories.values()):
            for file in os.listdir(directory):
                img = Image.open(os.path.join(directory, file))
                img = img.resize(self.new_size)
                array_temp = np.array(img)
                img_wide = array_temp.reshape(1, self.width * self.height)
                self.resized_image_array.append(img_wide[0])
                self.resized_image_array_label.append(label)

    def split_data(self):
        self.train_x, self.test_x, self.train_y, self.test_y = train_test_split(
            self.resized_image_array, self.resized_image_array_label, test_size=0.2)

    def train_model(self):
        self.clf.fit(self.train_x, self.train_y)

    def evaluate_model(self):
        y_pred_train = self.clf.predict(self.train_x)
        y_pred_test = self.clf.predict(self.test_x)
        accuracy_train = accuracy_score(self.train_y, y_pred_train)
        accuracy_test = accuracy_score(self.test_y, y_pred_test)
        self.accuracy_train = accuracy_train
        self.accuracy_test = accuracy_test

    def save_performance(self):
        with open(os.path.join(self.directory_path, 'model_performance.txt'), 'w') as f:
            f.write(f'Training Accuracy: {self.accuracy_train}\n')
            f.write(f'Testing Accuracy: {self.accuracy_test}\n')

    def save_model(self):
        model_path = os.path.join(self.directory_path, 'SVM_Linear_model.pkl')
        joblib.dump(self.clf, model_path)
        print(f"Model saved at {model_path}")

    def create_confusion_matrix(self):
        y_pred = self.clf.predict(self.test_x)
        cm = confusion_matrix(self.test_y, y_pred)
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=self.class_names, yticklabels=self.class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.savefig(os.path.join(self.directory_path, 'confusion_matrix.jpg'))
        plt.close()

    def create_roc_curve(self):
        print('Analizando ROC')
        y_bin = label_binarize(self.test_y, classes=[0, 1, 2, 3])
        n_classes = y_bin.shape[1]
        clf = OneVsRestClassifier(self.clf)
        y_score = clf.fit(self.train_x, self.train_y).decision_function(self.test_x)
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in tqdm(range(n_classes), desc="Calculando ROC por clase"):
            fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        plt.figure(figsize=(10, 8))
        colors = ['blue', 'red', 'green', 'purple']
        for i, color in zip(range(n_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=2,
                    label=f'ROC curve of class {self.class_names[i]} (area = {roc_auc[i]:.2f})')
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Multi-class ROC Curve')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(self.directory_path, 'roc_curve.jpg'))
        plt.close()

