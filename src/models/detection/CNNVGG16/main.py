import numpy as np
import os
import cv2
from keras.applications import VGG16
from keras.models import Sequential
from keras.layers import Flatten, Dense
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc

from utils.directory_manager import DirectoryManager

class CNNVGG16:
    def __init__(self, x_train, x_test, y_train, y_test):
        self.directory_path = r'C:\Users\ocuen\Developer\Repos\NeuroEngine\assets\Models\detection\CNNVGG16'
        self.directory = DirectoryManager(self.directory_path)
        self.X_train = x_train
        self.X_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.model = None
        self.fpr = None
        self.tpr = None
        self.roc_auc = None
        self.cm = None
        self.rgb_arr_to_3d_train = None
        self.rgb_arr_to_3d_test = None

    def begin(self):
        self.directory.create_directory()
        self.__convert_data_to_rgb_and_reshape()
        self.__create_and_fit_model()
        self.__train_model()
        self.__evaluate_model()
        self.__create_confusion_matrix()
        self.__create_roc_curve()
        self.__save_summary()
        self.__save_model()
        self.__save_confusion_matrix()
        self.__save_roc_figure()

    def __convert_data_to_rgb_and_reshape(self):
        self.rgb_arr_to_3d_train = self.__convert_to_rgb_and_reshape(self.X_train)
        self.rgb_arr_to_3d_test = self.__convert_to_rgb_and_reshape(self.X_test)
        self.y_train = np.array(self.y_train).astype(np.int32)  # Asegurarse de que y_train sea int32
        self.y_test = np.array(self.y_test).astype(np.int32)    # Asegurarse de que y_test sea int32


    def __convert_to_rgb_and_reshape(self, data):
        rgb_list = [self.__to_rgb(img) for img in data]
        rgb_arr = np.stack(rgb_list, axis=0)
        return rgb_arr

    def __to_rgb(self, img):
        img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)
        img_rgb = np.dstack((img, img, img))
        return img_rgb

    def __create_and_fit_model(self):
        vgg16 = VGG16(weights='imagenet', input_shape=(128, 128, 3), include_top=False)
        vgg16.trainable = False
        self.model = Sequential()
        self.model.add(vgg16)
        self.model.add(Flatten(name='flatten'))
        self.model.add(Dense(2, activation='softmax', name='predictions'))
        self.model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    def __train_model(self):
        # Asegurarse de que los datos son del tipo correcto antes de entrenar
        if isinstance(self.rgb_arr_to_3d_train, np.ndarray) and isinstance(self.y_train, np.ndarray):
            self.model.fit(self.rgb_arr_to_3d_train, self.y_train, batch_size=7, epochs=10, verbose=1)
            self.model.fit(self.rgb_arr_to_3d_test, self.y_test, batch_size=7, epochs=10, verbose=1)
        else:
            raise TypeError("Training data and labels must be numpy arrays.")

    def __evaluate_model(self):
        y_pred = self.model.predict(self.rgb_arr_to_3d_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        accuracy = accuracy_score(self.y_test, y_pred_classes)
        precision = precision_score(self.y_test, y_pred_classes)
        recall = recall_score(self.y_test, y_pred_classes)
        f1 = f1_score(self.y_test, y_pred_classes)
        cm = confusion_matrix(self.y_test, y_pred_classes)
        info_path = os.path.join(self.directory_path, 'model_performance.txt')
        with open(info_path, 'w') as file:
            file.write(f'Accuracy: {accuracy}\n')
            file.write(f'Precision: {precision}\n')
            file.write(f'Recall: {recall}\n')
            file.write(f'F1 Score: {f1}\n')
            file.write(f'Confusion Matrix: \n{cm}\n')

    def __create_confusion_matrix(self):
        y_pred = self.model.predict(self.rgb_arr_to_3d_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        self.cm = confusion_matrix(self.y_test, y_pred_classes)

    def __create_roc_curve(self):
        y_pred = self.model.predict(self.rgb_arr_to_3d_test)
        y_scores = y_pred[:, 1]
        self.fpr, self.tpr, _ = roc_curve(self.y_test, y_scores)
        self.roc_auc = auc(self.fpr, self.tpr)
    
    def __save_summary(self):
        model_path = os.path.join(self.directory_path, 'CNNVGG16_summary.txt')
        if not os.path.exists(self.directory_path):
            os.makedirs(self.directory_path)
        with open(model_path, 'w') as f:
            self.model.summary(print_fn=lambda x: f.write(x + '\n'))
        print(f"Summary saved at {model_path}")

    def __save_model(self):
        model_path = os.path.join(self.directory_path, 'CNNVGG16_model.h5')
        self.model.save(model_path)
        print(f"Model saved at {model_path}")

    def __save_confusion_matrix(self):
        plt.figure()
        sns.heatmap(self.cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Labels')
        plt.xlabel('Predicted Labels')
        matrix_path = os.path.join(self.directory_path, 'CNNVGG16_confusion_matrix.jpg')
        plt.savefig(matrix_path)
        plt.close()

    def __save_roc_figure(self):
        plt.figure()
        plt.plot(self.fpr, self.tpr, color='darkorange', lw=2, label=f'ROC curve (area = {self.roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        roc_path = os.path.join(self.directory_path, 'CNNVGG16_roc_curve.jpg')
        plt.savefig(roc_path)
        plt.close()
