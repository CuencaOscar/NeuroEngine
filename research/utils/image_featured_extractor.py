import os
import numpy as np
from glob import glob
from PIL import Image
import matplotlib.image as mpimg
from skimage.transform import resize


class ImageFeatureExtractor:
    def __init__(self):
        pass

    def extract_feature(self, image_path):  # PCA
        img = mpimg.imread(image_path)
        img = img / 255.0
        img = resize(img, (128, 128, 3), anti_aliasing=True)
        img = np.reshape(img, (128, 384))
        return img

    def resizer(self, image_directory, label, new_size, resized_image_array, resized_image_array_label):  # SVM
        for file_path in glob(os.path.join(image_directory, "*")):
            with Image.open(file_path) as img:
                img = img.resize(new_size, Image.LANCZOS)
                img_array = np.array(img).flatten()
                resized_image_array.append(img_array)
                resized_image_array_label.append(label)
        return resized_image_array, resized_image_array_label
