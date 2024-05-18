import matplotlib.image as mpimg
import numpy as np
from skimage.transform import resize

class ImageFeatureExtractor:
    def __init__(self, image_path):
        self.image_path = image_path
    
    def extract_feature(self):
        img = mpimg.imread(self.image_path)
        img = img / 255.0
        img = resize(img, (128, 128, 3), anti_aliasing=True)
        img = np.reshape(img, (128, 384))
        return img
