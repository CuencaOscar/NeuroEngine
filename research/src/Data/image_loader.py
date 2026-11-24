import os
from glob import glob


class ImageLoader:
    def __init__(self, base_path):
        self.base_path = base_path

    def get_images(self, category):
        path_pattern = os.path.join(self.base_path, category, "*")
        image_files = glob(path_pattern)
        return image_files
