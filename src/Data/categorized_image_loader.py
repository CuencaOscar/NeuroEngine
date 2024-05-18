# categorized_image_loader.py
from src.Data.image_loader import ImageLoader


# Este codigo lo que hace es obtener las ritas edlas imagenes de toda la base de datos

class CategorizedImageLoader:
    def __init__(self):
        self.base_path = r'C:\Users\ocuen\Developer\Repos\NeuroNet-Engine\Dataset'
        self.categories = ['Very_Mild_Demented',
                           'Mild_Demented',
                           'Moderate_Demented',
                           'Non_Demented']
        self.categorized_images = {}
        self.loader = ImageLoader(self.base_path)

    def load_images(self):
        for category in self.categories:
            image_files = self.loader.get_images(category)
            self.categorized_images[category] = image_files
            print(f'Found {len(image_files)} images in category {category}')

    def get_all_images(self):
        return self.categorized_images
