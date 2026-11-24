from paths import DATASET_DIR_NAME, RESULTS_ROOT, ROOT  # noqa: F401
from src.Data.image_loader import ImageLoader


class CategorizedImageLoader:
    def __init__(self):
        self.base_path = str(ROOT.joinpath(DATASET_DIR_NAME))
        self.categories = ["Very_Mild_Demented", "Mild_Demented", "Moderate_Demented", "Non_Demented"]
        self.categorized_images = {}
        self.loader = ImageLoader(self.base_path)

    def load_images(self):
        for category in self.categories:
            image_files = self.loader.get_images(category)
            self.categorized_images[category] = image_files
            print(f"Found {len(image_files)} images in category {category}")

    def get_all_images(self):
        return self.categorized_images
