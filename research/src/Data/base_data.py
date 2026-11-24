from src.Data.categorized_image_loader import CategorizedImageLoader


class BaseData:
    def __init__(self):
        self.image_loader = CategorizedImageLoader()

    def load_and_display_images(self):
        self.image_loader.load_images()
        all_images = self.image_loader.get_all_images()
        for category, images in all_images.items():
            print(f"{category}: {len(images)} images")
        return all_images
