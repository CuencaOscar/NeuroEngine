import numpy as np
from tqdm import tqdm
from utils.image_featured_extractor import ImageFeatureExtractor
from utils.directory_manager import DirectoryManager

NAMES_IMAGE_WORKS = ['vmild_ALZ', 'mild_ALZ', 'moderate_ALZ', 'non_ALZ']
FILE_PATH = [r'C:\Users\ocuen\Developer\Repos\NeuroNet-Engine\assets\data\v_mild.npy',
             r'C:\Users\ocuen\Developer\Repos\NeuroNet-Engine\assets\data\mild.npy',
             r'C:\Users\ocuen\Developer\Repos\NeuroNet-Engine\assets\data\moderate.npy',
             r'C:\Users\ocuen\Developer\Repos\NeuroNet-Engine\assets\data\non.npy']
FILE_PATH_GENERAL = r'C:\Users\ocuen\Developer\Repos\NeuroNet-Engine\assets\data\alz.npy'

class ResultSaver:
    def __init__(self, categories):
        self.file_paths = FILE_PATH
        self.categories = categories
        self.directory = DirectoryManager

    def process_features(self, save_files):
        print('Estamos procesando las imágenes')
        results = {}
        for index, (category, files) in enumerate(self.categories.items()):
            print(f'Procesando categoría: {category}')
            features = [ImageFeatureExtractor(filename).extract_feature() for filename in tqdm(files)]
            if features:
                features_array = np.array(features)
                results[NAMES_IMAGE_WORKS[index]] = features_array
                self.save_results(self.file_paths[index], features_array) if save_files else None
            else:
                print(f'No se extrajeron características para {category}')
        if all(k in results for k in ['vmild_ALZ', 'mild_ALZ', 'moderate_ALZ']):
            combined_features = np.concatenate([results[k] for k in ['vmild_ALZ', 'mild_ALZ', 'moderate_ALZ']], axis=0)
            results['ALZ'] = combined_features
            self.save_results(FILE_PATH_GENERAL, combined_features) if save_files else None
        self.load_npy_data(FILE_PATH_GENERAL)
        return results

    def save_results(self, file_path, features):
        np.save(file_path, features)

    def load_npy_data(self, file_path):
        try:
            data = np.load(file_path, allow_pickle=True)
            return data
        except Exception as e:
            print(f"Error al cargar el archivo: {e}")