import os

class DirectoryManager:
    def __init__(self, directory_path=''):
        self.directory_path = directory_path

    def create_directory(self):
        if not os.path.exists(self.directory_path):
            os.makedirs(self.directory_path)
            print(f"Directorio creado: {self.directory_path}")
        else:
            print(f"El directorio ya existe: {self.directory_path}")

    def recreate_files(self, file_paths: list):
        for file_path in file_paths:
            if os.path.exists(file_path):
                os.remove(file_path)
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            open(file_path, 'a').close()
            print(f"Archivo creado: {file_path}")
