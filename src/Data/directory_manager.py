import os

class DirectoryManager:
    def __init__(self, base_path):
        self.base_path = base_path

    def create_directory(self):
        """Check if a directory exists, and create it if it doesn't."""
        if not os.path.exists(self.base_path):
            os.makedirs(self.base_path)
            print(f"Carpeta '{self.base_path}' creada.")
        else:
            print(f"La carpeta '{self.base_path}' ya existe.")