import torch

class CudaChecker:
    def __init__(self):
        self.cuda_available = torch.cuda.is_available()

    def check_cuda_availability(self):
        print("CUDA disponible:", self.cuda_available)
        return self.cuda_available

    def print_gpu_details(self):
        if self.cuda_available:
            print("Cantidad de GPUs disponibles:", torch.cuda.device_count())
            print("Nombre de la GPU:", torch.cuda.get_device_name(0))