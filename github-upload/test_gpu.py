import torch

print("PyTorch:", torch.__version__)
print("GPU disponibil:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("Placa video:", torch.cuda.get_device_name(0))
    x = torch.randn(1000, 1000, device='cuda')
    y= x @ x
    print("Calcul pe GPU reusit:", y.shape)
    