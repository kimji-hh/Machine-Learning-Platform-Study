import torch

def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

if __name__ == "__main__":
    d = pick_device()
    x = torch.randn(1024, 1024, device=d)
    print(f"torch {torch.__version__} | device: {d} | matmul ok: {(x @ x).shape}")
