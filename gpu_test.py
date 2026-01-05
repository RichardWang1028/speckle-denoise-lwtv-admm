import torch, time
assert torch.cuda.is_available(), "CUDA not available"

x = torch.randn(4096, 4096, device="cuda")
torch.cuda.synchronize()

t = time.time()
for _ in range(80):
    x = x @ x

torch.cuda.synchronize()
print("seconds", time.time() - t)
