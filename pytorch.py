import torch
x = torch.Tensor([[1,2,3,4,5],[9,8,7,6,5]])
y = torch.Tensor([[1,2,3,4,5],[9,8,7,6,5]])

has_gpu = torch.cuda.is_available()
has_mps = torch.backends.mps.is_built()
device = "mps" if has_mps else "cuda" if has_gpu else "cpu"
print("MPS (Apple Metal) is", "AVAILABLE" if has_mps else "NOT AVAILABLE")
print("CUDA (NVIDIA GPU) is", "AVAILABLE" if has_gpu else "NOT AVAILABLE")

if has_mps:
    x = x.to("mps")
    y = y.to("mps")
elif has_gpu:
    x = x.cuda()
    y = y.cuda()

print(x + y)
'''
x = torch.randn(2,3)
print(x)
y = torch.randn(2,3)
print(y)
print (torch.add(x,y))
'''