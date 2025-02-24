import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim

has_gpu = torch.cuda.is_available()
has_mps = torch.backends.mps.is_built()
device = "mps" if has_mps else "cpu"
print("MPS (Apple Metal) is", "AVAILABLE" if has_mps else "NOT AVAILABLE")
print("CUDA (NVIDIA GPU) is", "AVAILABLE" if has_gpu else "NOT AVAILABLE")

class MeinNetz(nn.Module):
    def __init__(self):
        super(MeinNetz, self).__init__()
        self.lin1 = nn.Linear(10, 10)
        self.lin2 = nn.Linear(10, 10)

    def forward(self, x):
        x = F.relu(self.lin1(x))
        x = self.lin2(x)
        return x
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num = 1
        for i in size:
            num *= i
        return num 

if os.path.isfile("meinNetz.pt"):
    netz = torch.load("meinNetz.pt")
else : 
    netz = MeinNetz()
if has_mps :
    netz = netz.to("mps") # mps apple metal
print(netz)
for i in range(100) :
    x = [1,0,0,0,1,0,0,0,1,1]
    input = Variable(torch.Tensor([x for _ in range(10)]))
    if has_mps :
        input = input.to("mps") # mps apple metal
    out=netz(input)
    # print(out)
    x = [0,1,1,1,0,1,1,1,0,0]
    target = Variable(torch.Tensor([x for _ in range(10)]))
    if has_mps :
        target = target.to("mps") # mps apple metal
    criterion = nn.MSELoss()
    loss =  criterion(out, target)
    print(loss)
    # print(loss.grad_fn.next_functions[0][0])
    netz.zero_grad()
    loss.backward()
    optimzer = optim.SGD(netz.parameters(), lr=0.10)
    # print(netz.lin1.bias.grad)
    optimzer.step()

torch.save(netz, "meinNetz.pt")