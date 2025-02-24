import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import os

kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}
train_data = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=64, shuffle=True, **kwargs)
test_data = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])),
    batch_size=64, shuffle=True, **kwargs)

has_gpu = torch.cuda.is_available()
has_mps = torch.backends.mps.is_built()
device = "mps" if has_mps else "cpu"
print("MPS (Apple Metal) is", "AVAILABLE" if has_mps else "NOT AVAILABLE")
print("CUDA (NVIDIA GPU) is", "AVAILABLE" if has_gpu else "NOT AVAILABLE")

class Netz(nn.Module):
    def __init__(self):
        super(Netz, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_dropout = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 60)
        self.fc2 = nn.Linear(60, 10)
    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x,2)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.conv2_dropout(x)
        x = F.max_pool2d(x,2)
        x = F.relu(x)
        # print(x.size()) # 64, 20, 4, 4 20*16 = 320
        # exit()
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)  # Specify the dimension for log_softmax
    
''''''
if os.path.isfile("mnist.pt"):
    model = torch.load("mnist.pt")
else : 
    model = Netz() 
''''''
model = Netz()  
if has_gpu:
    model.cuda()
if has_mps: 
    model.to("mps")   

optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
def train(epoch):
    model.train()
    for batch_id, (data, target) in enumerate(train_data):
        if has_gpu:
            data, target = data.cuda(), target.cuda()
        if has_mps:
            data, target = data.to("mps"), target.to("mps")
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        criterion = F.nll_loss
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        print('Epoche: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_id * len(data), len(train_data.dataset),
                100. * batch_id / len(train_data), loss.item()))  # Use item() to get the value

def test():
    model.eval()
    loss  = 0
    correct = 0
    for data, target in test_data:
        if has_gpu:
            data, target = data.cuda(), target.cuda()
        if has_mps:
            data, target = data.to("mps"), target.to("mps")
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        loss += F.nll_loss(output, target, size_average=False).item()
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    loss /= len(test_data.dataset)
    print('\nDurchschnittsloss: {:.4f}, Genauigkeit: {}/{} ({:.0f}%)\n'.format(
        loss, correct, len(test_data.dataset),
        100. * correct / len(test_data.dataset)))

for epoch in range(1, 30): # or 30
    train(epoch)
    test()

torch.save(model, "mnist.pt")