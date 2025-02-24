import torch
import torchvision
from torchvision import transforms
import torch.optim as optim
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import os
from PIL import Image
from os import listdir
import random


has_gpu = torch.cuda.is_available()
has_mps = torch.backends.mps.is_built()
device = "mps" if has_mps else "cpu"
print("MPS (Apple Metal) is", "AVAILABLE" if has_mps else "NOT AVAILABLE")
print("CUDA (NVIDIA GPU) is", "AVAILABLE" if has_gpu else "NOT AVAILABLE")
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(256),
    transforms.ToTensor(),  
    normalize
])
#target:  [isCat, isDog] [0,1]
train_data_list = []
target_list = []
train_data = []
files = listdir('catdog/train/')
for i in range(len(listdir('catdog/train/'))) :
    f = random.choice(files)
    files.remove(f)
    img = Image.open('catdog/train/'+f)
    img_tensor = transform(img)
    train_data_list.append(img_tensor)
    isCat = 1 if 'cat' in f else 0
    isDog = 1 if 'dog' in f else 0
    target = isCat if isCat else isDog  # Use a single integer for the target
    target_list.append(target)
    if len(train_data_list) >= 64:
        train_data.append((torch.stack(train_data_list), torch.tensor(target_list)))
        train_data_list = []
        target_list = []
        # if len(train_data) > 150: # for testing purposes
        #     break # remove this line to use the entire dataset

class Netz(nn.Module):
    def __init__(self):
        super(Netz, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 12, kernel_size=5)
        self.conv3 = nn.Conv2d(12, 18, kernel_size=5)
        self.fc1 = nn.Linear(18 * 28 * 28, 1000)  # Angepasst: tatsächliche Eingabegröße ist 18x28x28
        self.fc2 = nn.Linear(1000, 2)
    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        x = x.view(x.size(0), -1)  # Richtig flatten: behält die Batchgröße bei
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x  # Roh-Logits, kein Sigmoid!
        
model = Netz()
if has_gpu:
    model.cuda()
if has_mps:
    model.to("mps")

optimizer = optim.Adam(model.parameters(), lr=0.01)

def train(epoch):
    model.train()
    batch_id = 0
    
    for data, target in train_data:
        if has_gpu:
            data, target = data.cuda(), torch.Tensor(target).cuda()
        if has_mps:
            data, target = data.to("mps"), torch.Tensor(target).to("mps")
        data, target = Variable(data), Variable(target.long())
        optimizer.zero_grad()
        out = model(data)
        criterion = F.nll_loss
        loss = criterion(out, target)
        loss.backward()
        optimizer.step()
        
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6e}'.format(
            epoch, batch_id * len(data), len(train_data) * len(data),
            100. * batch_id / len(train_data), loss.item()))
        batch_id += 1
def test():
    model.eval()
    files = listdir('catdog/test/')
    f = random.choice(files)
    img = Image.open('catdog/test/' + f)
    img_eval_tensor = transform(img)  # statt transforms(img)
    img_eval_tensor = img_eval_tensor.unsqueeze(0)
    if has_gpu:
       img_eval_tensor = img_eval_tensor.cuda()
    if has_mps:
       img_eval_tensor = img_eval_tensor.to("mps")
    data = Variable(img_eval_tensor)
    out = model(data)
    pred = out.data.max(1, keepdim=True)[1].item()  # Extrahiere Vorhersage als integer
    print("Prediction: ", "Cat" if pred == 0 else "Dog")
    # print("Actual: ", "Cat" if 'cat' in f else "Dog")
    # note image is not labeled, so we can't check for correctness
    # correct = (('cat' in f and pred == 0) or ('dog' in f and pred == 1))
    # print("Correct!" if correct else "Incorrect!")
    print("Testing on", f)
    img.show()
    x = input("Press Enter to continue testing...")
    print("")
for epoch in range(1, 30):
    train(epoch)
test()


torch.save(model, "catsanddogs.pt")