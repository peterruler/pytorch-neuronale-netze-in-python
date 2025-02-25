from os import listdir
import unicodedata
import string
import torch
import torch.nn as nn
from torch.autograd import Variable
import random
import matplotlib.pyplot as plt  # Korrigierter Import

letters = string.ascii_letters + ".,:'"
def toAscii(s):
    return ''.join(
        char for char in unicodedata.normalize('NFD', s)
        if unicodedata.category(char) != 'Mn'
        and char in letters
    )

def lines(datei):
    f = open(datei, encoding='utf-8').read().strip().split('\n')
    return [toAscii(l) for l in f]

def charToIndex(char):
    return letters.find(char)

def charToTensor(char):
    tensor = torch.zeros(1, len(letters))
    tensor[0][charToIndex(char)] = 1
    return tensor

def nameToTensor(line):
    tensor = torch.zeros(len(line), 1, len(letters))
    for li, char in enumerate(line):
        tensor[li][0][charToIndex(char)] = 1
    return tensor
langs = []
data = {}
for f in listdir('data/names'):
    lang = f.split('.')[0]
    ls = lines('data/names/' + f)
    langs.append(lang)
    data[lang] = ls

# print(nameToTensor(data["German"][0]).size()) # torch.Size([6, 1, 56])
# print(data["German"])

class Netz(nn.Module):
    def __init__(self, input, hiddens, output):
        super(Netz, self).__init__()
        self.hiddens = hiddens
        self.hid = nn.Linear(input + hiddens, hiddens)
        self.out = nn.Linear(input + hiddens, output)
        self.logsoftmax = nn.LogSoftmax(dim=1)
    
    def forward(self, x, hidden):
        x = torch.cat((x, hidden), 1)
        new_hidden = self.hid(x)
        out = self.logsoftmax(self.out(x))
        return out, new_hidden
    
    def initHidden(self):
        return Variable(torch.zeros(1, self.hiddens))
    
model = Netz(len(letters), 128, len(data))

def langFromOutput(out):
    _, i = out.data.topk(1)
    return langs[i[0][0]]

def getTrainData():
    lang = random.choice(langs)
    name = random.choice(data[lang])
    name_tensor = Variable(nameToTensor(name))
    lang_tensor = Variable(torch.LongTensor([langs.index(lang)]))
    return lang, name, lang_tensor, name_tensor

critereon = nn.NLLLoss()
learning_rate = 0.005
def train(lang_tensor, name_tensor):
    hidden = model.initHidden()
    model.zero_grad()
    for i in range(name_tensor.size()[0]):
        output, hidden = model(name_tensor[i], hidden)
    loss = critereon(output, lang_tensor)
    loss.backward()
    for i in model.parameters():
        i.data.add_(-learning_rate, i.grad.data)
    return output, loss

avg = []
sum = 0
for i in range(1,100000):
    lang, name, lang_tensor, name_tensor = getTrainData()
    output, loss = train(lang_tensor, name_tensor) 
    sum = sum + loss.item() 
    if i % 1000 == 0:
        print(i/1000, "% done.")
        avg.append(sum / 1000)
        sum = 0
plt.figure()
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss over time")
plt.plot(avg)
plt.show()