from os import listdir
import unicodedata
import string
import torch
import torch.nn as nn
from torch.autograd import Variable
import random
import matplotlib.pyplot as plt  # Korrigierter Import

letters = string.ascii_letters + string.digits + string.punctuation
letters_num = len(letters) + 1

def toAscii(s):
    return ''.join(
        char for char in unicodedata.normalize('NFD', s)
        if unicodedata.category(char) != 'Mn'
        and char in letters
    )

def lines(datei):
    with open(datei, encoding='utf-8', errors='ignore') as f:  # Fehlerhafte Zeichen ignorieren
        return [toAscii(l) for l in f.read().split('\n')]

def charToIndex(char):
    return letters.find(char)

def charToTensor(char):
    tensor = torch.zeros(1, letters_num)
    tensor[0][charToIndex(char)] = 1
    return tensor

def passwordToTensor(line):
    tensor = torch.zeros(len(line), 1, letters_num)
    for li, char in enumerate(line):
        tensor[li][0][charToIndex(char)] = 1
    return tensor

def targetToTensor(password):
    letter_indexes = [letters.find(password[i]) for i in range(1, len(password))]
    letter_indexes.append(letters_num - 1)  # EOS
    return torch.LongTensor(letter_indexes)

lines_file = lines('data/wordlist.txt')
# Leere Zeilen filtern
lines_file = [line for line in lines_file if line.strip()]

class Netz(nn.Module):
    def __init__(self, inputs, hiddens, outputs):
        super(Netz, self).__init__()
        self.hidden_size = hiddens
        self.input_to_output = nn.Linear(inputs + hiddens, outputs)
        self.input_to_hidden = nn.Linear(inputs + hiddens, hiddens)
        self.output_to_output = nn.Linear(hiddens + outputs, outputs)
        self.dropout = nn.Dropout(0.1)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), dim=1)
        hidden = self.input_to_hidden(combined)
        output = self.input_to_output(combined)
        output_combined = torch.cat((hidden, output), dim=1)
        output = self.output_to_output(output_combined)
        output = self.dropout(output)
        output = self.softmax(output)
        return output, hidden
    
    def initHidden(self):
        return Variable(torch.zeros(1, self.hidden_size))

def get_random_example() :
    return random.choice(lines_file)

# print(get_random_example())

def get_random_train(): 
    # Wiederhole, bis ein Passwort gefunden wurde, das länger als 1 Zeichen ist
    # und nach Verarbeitung einen nicht-leeren target_tensor ergibt
    while True:
        pw = get_random_example().strip()
        # Stelle sicher, dass nach toAscii auch mehr als 1 Zeichen übrig bleibt
        pw = toAscii(pw)
        if len(pw) > 1:
            inout_tensor = passwordToTensor(pw)
            target_tensor = targetToTensor(pw)
            if target_tensor.size(0) > 0:
                break
    return Variable(inout_tensor), Variable(target_tensor)

model = Netz(letters_num, 128, letters_num)
criterion = nn.NLLLoss()
learning_rate = 0.0005
def train(input_tensor, target_tensor):
    hidden = model.initHidden()
    model.zero_grad()
    loss = 0
    for i in range(input_tensor.size()[0]):
        output, hidden = model(input_tensor[i], hidden)
        loss += criterion(output, target_tensor[i].unsqueeze(0))  # unsqueeze hinzugefügt
    loss.backward()
    for p in model.parameters():
        p.data.add_(-learning_rate, p.grad.data)
    return output, loss.item() / input_tensor.size()[0]
def sample() :
    input = Variable(passwordToTensor("F"))
    hidden = model.initHidden()
    output = "F"
    for i in range(15):
        out, hidden = model(input[0], hidden)
        _, i = out.data.topk(1)
        i = i[0][0]
        if i == letters_num - 1:
            break
        else:
            letter = letters[i]
            output += letter
            input = Variable(passwordToTensor(letter))
    return output
print("Sample:", sample())


loss_sum = 0
plots = []
for i in range(1, 1000000):
    input_tensor, target_tensor = get_random_train()
    output, loss = train(input_tensor, target_tensor)
    loss_sum += loss
    if i % 100 == 0:
        avg_loss = loss_sum / 100
        print(i / 100, "% made. Loss:", loss, "average Loss:", avg_loss)
        plots.append(avg_loss)
        loss_sum = 0
print("Sample:", sample())
plt.plot(plots)
plt.show()  # Anzeige des Plots