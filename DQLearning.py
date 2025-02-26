import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torchvision.transforms as T
import matplotlib
import matplotlib.pyplot as plt
import random
import torch.nn.functional as F
import numpy as np
from PIL import Image
import math
from itertools import count
import copy
import matplotlib.animation as animation

import gym

env = gym.make('CartPole-v0', render_mode='rgb_array').unwrapped
if 'inline' in matplotlib.get_backend():
    from IPython import display
# Kommentar oder entferne:
# plt.ion()

FloatTensor = torch.FloatTensor
LongTensor = torch.LongTensor
ByteTensor = torch.ByteTensor
Tensor = FloatTensor

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

class Memory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.pos = 0

    def push(self, state, action, next_state, reward):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.pos] = (state, action, next_state, reward)
        self.pos = (self.pos + 1) % self.capacity       

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)   

    def __len__(self):
        return len(self.memory)

class Netz(nn.Module):
    def __init__(self):
        super(Netz, self).__init__()
        # Neue BatchNorm-Schicht für den Input (3 Kanäle)
        self.input_bn = nn.BatchNorm2d(3)
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.norm1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.norm2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.norm3 = nn.BatchNorm2d(32)
        self.fc = nn.Linear(448, 2)

    def forward(self, x):
        # Normalisiere den Input zuerst
        x = self.input_bn(x)
        x = F.relu(self.norm1(self.conv1(x)))
        x = F.relu(self.norm2(self.conv2(x)))
        x = F.relu(self.norm3(self.conv3(x)))
        return self.fc(x.view(x.size(0), -1))

resize = T.Compose([T.ToPILImage(),
                    T.Resize(40, interpolation=Image.BICUBIC),
                    T.ToTensor()])
width = 600
def cart_pos():
    env_width = env.x_threshold * 2
    return int(env.state[0] * width / env_width + width / 2.0)

def get_image():
    screen = env.render().transpose((2, 0, 1))  # => CHW
    screen = screen[:, 160:320]
    view = 320
    cart = cart_pos()
    if cart < view // 2:
        sliced = slice(view)
    elif cart > width - view // 2:
        sliced = slice(-1*view, None)
    else:
        sliced = slice(cart - view // 2, cart + view // 2)
    screen = screen[:, :, sliced]
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)
    # Entferne .type(Tensor) und verschiebe stattdessen auf das Device
    # return resize(screen).unsqueeze(0).to(device)
    return resize(screen).unsqueeze(0).type(Tensor).to(device)
env.reset()
# plt.figure()
# plt.imshow(get_image().cpu().squeeze(0).permute(1, 2, 0).numpy(),
#            interpolation='none')
# plt.title('Example extracted screen')
# plt.show()  # Blockiert, bis das Fenster geschlossen wird

model = Netz().to(device)
target_model = copy.deepcopy(model).to(device)  # Zielnetzwerk initialisieren
# optimizer = optim.Adam(model.parameters(), lr=5e-5)  # Adam statt RMSprop verwenden
optimizer = optim.RMSprop(model.parameters(), lr=1e-4, alpha=0.99, eps=1e-8, momentum=0)
mem = Memory(16384)
done = 0
eps_end = 0.05
eps_start = 0.95
eps_steps = 250
batch_size = 128
gamma = 0.99

# Weitere Hyperparameter:
target_update_frequency = 1  # Aktualisiere das Zielnetzwerk alle 10 Episoden
eps_decay = 1000  # Veränderung der Epsilon-Abklingrate
# Passen ggf. eps_start/eps_end an
eps_start = 0.95
eps_end = 0.1

# Polyak Averaging: weiche Aktualisierung des Zielnetzwerks
def update_target(model, target_model, tau=0.01):
    for target_param, param in zip(target_model.parameters(), model.parameters()):
        target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

def get_action(state):
    global done
    epsilon = random.random()
    threshold = eps_end + (eps_start - eps_end) * math.exp(-1. * done / eps_decay)
    done += 1
    if epsilon > threshold:
        with torch.no_grad():
            q_values = model(state)  # state ist bereits auf device
            return q_values.max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randint(0, 1)]], device=device, dtype=torch.long)
    
def train():
    if len(mem) < batch_size:
        return
    x = mem.sample(batch_size)  # ([s, a, n, r])
    batch = tuple(zip(*x))  # ((s1,s2,...), (a1,a2,...), (n1,n2,...), (r1,r2,...))
    
    # Boolescher Tensor für nicht-leere next_states auf dem richtigen Device
    non_final = torch.tensor([s is not None for s in batch[2]], dtype=torch.bool, device=device)
    non_final_next = torch.cat([s for s in batch[2] if s is not None]).to(device)
    state = torch.cat(batch[0]).to(device)
    action = torch.cat(batch[1]).to(device)
    reward = torch.cat(batch[3]).to(device)
    
    # Optional: Reward clipping
    reward = torch.clamp(reward, -1, 1)
    
    action_value = model(state).gather(1, action)
    
    # Stelle sicher, dass next_value auf dem richtigen Device liegt
    next_value = torch.zeros(batch_size, device=device, dtype=torch.float)
    with torch.no_grad():
        # Verwende hier das Ziel-Netzwerk für stabilere Q-Ziele!
        next_value[non_final] = target_model(non_final_next).max(1)[0]
    
    # Forme target_action_value so um, dass die Form [batch_size, 1] entsteht.
    target_action_value = (next_value * gamma + reward).unsqueeze(1)
    
    loss = F.smooth_l1_loss(action_value, target_action_value)
    optimizer.zero_grad()
    loss.backward()
    for p in model.parameters():
        p.grad.data.clamp_(-1, 1)
    optimizer.step()

# Live-Plot für die Umgebung einrichten:
plt.ion()  # Interaktiver Modus einschalten
fig_env = plt.figure("Live Environment")
# Zeige initial einen Frame der Umgebung
im_env = plt.imshow(get_image().cpu().squeeze(0).permute(1, 2, 0).numpy(), interpolation='none')
plt.title("Live Environment Render")

# Trainingsschleife:
made_it = []       # (optional) Liste für Episodenlängen
num_episodes = 1000
frames = []        # Gespeicherte Frames für die spätere Animation

for i in range(num_episodes):
    env.reset()
    last = get_image()
    current = get_image()
    state = current - last
    done = 0  # Schrittzähler pro Episode
    for j in count():
        action = get_action(state)
        _, reward, done_flag, truncated, _ = env.step(action[0, 0].item())
        lost = done_flag or truncated
        reward = Tensor([reward])
        last = current
        current = get_image()
        
        # Aktualisiere live den angezeigten Frame:
        frame = current.cpu().squeeze(0).permute(1, 2, 0).numpy()
        im_env.set_array(frame)
        plt.pause(0.001)
        
        # Speichere den Frame für die spätere Animation:
        frames.append(frame)
        
        if lost:
            break
        
        next_state = current - last
        mem.push(state, action, next_state, reward)
        state = next_state
        train()
        
    update_target(model, target_model, tau=0.01)
    print(f'Episode {i}: Schritte = {j}')
    made_it.append(j)

env.close()
plt.ioff()  # Interaktivmodus ausschalten
plt.show()   # Zeige den finalen Zustand des Live-Fensters an

# Animation der gesammelten Bilder erstellen:
fig_anim = plt.figure("Animation")
im_anim = plt.imshow(frames[0], interpolation='none')

def update(frame):
    im_anim.set_array(frame)
    return [im_anim]

ani = animation.FuncAnimation(fig_anim, update, frames=frames, interval=50, blit=True)
plt.title('Animation: Bewegung des Sticks (Pole) während des Trainings')
plt.show()

# Modell speichern
torch.save(model, 'dqlearning-model.pth')