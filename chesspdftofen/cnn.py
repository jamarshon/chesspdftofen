import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    import importlib.resources as pkg_resources
except ImportError:
    # Try backported to PY<37 `importlib_resources`.
    import importlib_resources as pkg_resources

from . import data

class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    self.pool = nn.MaxPool2d(2, 2)
    
    self.conv1 = nn.Conv2d(1, 6, 5)
    # self.batch1 = nn.BatchNorm2d(6)

    self.conv2 = nn.Conv2d(6, 16, 5)
    # self.batch2 = nn.BatchNorm2d(16)
    
    self.drop1 = nn.Dropout(0.5)
    self.fc1 = nn.Linear(16 * 13 * 13, 120)
    
    self.drop2 = nn.Dropout(0.4)
    self.fc2 = nn.Linear(120, 84)
    self.fc3 = nn.Linear(84, 14)

  def forward(self, x):
    # conv
    # x = self.pool(self.batch1(F.relu(self.conv1(x))))
    # x = self.pool(self.batch2(F.relu(self.conv2(x))))
    x = self.pool(F.relu(self.conv1(x)))
    x = self.pool(F.relu(self.conv2(x)))
    
    x = x.view(-1, 16 * 13 * 13)
    
    # fc
    x = self.drop1(x)
    x = F.relu(self.fc1(x))
    x = self.drop2(x)
    x = F.relu(self.fc2(x))
    x = self.fc3(x)
    return x

def get_model():
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  print('Model inference device', device)
  net = Net()

  path = 'model_exp_cbnn1_last7800.pth'
  # path = 'model_exp_cbnn1_last_bn7800.pth'
  with pkg_resources.path(data, path) as f:
    net.load_state_dict(torch.load(f))

  print('Model loaded')
  return net.eval()