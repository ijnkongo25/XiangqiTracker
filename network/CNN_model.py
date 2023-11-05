import torch
import torch.nn as nn
from mypath import Path

class CNNModel(nn.Module):
    def __init__(self, num_classes=14, pretrained=None):
        super(CNNModel, self).__init__()

        self.input_dim =  3 * 50 * 50  # Linearized input dimensions
        
        # Define convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        # Define fully connected layers
        self.fc1 = nn.Linear(4608, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, num_classes)

        # Weight initialization
        self.__init_weight()

        # If a pretrained path is provided, load the weights
        if pretrained:
            self.__load_pretrained_weights(pretrained)

    def forward(self, x):
        x = x.to(self.conv1.bias.dtype)
        x = nn.ReLU()(self.conv1(x))
        x = nn.MaxPool2d(2)(x)
        x = nn.ReLU()(self.conv2(x))
        x = nn.MaxPool2d(2)(x)
        x = nn.ReLU()(self.conv3(x))
        x = nn.MaxPool2d(2)(x)

        # Flatten the output
        x = x.reshape(-1, 4608)
        x = nn.ReLU()(self.fc1(x))
        x = nn.ReLU()(self.fc2(x))
        x = nn.ReLU()(self.fc3(x))
        x = self.fc4(x)
        return x
    

    def __init_weight(self):
        """Private method for weight initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def __load_pretrained_weights(self, path):
        """Private method to load pretrained weights into the model."""
        state_dict = torch.load(path)
        self.load_state_dict(state_dict)

    
def get_1x_lr_params(model):
    """
    This generator returns all the parameters for conv and two fc layers of the net.
    """
    b = [model.fc1, model.fc2, model.fc3]
    for i in range(len(b)):
        for k in b[i].parameters():
            if k.requires_grad:
                yield k

def get_10x_lr_params(model):
    """
    This generator returns all the parameters for the last fc layer of the net.
    """
    b = [model.fc4]
    for j in range(len(b)):
        for k in b[j].parameters():
            if k.requires_grad:
                yield k

if __name__ == "__main__":
    inputs = torch.randn(1, 3, 50, 50)
    print('size is', inputs.shape)
    net = CNNModel(num_classes=14, pretrained=False)
    outputs = net.forward(inputs)
    print(outputs.size())