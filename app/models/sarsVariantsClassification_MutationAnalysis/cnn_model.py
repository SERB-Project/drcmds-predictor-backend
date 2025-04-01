import torch
import torch.nn as nn

class InterSSPPCNN(nn.Module):
    def __init__(self, input_length):
        super(InterSSPPCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=4, out_channels=32, kernel_size=7, padding=3)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(0.2)
        
        self.conv2 = nn.Conv1d(32, 8, kernel_size=4)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout2 = nn.Dropout(0.2)
        
        self.conv3 = nn.Conv1d(8, 8, kernel_size=3)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout3 = nn.Dropout(0.2)
        
        conv_out_size = self.calculate_conv_output_size(input_length)
        self.fc1 = nn.Linear(conv_out_size * 8, 72)
        self.fc2 = nn.Linear(72, 32)
        self.output = nn.Linear(32, 5)

    def calculate_conv_output_size(self, input_length):
        x = torch.zeros(1, 4, input_length)
        x = self.pool1(self.conv1(x))
        x = self.pool2(self.conv2(x))
        x = self.pool3(self.conv3(x))
        return x.size(2)
    # def forward(self, x, mask=None):
    #     # First Convolutional Block
        
    #     x = self.dropout1(self.pool1(torch.relu(self.conv1(x))))
    #     if(mask==None):
    #         mask = (x.sum(dim=1, keepdim=True) != 0).float()
    #     # mask = self.pool1(mask.unsqueeze(1))  # Pool mask
    #     # mask = mask[:, :, :x.size(2)]  # Adjust mask size to match x
    #     mask = self.pool1(mask).squeeze(1).unsqueeze(1)  # Ensure shape remains [batch, 1, length]
    #     # x *= mask
    #     x = x * mask.expand_as(x)  # Ensure proper broadcasting



    #     # Second Convolutional Block
    #     x = self.dropout2(self.pool2(torch.relu(self.conv2(x))))
    #     mask = self.pool2(mask)
    #     mask = mask[:, :, :x.size(2)]  # Adjust mask size to match x
    #     x *= mask

    #     # Third Convolutional Block
    #     x = self.dropout3(self.pool3(torch.relu(self.conv3(x))))
    #     mask = self.pool3(mask)
    #     mask = mask[:, :, :x.size(2)]  # Adjust mask size to match x
    #     x *= mask

    #     # Flatten and Fully Connected Layers
    #     x = x.view(x.size(0), -1)
    #     x = torch.relu(self.fc1(x))
    #     x = torch.relu(self.fc2(x))
    #     x = self.output(x)
    #     return x
    def forward(self, x):
        # Compute mask inside the model
        mask = (x.sum(dim=1, keepdim=True) != 0).float()

        # Apply convolutions
        x = self.dropout1(self.pool1(torch.relu(self.conv1(x))))
        mask = self.pool1(mask)[:, :, :x.size(2)]
        x *= mask

        x = self.dropout2(self.pool2(torch.relu(self.conv2(x))))
        mask = self.pool2(mask)[:, :, :x.size(2)]
        x *= mask

        x = self.dropout3(self.pool3(torch.relu(self.conv3(x))))
        mask = self.pool3(mask)[:, :, :x.size(2)]
        x *= mask

        # Flatten and pass through dense layers
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.output(x)

        return x
