import torch.nn as nn
import torch.nn.functional as F


class NetAgMaxOverTime(nn.Module): #acc %, loss: , kernel_conv1=80, learn_rate=0.0001
    def __init__(self, dropout_p=0.4):
        super(NetAgMaxOverTime, self).__init__()

        # Define parameters
        self.dropout_p = dropout_p
        # self.max_length = 140  # L
        self.training = True

        self.conv1 = nn.Conv3d(1, 50, (20, 131, 2), stride=(1, 1, 1))
        self.pool1d = nn.MaxPool1d(3, 3, dilation=2)
        self.fc1 = nn.Linear(50*25, 512)
        self.fc2 = nn.Linear(512, 100)
        self.fc3 = nn.Linear(100, 4)


    #         self.dropout = nn.Dropout(dropout_p)

    def forward(self, x):
        x = self.conv1(x)
        x = x.squeeze(2)
        x_after_conv1 = x.squeeze(2)  # (batch, channel, words) (batch, 50, 78)
        x = F.relu(self.pool1d(x_after_conv1))
        # print(x.shape)
        x = x.view(-1, 50*25)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, training=self.training)
        x = self.fc3(x)
        return x, x_after_conv1


class NetDBPediaMaxOverTime(nn.Module): #acc %, loss: , kernel_conv1=80, learn_rate=0.0001
    def __init__(self, dropout_p=0.4):
        super(NetDBPediaMaxOverTime, self).__init__()

        # Define parameters
        self.dropout_p = dropout_p
        # self.max_length = 140  # L
        self.training = True

        self.conv1 = nn.Conv3d(1, 50, (20, 131, 3), stride=(1, 1, 1))
        self.pool1d = nn.MaxPool1d(3, 3, dilation=2)
        self.fc1 = nn.Linear(50*25, 512)
        self.fc2 = nn.Linear(512, 100)
        self.fc3 = nn.Linear(100, 14)


    #         self.dropout = nn.Dropout(dropout_p)

    def forward(self, x):
        x = self.conv1(x)
        x = x.squeeze(2)
        x = x.squeeze(2)  # (batch, kernel, words)
        x = F.relu(self.pool1d(x))
        # print(x.shape)
        x = x.view(-1, 50*25)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, training=self.training)
        x = self.fc3(x)
        return x


class NetYelpFullMaxOverTime(nn.Module): #acc %, loss: , kernel_conv1=80, learn_rate=0.0001
    def __init__(self, dropout_p=0.4):
        super(NetYelpFullMaxOverTime, self).__init__()

        # Define parameters
        self.dropout_p = dropout_p
        # self.max_length = 140  # L
        self.training = True

        self.conv1 = nn.Conv3d(1, 50, (20, 131, 3), stride=(1, 1, 1))
        self.pool1d = nn.MaxPool1d(3, 3, dilation=2)
        self.fc1 = nn.Linear(50*25, 512)
        self.fc2 = nn.Linear(512, 100)
        self.fc3 = nn.Linear(100, 5)


    #         self.dropout = nn.Dropout(dropout_p)

    def forward(self, x):
        x = self.conv1(x)
        x = x.squeeze(2)
        x = x.squeeze(2)  # (batch, kernel, words)
        x = F.relu(self.pool1d(x))
        # print(x.shape)
        x = x.view(-1, 50*25)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, training=self.training)
        x = self.fc3(x)
        return x


class NetAmazonMaxOverTime(nn.Module): #acc %, loss: , kernel_conv1=80, learn_rate=0.0001
    def __init__(self, dropout_p=0.4):
        super(NetAmazonMaxOverTime, self).__init__()

        # Define parameters
        self.dropout_p = dropout_p
        # self.max_length = 140  # L
        self.training = True

        self.conv1 = nn.Conv3d(1, 50, (20, 131, 3), stride=(1, 1, 1))
        self.pool1d = nn.MaxPool1d(3, 3, dilation=2)
        self.fc1 = nn.Linear(50*25, 512)
        self.fc2 = nn.Linear(512, 100)
        self.fc3 = nn.Linear(100, 1)
        self.sigmoid = nn.Sigmoid()


    #         self.dropout = nn.Dropout(dropout_p)

    def forward(self, x):
        x = self.conv1(x)
        x = x.squeeze(2)
        x = x.squeeze(2)  # (batch, kernel, words)
        x = F.relu(self.pool1d(x))
        # print(x.shape)
        x = x.view(-1, 50*25)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, training=self.training)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x