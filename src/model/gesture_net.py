import torch
import torch.nn as nn
import torch.nn.functional as F


class GestureNet(nn.Module):
    def __init__(self, in_features=63, num_classes=5):
        super(GestureNet, self).__init__()
        self.fc1 = nn.Linear(in_features, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 64)
        self.classifier = nn.Linear(64, num_classes)

        dropout_p = 0.2
        self.dropout1 = nn.Dropout(p=dropout_p)
        self.dropout2 = nn.Dropout(p=dropout_p)
        self.dropout3 = nn.Dropout(p=dropout_p)
        self.dropout4 = nn.Dropout(p=dropout_p)
        self.dropout5 = nn.Dropout(p=dropout_p)

    def forward(self, x):
        x = self.dropout1(F.relu(self.fc1(x)))
        x = self.dropout2(F.relu(self.fc2(x)))
        x = self.dropout3(F.relu(self.fc3(x)))
        x = self.dropout4(F.relu(self.fc4(x)))
        x = self.dropout5(F.relu(self.fc5(x)))
        logits = F.softmax(self.classifier(x), dim=1)
        return logits
