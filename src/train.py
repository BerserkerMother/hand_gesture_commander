import torch
import torch.nn.functional as F
from torch.utils import data

from model import GestureNet
from data import HandGesture
from utils import AverageMeter


def main():
    dataset = HandGesture("data.csv")

    data_length = len(dataset)
    train_length = int(data_length * 0.8)  # uses 80% for training
    val_length = data_length - train_length
    train_set, val_set = data.random_split(
        dataset, lengths=[train_length, val_length])

    train_loader = data.DataLoader(
        dataset=train_set,
        shuffle=True,
        batch_size=16
    )

    val_loader = data.DataLoader(
        dataset=val_set,
        shuffle=True,
        batch_size=16
    )

    model = GestureNet().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    for e in range(1000):
        train_acc = train(model, optimizer, train_loader, e)
        val_acc = val(model, val_loader, e)
        print(train_acc)
        print(val_acc)

    torch.save(model, "model.pth")


def train(model, optimizer, data_loader, e):
    model.train()

    train_acc = AverageMeter()
    print("EPOCH:%d _______________________________________" % e)
    for i, (features, target) in enumerate(data_loader):
        features = features.cuda()
        target = target.cuda()

        batch_size = features.size()[0]

        logits = model(features)
        loss = F.cross_entropy(logits, target)

        prediction = logits.max(1)[1]
        num_correct = (prediction == target).sum()
        train_acc.update(num_correct.item(), batch_size)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 4 == 0:
            print(loss.item())

        return train_acc.avg() * 100


def val(model, data_loader, e):
    model.eval()

    val_acc = AverageMeter()
    for i, (features, target) in enumerate(data_loader):
        features = features.cuda()
        target = target.cuda()

        batch_size = features.size()[0]

        with torch.no_grad():
            logits = model(features)
            loss = F.cross_entropy(logits, target)

        prediction = logits.max(1)[1]
        num_correct = (prediction == target).sum()
        val_acc.update(num_correct.item(), batch_size)

        return val_acc.avg() * 100


def infer(model, features):
    model.eval().cuda()
    features = torch.tensor(features,
                            dtype=torch.float,
                            device=torch.device("cuda")).unsqueeze(0)
    output = model(features)
    prediction = output.max(1)[1].item()

    return prediction


if __name__ == '__main__':
    main()
