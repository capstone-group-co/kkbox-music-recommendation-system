import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale, Imputer
from sklearn.metrics import confusion_matrix

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data.sampler import SequentialSampler

output_file = 'submission.csv'
train_file = ''

# training MLP
# set random seed
torch.manual_seed(1122)
print(">>> load train and test for pytorch training")
train = np.loadtxt('full_train.csv', delimiter=',', skiprows=1)
test = np.loadtxt('full_test.csv', delimiter=',', skiprows=1)
test_id = test[:, 0]
test = test[:, 1:]

print(">>> generate train, val, test data for mlp")
# create np matrices
x_train, x_val, y_train, y_val = train_test_split(
    train[:, 1:], train[:, 0], test_size=0.001)
x_test, y_test = test, np.zeros((test.shape[0], 1))

# impute missing values
print(">>> impute missing values")
imputer = Imputer()
x_train = imputer.fit_transform(x_train)
x_val = imputer.transform(x_val)
x_test = imputer.transform(x_test)

# rescale training data
print(">>> rescale data")
x_train = scale(x_train, axis=0)
x_val = scale(x_val, axis=0)
x_test = scale(x_test, axis=0)

# create pytorch compatible dataset that has API for automated loaders
trainset = TensorDataset(torch.Tensor(x_train.tolist()).view(
                         x_train.shape[0], -1),
                         torch.Tensor(y_train.tolist()).long())

valset = TensorDataset(torch.Tensor(x_val.tolist()).view(
                       x_val.shape[0], -1),
                       torch.Tensor(y_val.tolist()).long())

testset = TensorDataset(torch.Tensor(x_test.tolist()).view(
                        x_test.shape[0], -1),
                        torch.Tensor(y_test.tolist()).long())

# create pytorch mini-batch loader DataLoader for the dataset
trainloader = DataLoader(trainset, batch_size=2000, shuffle=True)

valloader = DataLoader(valset, batch_size=2000, shuffle=True)

# for test set, we want to maintain the sequence of the data
testsampler = SequentialSampler(testset)
testloader = DataLoader(testset, batch_size=2000, shuffle=False,
                        sampler=testsampler)
print(">>> train, val, test dataset created")


# define and initialize a multilayer-perceptron, a criterion, and an optimizer
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.l1 = nn.Linear(126, 20)
        self.t1 = nn.Tanh()
        self.l2 = nn.Linear(20, 2)
        self.t2 = nn.LogSoftmax()

    def forward(self, x):
        x = self.t1(self.l1(x))
        x = self.t2(self.l2(x))
        return x

print(">>> initiate mlp models")
mlp = MLP()
mlp.cuda()
criterion = nn.NLLLoss()
optimizer = optim.SGD(mlp.parameters(), lr=0.1, momentum=0.9)


# define a training epoch function
def trainEpoch(dataloader, epoch):
    print("Training Epoch %i" % (epoch + 1))
    mlp.train()
    for i, data in enumerate(dataloader, 0):
        inputs, labels = data
        inputs, labels = inputs.cuda(), labels.cuda()
        inputs, labels = Variable(inputs), Variable(labels)
        optimizer.zero_grad()
        outputs = mlp(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()


def validateModel(dataloader, epoch):
    mlp.eval()
    test_loss = 0
    correct = 0
    pred = np.array([])
    targ = np.array([])
    for inputs, targets in dataloader:
        inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs), Variable(targets)
        outputs = mlp(inputs)
        test_loss += F.nll_loss(outputs, targets, size_average=False).data[0]
        pred = np.append(pred, outputs.topk(1)[1].data.view(1, -1).numpy())
        targ = np.append(targ, targets.data.numpy())
        prd = outputs.topk(1)[1].data
        correct += prd.eq(targets.data.view_as(prd)).cpu().sum()
    test_loss /= len(dataloader.dataset)
    test_acc = correct / len(dataloader.dataset)
    cm = confusion_matrix(targ, pred)
    print('[Epoch %i] Accuracy: %.2f, Average Loss: %.2f' %
          (epoch, test_acc, test_loss))
    print(cm)
    return test_loss, test_acc, cm


def testModel(dataloader):
    mlp.eval()
    pred = np.array([])
    for inputs, _ in dataloader:
        inputs = inputs.cuda()
        inputs = Variable(inputs)
        outputs = mlp(inputs)
        pred = np.append(pred,
                         outputs.topk(1)[1].data.view(1, -1).cpu().numpy())
    return pred

# run the training epoch 100 times and test the result
print(">>> training model with mlp")
epoch_loss = []
epoch_acc = []
for epoch in range(30):
    trainEpoch(trainloader, epoch)
    loss, acc, _ = validateModel(valloader, epoch)
    epoch_loss.append(loss)
    epoch_acc.append(acc)

print(">>> creating predictions with mlp")
pred = 1 - testModel(testloader)

print(">>> outputing predictions to local file")
pred = pred.astype(int)
submission = pd.DataFrame()
submission['id'] = test_id.astype(int) + 1
submission['target'] = pred
submission.to_csv('submission.csv', index=False)

epoch_performance = pd.DataFrame()
epoch_performance['epoch_id'] = np.arange(len(epoch_loss)) + 1
epoch_performance['epoch_loss'] = np.array(epoch_loss)
epoch_performance['epoch_acc'] = np.array(epoch_acc)
epoch_performance.to_csv('epoch_performance.csv', index=False)
