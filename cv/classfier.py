import torch.optim as optim
import torch.utils.data
import torchvision
from torchvision import transforms as transforms
import numpy as np
import torch.nn as nn
from torch.autograd import Variable

# Cifar-10's labels
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

#hyper-parameter
batch_size = 100
epochs = 20
LR = 0.001  #learning rate

#use GPU or CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")

#  data pre-treatment
data_transform = {
    "train": transforms.Compose([transforms.RandomCrop(32, padding=4),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
    "val": transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

# load train data
trainset = torchvision.datasets.CIFAR10(root='./data',
                                        train=True,
                                        download=True,
                                        transform=data_transform["train"])

trainloader = torch.utils.data.DataLoader(dataset=trainset,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          num_workers=2)

# load test data
testset = torchvision.datasets.CIFAR10(root='./data',
                                       train=False,
                                       download=True,
                                       transform=data_transform["val"])
testloader = torch.utils.data.DataLoader(dataset=testset,
                                         batch_size=batch_size,
                                         shuffle=False,
                                         num_workers=2)

# ===========在下面引入预训练模型============
#load model
model = 

#====在下方修改模型全连接层，以符合分类类别====


#loss function and optimizer
optimizer = optim.Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss().to(device)

if __name__ == "__main__":
    #begain train
    accuracy = 0
    for epoch in range(1, epochs+1):
        print("\n===> epoch: %d/%d" % (epoch, epochs))
        print("train:")
        train_loss = 0
        train_correct = 0
        train_total = 0
        model.train()
        for batch_num, (inputs, labels) in enumerate(trainloader):
            # get the inputs
            inputs, labels = inputs.to(device), labels.to(device)
            # warp them in Variable
            inputs, labels = Variable(inputs), Variable(labels)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward
            outputs = model(inputs)
            # loss
            loss = criterion(outputs, labels)
            # backward
            loss.backward()
            # update weights
            optimizer.step()
            train_loss += loss.item()
            prediction = torch.max(outputs, 1)  # second param "1" represents the dimension to be reduced
            train_total += labels.size(0)
            # train_correct incremented by one if predicted right
            train_correct += np.sum(prediction[1].cpu().numpy() == labels.cpu().numpy())

            if batch_num % 50 == 49:  # print every 50 batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch, batch_num + 1, train_loss / (batch_num + 1)))
        print("Loss: %.4f | Acc: %.3f%% (%d/%d)"
                         % (train_loss / (len(trainloader)), 100. * train_correct / train_total, train_correct, train_total))

        print("test:")
        model.eval()
        test_loss = 0
        test_correct = 0
        testTotal = 0
        for batch_num, (data, target) in enumerate(testloader):
            data, target = data.to(device), target.to(device)
            outputs = model(Variable(data))
            loss = criterion(outputs, target)
            test_loss += loss.item()
            prediction = torch.max(outputs, 1)
            testTotal += target.size(0)
            test_correct += np.sum(prediction[1].cpu().numpy() == target.cpu().numpy())
        print("Loss: %.4f | Acc: %.3f%% (%d/%d)"
              % (test_loss / len(testloader), 100. * test_correct / testTotal, test_correct, testTotal))
        accuracy = max(accuracy, test_correct/testTotal)
        if epoch == epochs:
            print("===> BEST ACC. PERFORMANCE: %.3f%%" % (accuracy * 100))
    torch.save(model.state_dict(), 'cifarNet.pth')
    
