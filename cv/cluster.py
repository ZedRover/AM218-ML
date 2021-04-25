from torchvision.utils import make_grid
from PIL import Image
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
from sklearn.cluster import KMeans
import torch.nn as nn

# get the pretrained model
model = models.resnet18(pretrained=True)
#print(model) #structure of the model
model.eval()

transform = transforms.Compose([transforms.Resize(256),
                               transforms.CenterCrop(224),
                               transforms.ToTensor(),
                               transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
#get data
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=200,
                                         shuffle=True, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def save_img(tensor, name):
    tensor = tensor.detach() 
    tensor = tensor.permute((1, 0, 2, 3))
    im = make_grid(tensor, normalize=True, scale_each=True, nrow=8, padding=2).permute((1, 2, 0))
    im = (im.data.numpy() * 255.).astype(np.uint8)
    Image.fromarray(im).save(name + '.jpg')

#200 images and labels
inputs, labels = iter(testloader).next()
#get the conv1 features
feature = model.conv1(inputs)  # [1, 64, 112, 112].
#save the image features
for i in range(200):
    save_img(feature[i].unsqueeze(0), 'features_' + classes[labels[i]] + str(i))
feature = feature.view(feature.size(0), -1)


print('构建K-means')
#===在下方补充k-means聚类方法相关代码======



