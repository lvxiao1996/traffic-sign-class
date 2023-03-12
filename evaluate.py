import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch
import os
from torch.autograd import Variable
import numpy as np
import time
import torchvision.models as models
import matplotlib.pyplot as pl
from PIL import Image

testpath = 'data/test'
traintransform = transforms.Compose([
    transforms.RandomRotation(20),  # optional
    transforms.ColorJitter(brightness=0.1),
    transforms.Resize([224, 224]),
    transforms.ToTensor(),  # 将图片数据变为tensor格式
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
testData = dsets.ImageFolder(testpath, transform=traintransform)



model = torch.load('bestmodel.pth')
model = model.cuda()

# model = models.resnet34()
# model.load_state_dict(torch.load(best_model_path, map_location=lambda storage, loc: storage), strict=True) # 利用cpu进行测试

model.eval()
testLoader = torch.utils.data.DataLoader(dataset=testData, batch_size=1, shuffle=False)

for i, (image, label) in enumerate(testLoader):
    image = Variable(image.cuda())  # 如果不使用GPU，删除.cuda()
    # image = Variable(image)
    pred = model(image)
    pred = torch.argmax(pred)
    # max_value, max_index = torch.max(pred, 1)
    # pred_label = max_index.cpu().numpy()
    print(pred)


