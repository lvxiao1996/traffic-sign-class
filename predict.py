import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch
import os
from torch.autograd import Variable
import numpy as np
import time
import torchvision.models as models
import matplotlib.pyplot as plt

from PIL import Image
import os




print('please select a test picture:')
image=input()
# img_path = "data/test/1/011_0059.png"
img_path = image
traintransform = transforms.Compose([
    transforms.ColorJitter(brightness=0.1),
    transforms.Resize([224, 224]),
    transforms.ToTensor(),  # 将图片数据变为tensor格式
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
img = Image.open(img_path)
plt.imshow(img)
plt.show()
# [C, H, W]，转换图像格式
img = traintransform(img)
# [N, C, H, W]，增加一个维度N
img = torch.unsqueeze(img, dim=0)
# required_grad=False
model = torch.load('bestmodel.pth')
model = model.cuda()
# model = models.resnet34()
# model.load_state_dict(torch.load(best_model_path, map_location=lambda storage, loc: storage), strict=True) # 利用cpu进行测试
model.eval()
image = Variable(img.cuda())  # 如果不使用GPU，删除.cuda()
pred = model(image)
max_value, max_index = torch.max(pred, 1)
pred_label = max_index.cpu().numpy()
print("softmax code:",end='')
A=torch.softmax(pred,1).detach().cpu().numpy()
onehot=np.where(A==A.max(),1,0)
print(A)
print("onehot code:",end='')
print(onehot)
print("The class is:",end='')
print(pred_label[0])
if pred_label[0]==0:
    print('禁止鸣笛')
elif pred_label[0]==1:
    print('禁止转向')
elif pred_label[0]==2:
    print('限速')
elif pred_label[0]==3:
    print('靠边行驶')
else:
    print('分类失败')


