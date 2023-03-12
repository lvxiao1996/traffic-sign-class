import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch
import os
from torch.autograd import Variable
import numpy as np
import time
import torchvision.models as models
import matplotlib.pyplot as plt
import resnet34
torch.backends.cudnn.benchmark = True


batch_size = 16
learning_rate = 1e-4
epoches = 80
num_class = 4


#数据路径
trainpath = 'data/train/'
valpath = 'data/val/'

#Dataloader
def data_load(trainpath,valpath):
    traintransform = transforms.Compose([
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.1),
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    valtransform = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),  # 将图片数据变为tensor格式
    ])
    trainData = dsets.ImageFolder(trainpath, transform=traintransform)  # 读取训练集，标签就是train目录下的文件夹的名字，图像保存在格子标签下的文件夹里
    valData = dsets.ImageFolder(valpath, transform=valtransform)
    trainLoader = torch.utils.data.DataLoader(dataset=trainData, batch_size=batch_size, shuffle=True)
    valLoader = torch.utils.data.DataLoader(dataset=valData, batch_size=batch_size, shuffle=False)


    val_sum = sum([len(x) for _, _, x in os.walk(os.path.dirname(valpath))])
    train_sum = sum([len(x) for _, _, x in os.walk(os.path.dirname(trainpath))])
    return trainLoader,valLoader,val_sum,train_sum


# 加载网络模型
def build_model():
    
    model=resnet34.ResNet()

    model = model.cuda()  # 如果有GPU，而且确认使用则保留；如果没有GPU，请删除
    criterion = torch.nn.CrossEntropyLoss()  # 定义损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    return model,criterion,optimizer



def train(model, optimizer, criterion):
    model.train()
    total_loss = 0
    train_corrects = 0
    trainLoader, valLoader, val_sum, train_sum = data_load(trainpath,valpath)
    for i, (image, label) in enumerate(trainLoader):
        image = Variable(image.cuda())  # 同理
        label = Variable(label.cuda())  # 同理
        optimizer.zero_grad()

        target = model(image)
        loss = criterion(target, label)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        max_value, max_index = torch.max(target, 1)
        pred_label = max_index.cpu().numpy()
        true_label = label.cpu().numpy()
        train_corrects += np.sum(pred_label == true_label)

    return total_loss / float(len(trainLoader)), train_corrects / train_sum


def evaluate(model, criterion):
    trainLoader, valLoader, val_sum, train_sum = data_load(trainpath, valpath)
    model.eval()
    corrects = eval_loss = 0
    with torch.no_grad():
        for image, label in valLoader:
            image = Variable(image.cuda())  
            label = Variable(label.cuda())  

            pred = model(image)
            loss = criterion(pred, label)

            eval_loss += loss.item()

            max_value, max_index = torch.max(pred, 1)
            pred_label = max_index.cpu().numpy()
            true_label = label.cpu().numpy()
            corrects += np.sum(pred_label == true_label)

    return eval_loss / float(len(valLoader)), corrects, corrects / val_sum




def main():
    train_loss = []
    valid_loss = []
    accuracy = []
    bestacc=0
    model, criterion, optimizer = build_model()
    for epoch in range(1, epoches + 1):
        epoch_start_time = time.time()
        loss, train_acc = train(model, optimizer, criterion)
        train_loss.append(loss)
        print('| start of epoch {:3d} | time: {:2.2f}s | train_loss {:5.6f}  | train_acc {}'.format(epoch, time.time() - epoch_start_time, loss, train_acc))
        loss, corrects, acc = evaluate(model, criterion)
        valid_loss.append(loss)
        accuracy.append(acc)
        if epoch%10==0:
            torch.save(model, 'model_training/epoch_%d_model.pth'%(epoch))
        if acc > bestacc:
            torch.save(model,'model_training/bestmodel.pth')
            bestacc = acc

        print('| end of epoch {:3d} | time: {:2.2f}s | test_loss {:.6f} | accuracy {}'.format(epoch, time.time() - epoch_start_time, loss, acc))


    print("**********ending*********")
    plt.plot(train_loss)
    plt.plot(valid_loss)
    plt.title('loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig("./loss.jpg")
    plt.show()
    plt.cla()
    plt.plot(accuracy)
    plt.title('acc')
    plt.ylabel('acc')
    plt.xlabel('epoch')
    plt.savefig("./acc.jpg")
    plt.show()









if __name__ == "__main__":
    main()



