import torch
import torchvision
import torch.utils.data.distributed
from torchvision import transforms


def main():
    # 数据加载部分，直接利用torchvision中的datasets
    trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
    data_set = torchvision.datasets.MNIST("./", train=True, transform=trans, target_transform=None, download=True)
    data_loader_train = torch.utils.data.DataLoader(dataset=data_set, batch_size=256)
    # 网络搭建，调用torchvision中的resnet
    net = torchvision.models.resnet101(num_classes=10)
    net.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    net = net.cuda()
    print(net)

    # 定义loss与opt
    criterion = torch.nn.CrossEntropyLoss()
    opt = torch.optim.Adam(net.parameters(), lr=0.001)
    # # 网络训练  
    # for epoch in range(10):
    #     for i, data in enumerate(data_loader_train):
    #         images, labels = data
    #         images, labels = images.cuda(), labels.cuda()
    #         opt.zero_grad()
    #         outputs = net(images)
    #         loss = criterion(outputs, labels)
    #         loss.backward()
    #         opt.step()
    #         if i % 10 == 0:
    #             print("loss: {}".format(loss.item()))
    # # 保存checkpoint
    # torch.save(net, "my_net.pth")


if __name__ == "__main__":
    main()