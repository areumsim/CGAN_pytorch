# DCGAN
# ref. https://tutorials.pytorch.kr/beginner/dcgan_faces_tutorial.html

# CGAN +

import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

######################################################
random.seed(999)
torch.manual_seed(999)

## 데이터셋의 경로
# dataroot = "/root/arsim/data/DCGAN/SVHN" #"data/celeba"
svhn_path = "/projects/dcgan/SVHN/"
output_path = "/projects/dcgan/output_cgan_A/"
os.makedirs(output_path, exist_ok=True)

## paras.
workers = 2  # dataloader에서 사용할 쓰레드 수

image_size = 32  # 이미지 크기
nc = 3  # 이미지의 채널 수 (RGB - 3)

nz = 128  # 잠재공간 벡터의 크기 (예. 생성자의 입력값 크기)
ngf = 32  # 생성자를 통과하는 특징 데이터들의 채널 크기
ndf = 32  # 구분자를 통과하는 특징 데이터들의 채널 크기

batch_size = 11
num_epochs = 20  # 학습할 에폭 수
lr = 0.0002  # 옵티마이저의 학습률
beta1 = 0.5  # Adam 옵티마이저의 beta1 하이퍼파라미터

ngpu = 1  # 사용가능한 gpu

# The number of class label
n_class = 10
condition_size = n_class

######################################################
### SVHN torch vision dataloader
transform = transforms.Compose(
    [
        transforms.Resize(image_size),
        transforms.RandomCrop(image_size),
        transforms.ToTensor(),  # image 2 tensor
        transforms.Normalize(
            (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
        ),  # image normalization (mean=0, std=1)
        # transforms.Normalize(mean=params.dataset_mean, std=params.dataset_std)
    ]
)

## SVHN 데이터셋 다운로드
train_dataset = dset.SVHN(
    root=svhn_path, split="train", transform=transform, download=True
)
test_dataset = dset.SVHN(root=svhn_path, split="test", transform=transform)

train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset, batch_size=batch_size, shuffle=True
)
test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=4
)


# %%
# DATA
# http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html https://drive.google.com/drive/folders/0B7EVK8r0v71pTUZsaXdaSnZBZzg?resourcekey=0-rJlzl934LzC-Xp28GeIBzQ
# img_align_celeba.zip
# /path/to/celeba -> img_align_celeba -> 188242.jpg -> 173822.jpg -> 284702.jpg -> 537394.jpg ...

# dataset = dset.ImageFolder(
#     root=dataroot,
#     transform=transforms.Compose(
#         [
#             transforms.Resize(image_size),
#             transforms.CenterCrop(image_size),
#             transforms.ToTensor(),
#             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
#         ]
#     ),
# )

# dataloader = torch.utils.data.DataLoader(
#     dataset, batch_size=batch_size, shuffle=True, num_workers=workers
# )

######################################################
# GPU 사용여부
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
print(torch.cuda.is_available(), device)

# 학습 데이터들 중 몇가지 이미지
real_batch = next(iter(train_dataset))
plt.figure(figsize=(8, 8))
plt.axis("off")
plt.title("Training Images")
plt.imshow(
    np.transpose(
        vutils.make_grid(
            real_batch[0].to(device)[:64], padding=2, normalize=True
        ).cpu(),
        (1, 2, 0),
    )
)


######################################################
### 가중치 초기화
# netG 와 netD 에 적용시킬 커스텀 가중치 초기화 함수
# 모든 가중치의 평균을 0( mean=0 ), 분산을 0.02( stdev=0.02 )로 초기화
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


######################################################
# 생성자 코드
class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu

        self.conv1 = nn.ConvTranspose2d(
            nz + condition_size, ngf * 8, 4, 2, 1, bias=False
        )
        self.batchNorm1 = nn.BatchNorm2d(ngf * 8)
        self.conv2 = nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False)
        self.batchNorm2 = nn.BatchNorm2d(ngf * 4)
        self.conv3 = nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False)
        self.batchNorm3 = nn.BatchNorm2d(ngf * 2)
        self.conv4 = nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False)
        self.batchNorm4 = nn.BatchNorm2d(ngf)
        self.conv5 = nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False)

        self.relu = nn.ReLU(True)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.batchNorm1(x))
        x = self.relu(self.relu(x))

        x = self.relu(self.conv2(x))
        x = self.relu(self.batchNorm2(x))
        x = self.relu(self.relu(x))

        x = self.relu(self.conv3(x))
        x = self.relu(self.batchNorm3(x))
        x = self.relu(self.relu(x))

        x = self.relu(self.conv4(x))
        x = self.relu(self.batchNorm4(x))
        x = self.relu(self.relu(x))

        x = self.relu(self.conv5(x))
        x = self.relu(self.relu(x))

        x = self.tanh(x)
        return x


######################################################
# 생성자
netG = Generator(ngpu).to(device)
# 가중치 초기화
netG.apply(weights_init)

print(netG)

######################################################
### generate input test ##
# nn.ConvTranspose2d( nz, ngf * 8, 4, 2, 1, bias=False).cuda()(fixed_noise).shape

# nn.Sequential(
#         # 입력데이터 Z가 가장 처음 통과하는 전치 합성곱 계층입니다.
#         nn.ConvTranspose2d( nz, ngf * 8, 4, 2, 1, bias=False),
#         nn.BatchNorm2d(ngf * 8),
#         nn.ReLU(True),
#         # 위의 계층을 통과한 데이터의 크기. (ngf*8) x 4 x 4
#         nn.ConvTranspose2d(ngf * 8, ngf * 4, 3, 2, 1, bias=False),
#         nn.BatchNorm2d(ngf * 4),
#         nn.ReLU(True)).cuda()(fixed_noise).shape


# fixed_noise = torch.randn(batch_size, nz, 1, 1, device=device)

# random_labels = torch.randint(n_class, size=(batch_size,))
# test_label_encoded = (
#     nn.functional.one_hot(random_labels, num_classes=n_class)
#     .unsqueeze(2)
#     .unsqueeze(3)
#     .to(device)
# )

# fixed_noise_concat = torch.cat((fixed_noise, test_label_encoded), 1)

# fake = netG(fixed_noise_concat).detach().cpu()
# fake.shape


######################################################
# 구분자 코드
class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu

        self.Conv1 = nn.Conv2d(nc + condition_size, ndf, 3, 2, 1, bias=False)

        self.Conv2 = nn.Conv2d(ndf, ndf * 2, 3, 2, 1, bias=False)
        self.Batch2 = nn.BatchNorm2d(ndf * 2)

        self.Conv3 = nn.Conv2d(ndf * 2, ndf * 4, 3, 2, 1, bias=False)
        self.Batch3 = nn.BatchNorm2d(ndf * 4)

        self.Conv4 = nn.Conv2d(ndf * 4, ndf * 8, 3, 2, 1, bias=False)
        self.Batch4 = nn.BatchNorm2d(ndf * 8)

        self.Conv5 = nn.Conv2d(ndf * 8, 1, 3, 2, 1, bias=False)

        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.Conv1(x)
        x = self.leaky_relu(x)

        x = self.Conv2(x)
        x = self.Batch2(x)
        x - self.leaky_relu(x)

        x = self.Conv3(x)
        x - self.leaky_relu(x)
        x = self.Batch3(x)

        x = self.Conv4(x)
        x = self.Batch4(x)
        x - self.leaky_relu(x)

        x = self.Conv5(x)
        x = self.sigmoid(x)
        return x


######################################################
# 구분자
netD = Discriminator(ngpu).to(device)

netD.apply(weights_init)

print(netD)


######################################################
### discriminator test ###
# netD(real_batch[0].cuda())

# random_labels = torch.randint(n_class, size=(batch_size,))
# real_label_encoded = (
#     nn.functional.one_hot(real_batch[1], num_classes=n_class).unsqueeze(2).unsqueeze(3)
# )
# real_images_concat = torch.cat((real_batch[0], real_label_encoded), 1)

# netD(real_images_concat.cuda())


######################################################
## 손실함수 / 옵티마이저
criterion = nn.BCELoss()

# G의 학습상태를 확인할 잠재 공간 벡터
fixed_noise = torch.randn(batch_size, nz, 1, 1, device=device)

# 학습에 사용되는 라벨
real_label = 1.0
fake_label = 0.0

# G와 D에서 사용할 Adam옵티마이저를 생성
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))


######################################################
######################################################
## 학습

img_list = []
G_losses = []
D_losses = []
iters = 0

for epoch in range(num_epochs):
    for i, data in enumerate(train_loader, 0):
        netD.zero_grad()

        real_cpu = data[0].to(device)
        b_size = real_cpu.size(0)
        label = torch.full((b_size,), real_label, dtype=torch.float, device=device)

        # 진짜 데이터를 D에
        output = netD(real_cpu).view(-1)

        # 손실값 계산
        errD_real = criterion(output, label)
        # 역전파 - 변화도 계산
        errD_real.backward()
        D_x = output.mean().item()

        ## 가짜 데이터로 학습
        noise = torch.randn(b_size, nz, 1, 1, device=device)
        # G를 이용해 가짜 이미지를 생성
        fake = netG(noise)
        label.fill_(fake_label)

        # D로 데이터 판별
        output = netD(fake.detach()).view(-1)

        # D의 손실값을 계산
        errD_fake = criterion(output, label)
        # 역전파 - 변화도를 계산(accumulate)
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        # 가짜 이미지와 진짜 이미지의 손실값 합 / errD는 역전파에 사용 X
        errD = errD_real + errD_fake
        # D를 업데이트 합니다
        optimizerD.step()

        ############################
        # (2) G 신경망 업데이트 : log(D(G(z)))를 최대화
        ###########################
        # 생성자의 손실값 계산 - 진짜 라벨

        netG.zero_grad()
        label.fill_(real_label)
        # D를 업데이트했기 때문에, D에 다시 가짜 데이터를 통과 (D만 업데이트, G는 X)

        output = netD(fake).view(-1)
        # G의 손실값 계산
        errG = criterion(output, label)
        # G의 변화도 계산
        errG.backward()
        D_G_z2 = output.mean().item()
        # G 업데이트
        optimizerG.step()

        if i % 50 == 0:
            print(
                "[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f"
                % (
                    epoch,
                    num_epochs,
                    i,
                    len(train_dataset),
                    errD.item(),
                    errG.item(),
                    D_x,
                    D_G_z1,
                    D_G_z2,
                )
            )

        G_losses.append(errG.item())
        D_losses.append(errD.item())

        # fixed_noise를 통과시킨 G의 출력값 저장
        if (iters % 500 == 0) or (
            (epoch == num_epochs - 1) and (i == len(train_dataset) - 1)
        ):
            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu()
            tmp_result = vutils.make_grid(fake, padding=2, normalize=True)
            img_list.append(tmp_result)

            plt.imshow(np.transpose(tmp_result, (1, 2, 0)), animated=True)
            plt.savefig(f"{output_path}/output{epoch}_{iters}.jpg")

        iters += 1


######################################################


######################################################
### 결과
plt.figure(figsize=(10, 5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses, label="G")
plt.plot(D_losses, label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()


######################################################
### G의 학습 과정 시각화
fig = plt.figure(figsize=(8, 8))
plt.axis("off")
ims = [[plt.imshow(np.transpose(i, (1, 2, 0)), animated=True)] for i in img_list]
ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)

HTML(ani.to_jshtml())


######################################################
### 진짜 이미지 vs. 가짜 이미지
# dataloader에서 진짜 데이터들을 가져옵니다
real_batch = next(iter(train_dataset))

# 진짜 이미지
plt.figure(figsize=(15, 15))
plt.subplot(1, 2, 1)
plt.axis("off")
plt.title("Real Images")
plt.imshow(
    np.transpose(
        vutils.make_grid(
            real_batch[0].to(device)[:64], padding=5, normalize=True
        ).cpu(),
        (1, 2, 0),
    )
)

# 가짜 이미지
plt.subplot(1, 2, 2)
plt.axis("off")
plt.title("Fake Images")
plt.imshow(np.transpose(img_list[-1], (1, 2, 0)))
plt.show()
