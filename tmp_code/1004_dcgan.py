
# DCGAN
# ref. https://tutorials.pytorch.kr/beginner/dcgan_faces_tutorial.html

#%%
from __future__ import print_function

#%matplotlib inline
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

# 코드 실행결과의 동일성을 위해 무작위 시드를 설정합니다
manualSeed = 999
#manualSeed = random.randint(1, 10000) # 만일 새로운 결과를 원한다면 주석을 없애면 됩니다
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)
######################################################

#%%
# 데이터셋의 경로
# dataroot = "/root/arsim/data/DCGAN/SVHN" #"data/celeba"
svhn_path = "/projects/dcgan/SVHN/"
output_path = "/projects/dcgan/output/"
os.makedirs(output_path, exist_ok=True)

workers = 2 # dataloader에서 사용할 쓰레드 수

image_size = 32 # 이미지 크기. 모든 이미지를 변환하여 크기가 통일됩니다.
nc = 3  # 이미지의 채널 수로, RGB 이미지이기 때문에 3으로 설정합니다.

nz = 128 #100   # 잠재공간 벡터의 크기 (예. 생성자의 입력값 크기)
ngf = 32 #64    # 생성자를 통과하는 특징 데이터들의 채널 크기
ndf = 32 #64    # 구분자를 통과하는 특징 데이터들의 채널 크기

batch_size = 128
num_epochs = 20 # 학습할 에폭 수
lr = 0.0002 # 옵티마이저의 학습률
beta1 = 0.5 # Adam 옵티마이저의 beta1 하이퍼파라미터 

ngpu = 1    # 사용가능한
######################################################

#%%
### SVHN torch vision dataloader
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.RandomCrop(image_size), 
    transforms.ToTensor(), # image 2 tensor
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # image normalization (mean=0, std=1)
    # transforms.Normalize(mean=params.dataset_mean, std=params.dataset_std)
])

# SVHN 데이터셋 다운로드   
train_dataset = dset.SVHN(root=svhn_path, split='train', transform=transform)
test_dataset = dset.SVHN(root= svhn_path, split= 'test', transform = transform)
# data_extra = dset.SVHN(root= svhn_path, split= 'extra', transform = transform, download= True)

# data = torch.utils.data.ConcatDataset((data_train, data_extra))
# dataloader = torch.utils.data.DataLoader(dataset=data, batch_size=batch_size, shuffle=True)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)


# # 데이터셋 크기 확인
print(f"Train dataset size: {len(train_loader.dataset)}")   # Train dataset size: 73257
print(f"Test dataset size: {len(test_loader.dataset)}") # Test dataset size: 26032
######################################################

#%%
# GPU 사용여부를 결정
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
print(torch.cuda.is_available(), device)

# 학습 데이터들 중 몇가지 이미지
real_batch = next(iter(train_dataset))
plt.figure(figsize=(8,8))
plt.axis("off")
plt.title("Training Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))
######################################################

#%%
### 가중치 초기화
# ``netG`` 와 ``netD`` 에 적용시킬 커스텀 가중치 초기화 함수
# 모든 가중치의 평균을 0( ``mean=0`` ), 분산을 0.02( ``stdev=0.02`` )로 초기화
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

######################################################


#%%
# 생성자 코드
class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # 입력데이터 Z가 가장 처음 통과하는 전치 합성곱 계층입니다.
            nn.ConvTranspose2d( nz, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # 위의 계층을 통과한 데이터의 크기. ``(ngf*8) x 4 x 4``
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # 위의 계층을 통과한 데이터의 크기. ``(ngf*4) x 8 x 8``
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # 위의 계층을 통과한 데이터의 크기. ``(ngf*2) x 16 x 16``
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # 위의 계층을 통과한 데이터의 크기. ``(ngf) x 32 x 32``
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # 위의 계층을 통과한 데이터의 크기. ``(nc) x 64 x 64``
        )

    def forward(self, input):
        return self.main(input)


netG = Generator(ngpu).to(device)

# 필요한 경우 multi-GPU를 설정 해주세요
if (device.type == 'cuda') and (ngpu > 1):
    netG = nn.DataParallel(netG, list(range(ngpu)))

# 모든 가중치 초기화
netG.apply(weights_init)

# 모델의 구조를 출력합니다
print(netG)
######################################################

#%%
### Generator 계층 test / 확인
# nn.ConvTranspose2d( nz, ngf * 8, 4, 2, 1, bias=False).cuda()(fixed_noise).shape

# nn.Sequential(
#         # 입력데이터 Z가 가장 처음 통과하는 전치 합성곱 계층입니다.
#         nn.ConvTranspose2d( nz, ngf * 8, 4, 2, 1, bias=False),
#         nn.BatchNorm2d(ngf * 8),
#         nn.ReLU(True),
#         # 위의 계층을 통과한 데이터의 크기. ``(ngf*8) x 4 x 4``
#         nn.ConvTranspose2d(ngf * 8, ngf * 4, 3, 2, 1, bias=False),
#         nn.BatchNorm2d(ngf * 4),
#         nn.ReLU(True)).cuda()(fixed_noise).shape

fixed_noise = torch.randn(batch_size, nz, 1, 1, device=device)
fake = netG(fixed_noise).detach().cpu()
fake.shape
######################################################


#%%
# 구분자 코드
class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # 입력 데이터의 크기는 ``(nc) x 64 x 64`` 입니다
            nn.Conv2d(nc, ndf, 3, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # 위의 계층을 통과한 데이터의 크기. ``(ndf) x 32 x 32``
            nn.Conv2d(ndf, ndf * 2, 3, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # 위의 계층을 통과한 데이터의 크기. ``(ndf*2) x 16 x 16``
            nn.Conv2d(ndf * 2, ndf * 4, 3, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # 위의 계층을 통과한 데이터의 크기. ``(ndf*4) x 8 x 8``
            nn.Conv2d(ndf * 4, ndf * 8, 3, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # 위의 계층을 통과한 데이터의 크기. ``(ndf*8) x 4 x 4``
            nn.Conv2d(ndf * 8, 1, 3, 2, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)
#%%
netD = Discriminator(ngpu).to(device)

# 필요한 경우 multi-GPU를 설정 해주세요
if (device.type == 'cuda') and (ngpu > 1):
    netD = nn.DataParallel(netD, list(range(ngpu)))

# 모든 가중치 초기화
netD.apply(weights_init)

# 모델의 구조를 출력합니다
print(netD)
######################################################


#%%
### Discriminator test
# real_batch[0].shape : [3, 32, 32]
fixed_real = torch.randn(batch_size, *real_batch[0].shape, device=device)
test = netD(fixed_real).detach().cpu()
print(test.shape, "\n", test[:2])

#%%
reshaped_tensor = real_batch[0].unsqueeze(0)
netD(reshaped_tensor.cuda())
######################################################


#%%
## 손실함수와 옵티마이
# ``BCELoss`` 함수의 인스턴스를 초기화합니다
criterion = nn.BCELoss()

# 생성자의 학습상태를 확인할 잠재 공간 벡터를 생성합니다
fixed_noise = torch.randn(batch_size, nz, 1, 1, device=device)

# 학습에 사용되는 참/거짓의 라벨을 정합니다
real_label = 1.
fake_label = 0.

# G와 D에서 사용할 Adam옵티마이저를 생성합니다
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))
######################################################


######################################################
######################################################
#%%
## 학습

# 학습상태를 체크하기 위해 손실값들을 저장합니다
img_list = []
G_losses = []
D_losses = []
iters = 0

print("Starting Training Loop...")
# 에폭(epoch) 반복
for epoch in range(num_epochs):
    # 한 에폭 내에서 배치 반복
    for i, data in enumerate(train_loader, 0):

        ############################
        # (1) D 신경망을 업데이트 합니다: log(D(x)) + log(1 - D(G(z)))를 최대화 합니다
        ###########################
        ## 진짜 데이터들로 학습을 합니다
        netD.zero_grad()
        # 배치들의 사이즈나 사용할 디바이스에 맞게 조정합니다
        real_cpu = data[0].to(device)
        b_size = real_cpu.size(0)
        label = torch.full((b_size,), real_label,
                           dtype=torch.float, device=device)
        # 진짜 데이터들로 이루어진 배치를 D에 통과시킵니다
        output = netD(real_cpu).view(-1)

        # 손실값을 구합니다
        errD_real = criterion(output, label)
        # 역전파의 과정에서 변화도를 계산합니다
        errD_real.backward()
        D_x = output.mean().item()

        ## 가짜 데이터들로 학습을 합니다
        # 생성자에 사용할 잠재공간 벡터를 생성합니다
        noise = torch.randn(b_size, nz, 1, 1, device=device)
        # G를 이용해 가짜 이미지를 생성합니다
        fake = netG(noise)
        label.fill_(fake_label)
        # D를 이용해 데이터의 진위를 판별합니다
        output = netD(fake.detach()).view(-1)
        # D의 손실값을 계산합니다
        errD_fake = criterion(output, label)
        # 역전파를 통해 변화도를 계산합니다. 이때 앞서 구한 변화도에 더합니다(accumulate)
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        # 가짜 이미지와 진짜 이미지 모두에서 구한 손실값들을 더합니다
        # 이때 errD는 역전파에서 사용되지 않고, 이후 학습 상태를 리포팅(reporting)할 때 사용합니다
        errD = errD_real + errD_fake
        # D를 업데이트 합니다
        optimizerD.step()

        ############################
        # (2) G 신경망을 업데이트 합니다: log(D(G(z)))를 최대화 합니다
        ###########################
        netG.zero_grad()
        label.fill_(real_label)  # 생성자의 손실값을 구하기 위해 진짜 라벨을 이용할 겁니다
        # 우리는 방금 D를 업데이트했기 때문에, D에 다시 가짜 데이터를 통과시킵니다.
        # 이때 G는 업데이트되지 않았지만, D가 업데이트 되었기 때문에 앞선 손실값가 다른 값이 나오게 됩니다
        output = netD(fake).view(-1)
        # G의 손실값을 구합니다
        errG = criterion(output, label)
        # G의 변화도를 계산합니다
        errG.backward()
        D_G_z2 = output.mean().item()
        # G를 업데이트 합니다
        optimizerG.step()

        # 훈련 상태를 출력합니다
        if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, num_epochs, i, len(train_dataset),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        # 이후 그래프를 그리기 위해 손실값들을 저장해둡니다
        G_losses.append(errG.item())
        D_losses.append(errD.item())

        # fixed_noise를 통과시킨 G의 출력값을 저장해둡니다
        if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(train_dataset)-1)):
            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu()
            tmp_result = vutils.make_grid(fake, padding=2, normalize=True)
            img_list.append(tmp_result)
            
            plt.imshow(np.transpose(tmp_result,(1,2,0)), animated=True)
            plt.savefig(f"{output_path}/output{epoch}_{iters}.jpg")
            
        iters += 1
######################################################


# %%
# Save model 
output_path = "/projects/dcgan/output_model/"
os.makedirs(output_path, exist_ok=True)

torch.save(netG,  f'{output_path}/gan1_1_netG.pth')
torch.save(netD, f'{output_path}/gan1_1_netD.pth')

torch.save(netG.state_dict(),  f'{output_path}/gan1_1_netG_state_dict.pth')
torch.save(netD.state_dict(), f'{output_path}/gan1_1_netD_state_dict.pth')

torch.save({
    'epoch': num_epochs,
    'model_G_state_dict': netG.state_dict(),
    'model_D_state_dict': netD.state_dict(),
    'optimizer_G_state_dict': optimizerG.state_dict(),
    'optimizer_D_state_dict': optimizerD.state_dict()
}, f'{output_path}/gan1_1_model_weights.pth')
######################################################

#%%
### 결과
plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses,label="G")
plt.plot(D_losses,label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()


######################################################
#%%
### G의 학습 과정 시각화
fig = plt.figure(figsize=(8,8))
plt.axis("off")
ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in img_list]
ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)

HTML(ani.to_jshtml())


######################################################
#%%
### 진짜 이미지 vs. 가짜 이미지
# dataloader에서 진짜 데이터들을 가져옵니다
real_batch = next(iter(train_dataset))

# 진짜 이미지들을 화면에 출력합니다
plt.figure(figsize=(15,15))
plt.subplot(1,2,1)
plt.axis("off")
plt.title("Real Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(),(1,2,0)))

# 가짜 이미지들을 화면에 출력합니다
plt.subplot(1,2,2)
plt.axis("off")
plt.title("Fake Images")
plt.imshow(np.transpose(img_list[-1],(1,2,0)))
plt.show()
