#%%
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

#%%
# GPU 사용여부를 결정
ngpu = 1    # 사용가능한
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
print(torch.cuda.is_available(), device)


### paras.
batch_size = 128
image_size = 32 # 이미지 크기. 모든 이미지를 변환하여 크기가 통일됩니다.
nc = 3  # 이미지의 채널 수로, RGB 이미지이기 때문에 3으로 설정합니다.

nz = 128 #100   # 잠재공간 벡터의 크기 (예. 생성자의 입력값 크기)
ngf = 32 #64    # 생성자를 통과하는 특징 데이터들의 채널 크기
ndf = 32 #64    # 구분자를 통과하는 특징 데이터들의 채널 크기


### SVHN torch vision dataloader
svhn_path = "/projects/dcgan/SVHN/"

transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.RandomCrop(image_size), 
    transforms.ToTensor(), # image 2 tensor
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # image normalization (mean=0, std=1)
])

test_dataset = dset.SVHN(root= svhn_path, split= 'test', transform = transform)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)


### Model
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

######################################################

#%%
model_path = "/projects/dcgan/output_model/"

### 모델 불러오기
# strict=False : 일치하지 않는 모듈을 무시
# map_location=device : 모델을 CPU 또는 GPU로 올바르게 이동
generator = Generator(ngpu).to(device)
generator.load_state_dict(torch.load(f'{model_path}/gan1_1_netG.pth', map_location=device), strict=False)

discriminator = Discriminator(ngpu).to(device)
discriminator.load_state_dict(torch.load(f'{model_path}/gan1_1_netD.pth', map_location=device), strict=False)


######################################################
p_real, p_fake = 0., 0.

generator.eval()
discriminator.eval()

for img_batch, label_batch in test_loader:
    img_batch, label_batch = img_batch.to(device), label_batch.to(device)

    with torch.autograd.no_grad():
        p_real += (torch.sum(discriminator(img_batch)).item()) / len(test_loader.dataset)
        z = torch.randn(img_batch.size(0), batch_size, 4, 4).to(device)  # 무작위 노이즈 벡터 생성
        p_fake += (torch.sum(discriminator(generator(z))).item()) / len(test_loader.dataset)
                
        ################
        fake = generator(z).detach().cpu()
        tmp_result = vutils.make_grid(fake, padding=2, normalize=True)
        
        plt.subplot(1,2,2)
        plt.axis("off")
        plt.title("Fake Images")
        plt.imshow(np.transpose(tmp_result,(1,2,0)))
        plt.show()


# # evaluate_model 함수 정의
# def evaluate_model(generator, discriminator):
#     p_real, p_fake = 0., 0.

#     generator.eval()
#     discriminator.eval()

#     for img_batch, label_batch in test_loader:
#         img_batch, label_batch = img_batch.to(device), label_batch.to(device)

#         with torch.autograd.no_grad():
#             p_real += (torch.sum(discriminator(img_batch)).item()) / len(test_loader.dataset)
#             z = torch.randn(img_batch.size(0), 100, 1, 1).to(device)  # 무작위 노이즈 벡터 생성
#             p_fake += (torch.sum(discriminator(generator(z))).item()) / len(test_loader.dataset)

#     return p_real, p_fake

# # evaluate_model 함수 호출
# p_real, p_fake = evaluate_model(generator, discriminator)

# print(f"Real Data Score: {p_real:.4f}")
# print(f"Fake Data Score: {p_fake:.4f}")




#%%
### 진짜 이미지 vs. 가짜 이미지
# dataloader에서 진짜 데이터들을 가져옵니다
real_batch = next(iter(test_loader))

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

######################################################
#%%
def check_condition(_generator):
    
    test_image = torch.empty(0).to(device)

    for i in range(10):
        test_label = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        test_label_encoded = tf.one_hot(test_label, num_classes=10).to(device)

        # create noise(latent vector) 'z'
        # fixed_noise = torch.randn(10, nz, 1, 1, device=device)
        fixed_noise = torch.randn(10, nz, 1, 1, device=device)
        _z_concat = torch.cat((fixed_noise, test_label_encoded), 1)

        test_image = torch.cat((test_image, _generator(_z_concat)), 0)

    _result = test_image.reshape(100, 1, 28, 28)
    save_image(_result, os.path.join(dir_name, 'CGAN_test_result.png'), nrow=10)


check_condition(netG)

fixed_noise = torch.randn(batch_size, nz, 1, 1, device=device)
fake = netG(fixed_noise).detach().cpu()
fake.shape


#%%
### test
def evaluate_model(generator, discriminator):
    p_real, p_fake = 0., 0.

    generator.eval()
    discriminator.eval()

    for img_batch, label_batch in test_loader:
        img_batch, label_batch = img_batch.to(device), label_batch.to(device)

        with torch.autograd.no_grad():
            p_real += (torch.sum(discriminator(img_batch)).item()) / 10000.
            z = torch.randn(img_batch.size(0), 100, 1, 1).to(device)  # 무작위 노이즈 벡터 생성
            p_fake += (torch.sum(discriminator(generator(z))).item()) / 10000.

    return p_real, p_fake

p_real_trace = []
p_fake_trace= []

p_real, p_fake = evaluate_model(generator, discriminator)

p_real_trace.append(p_real)
p_fake_trace.append(p_fake)

if((epoch+1)% 50 == 0):
    print('(epoch %i/200) p_real: %f, p_g: %f' % (epoch+1, p_real, p_fake))
    imshow_grid(G(sample_z(16)).view(-1, 1, 28, 28))

