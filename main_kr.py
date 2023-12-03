import argparse
import getpass
import os
import random
import torch
import util
from siren import Siren
from torchvision import transforms
from torchvision.utils import save_image
from training import Trainer
from einops import rearrange
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve
import torch
import pytorch_ssim
import torch.nn.functional as F
import os
import numpy as np
from utility_fun import *
import mdn1

# 명령행 인수 파싱
parser = argparse.ArgumentParser()
parser.add_argument("-ld", "--logdir", help="로그를 저장할 경로", default=f"/tmp/{getpass.getuser()}")
parser.add_argument("-ni", "--num_iters", help="학습 반복 횟수", type=int, default=500)
parser.add_argument("-lr", "--learning_rate", help="학습률", type=float, default=2e-4)
parser.add_argument("-se", "--seed", help="랜덤 시드", type=int, default=random.randint(1, int(1e6)))
parser.add_argument("-lss", "--layer_size", help="레이어 크기 목록", type=int, default=28)
parser.add_argument("-nl", "--num_layers", help="레이어 수", type=int, default=10)
parser.add_argument("-w0", "--w0", help="SIREN 모델의 w0 매개변수", type=float, default=30.0)
parser.add_argument("-w0i", "--w0_initial", help="SIREN 모델 첫 번째 레이어의 w0 매개변수", type=float, default=30.0)
args = parser.parse_args()
#정상이미지 여러개 train 
#nll.loss
#RGB = mean,std 6개 채널 
# torch 및 cuda 설정
dtype = torch.float32
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(device)

# 랜덤 시드 설정
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

if not os.path.exists(args.logdir):
    os.makedirs(args.logdir)

import mvtechinr
prdt = "01"
patch_size = 64
batch_size = 400
results = {'fp_bpp': [], 'hp_bpp': [], 'fp_psnr': [], 'hp_psnr': []}
#Dataset
data = mvtechinr.Mvtec(batch_size,product=prdt)
loader = [data.train_loader,data.test_norm_loader,data.test_anom_loader]

func_rep = Siren(
    dim_in=2,
    dim_hidden=args.layer_size,
    dim_out=3,
    num_layers=args.num_layers,
    final_activation=torch.nn.Identity(),
    w0_initial=args.w0_initial,
    w0=args.w0
).to(device)

trainer = Trainer(func_rep, lr=args.learning_rate)
img_3_list = []
img_1_list = []
for j, m in data.train_loader:
    if j.size(1)==1:
        j = torch.stack([j,j,j]).squeeze(2).permute(1,0,2,3)
    img_3_list.append(j)
    img_1_list.append(m)
    
print(len(j),len(m))
    


