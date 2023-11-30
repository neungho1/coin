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
torch.set_default_tensor_type('torch.cuda.FloatTensor' if torch.cuda.is_available() else 'torch.FloatTensor')
ssim_loss = pytorch_ssim.SSIM()


# 랜덤 시드 설정
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

if not os.path.exists(args.logdir):
    os.makedirs(args.logdir)

import mvtechinr
prdt = "01"
patch_size = 64
results = {'fp_bpp': [], 'hp_bpp': [], 'fp_psnr': [], 'hp_psnr': []}
#Dataset
data = mvtechinr.Mvtec(1,product=prdt)
loader = [data.train_loader,data.test_norm_loader,data.test_anom_loader]
mdn_model = mdn1.MDN().cuda()

func_rep = Siren(
    dim_in=2,
    dim_hidden=args.layer_size,
    dim_out=3,
    num_layers=args.num_layers,
    final_activation=torch.nn.Identity(),
    w0_initial=args.w0_initial,
    w0=args.w0
).to(device)

def Patch_Overlap_Score(threshold, data_load = loader[1:], upsample =1):

    loss1_tn = []
    loss2_tn = []
    loss3_tn = []
    loss1_ta = []
    loss2_ta = []
    loss3_ta = []
    norm_loss_t = []
    normalised_score_t = []
    mask_score_t = []

    score_tn = []
    score_ta = []



    for n,data in enumerate(data_load):
        total_loss_all = []
        for c,(i, j) in enumerate(data):
            if i.size(1)==1:
                i = torch.stack([i,i,i]).squeeze(2).permute(1,0,2,3)
            print(c,c,c,c,c,c,c,c,)
            #print(i.size())
            img = i.squeeze(0)
            #print(img.size())

            
            trainer = Trainer(func_rep, lr=args.learning_rate)
            coordinates, features = util.to_coordinates_and_features(img)
            coordinates, features = coordinates.to(device, dtype), features.to(device, dtype)

            model_size = util.model_size_in_bits(func_rep) / 8000.
            print(f'Model size: {model_size:.1f}kB')
            fp_bpp = util.bpp(model=func_rep, image=img)
            print(f'Full precision bpp: {fp_bpp:.2f}')

            vector = trainer.train(coordinates, features, num_iters=args.num_iters)
            print(f'Best training psnr: {trainer.best_vals["psnr"]:.2f}')
            #print(vector,"\n",vector.size())

            results['fp_bpp'].append(fp_bpp)
            results['fp_psnr'].append(trainer.best_vals['psnr'])

            func_rep.load_state_dict(trainer.best_model)
            with torch.no_grad():
                img_recon = func_rep(coordinates).reshape(img.shape[1], img.shape[2], 3).permute(2, 1, 0)
                save_image(torch.clamp(img_recon, 0, 1).to('cpu'), args.logdir + f'/{n}_reconstruction_{c}.png')
                #print(img_recon.size())
           
            #pi, mu, sigma = mdn_model()

            #Loss calculations
            #print(img.cuda().size())
            print(img_recon.size())

            loss1 = F.mse_loss(img_recon,i.cuda().squeeze(0), reduction='mean') #Rec Loss
            loss2 = -ssim_loss(i.cuda(), img_recon)
            #loss3 = mdn1.mdn_loss_function(vector,mu,sigma,pi, test= True)
            loss = loss1 - loss2
            # 이미지 뽑기
            loss3 = img_recon -i.cuda()
            loss = loss1 - loss2 +loss3.max()
            print(loss3.size())
            #print(loss3,"\n------------\n",loss3.size(),loss3.squeeze(0).size(),loss3.squeeze(0).squeeze(0).size())
            loss3 = loss3.squeeze(0)
            loss3 = torch.mean(loss3, dim=0, keepdim=True)
            print(loss3.size())
            loss3.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)
            total_loss_all.append(loss.detach().cpu().numpy())
            print(loss3.size(),"--")
            norm_loss_t.append(loss3.detach().cpu().numpy())


            if upsample==0 :
                #Mask patch
                mask_patch = rearrange(j.squeeze(0).squeeze(0), '(h p1) (w p2) -> (h w) p1 p2', p1 = patch_size, p2 = patch_size)
                print(mask_patch.shape,'--')
                mask_patch_score = Binarization(mask_patch.sum(1).sum(1).cpu().numpy(),0.5)
                print(mask_patch_score.shape)
                mask_score_t.append(mask_patch_score) # Storing all masks
                norm_score = norm_loss_t[-1]
                print(norm_score.shape)
                normalised_score_t.append(norm_score)# Storing all patch scores

            elif upsample == 1:
                
                print(j.squeeze(0).squeeze(0).cpu().numpy().shape)
                mask_score_t.append(j.squeeze(0).squeeze(0).cpu().numpy()) # Storing all masks

                m = torch.nn.UpsamplingBilinear2d((512,512))
                print(norm_loss_t[-1].shape)

                norm_score = norm_loss_t[-1].reshape(-1,1,512//,512)
                score_map = m(torch.tensor(norm_score))
                score_map = Filter(score_map.cpu().numpy(), type =1) 


                   
                normalised_score_t.append(score_map) 

            ## Plotting
            """
            if c%5 == 0:
                plot(i,j,score_map[0][0])
            if n == 0:
                score_tn.append(score_map.max())
            if n ==1:
                score_ta.append(score_map.max())
            
            if n == 0 :
                loss1_tn.append(loss1.detach().cpu().numpy())
                loss2_tn.append(loss2.detach().cpu().numpy())
                
            if n == 1:
                loss1_ta.append(loss1.detach().cpu().numpy())
                loss2_ta.append(loss2.detach().cpu().numpy())
            """
        
        if n == 0 :
            t_loss_all_normal = total_loss_all
        if n == 1:
            t_loss_all_anomaly = total_loss_all
        
    ## PRO Score            
    scores = np.asarray(normalised_score_t).flatten()
    masks = np.asarray(mask_score_t).flatten()
    print(len(scores),len(masks))
    PRO_score = roc_auc_score(masks, scores)
    
    ## Image Anomaly Classification Score (AUC)
    roc_data = np.concatenate((t_loss_all_normal, t_loss_all_anomaly))
    roc_targets = np.concatenate((np.zeros(len(t_loss_all_normal)), np.ones(len(t_loss_all_anomaly))))
    AUC_Score_total = roc_auc_score(roc_targets, roc_data)
    
    # AUC Precision Recall Curve
    precision, recall, thres = precision_recall_curve(roc_targets, roc_data)
    AUC_PR = auc(recall, precision)

    
    return PRO_score, AUC_Score_total, AUC_PR


PRO, AUC, AUC_PR = Patch_Overlap_Score(threshold=0.5)

print(f'PRO Score: {PRO} \nAUC Total: {AUC} \nPR_AUC Total: {AUC_PR}')