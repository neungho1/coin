from torchvision import transforms
import torch
from torchvision.utils import save_image
import imageio
import argparse
import os
from siren_copy import Siren
import util
import numpy as np
from training import Trainer
import getpass
import random

parser = argparse.ArgumentParser()
parser.add_argument("-ld", "--logdir", help="Path to save logs", default=f"/tmp/{getpass.getuser()}")
parser.add_argument("-ni", "--num_iters", help="Number of iterations to train for", type=int, default=10000)
parser.add_argument("-lr", "--learning_rate", help="Learning rate", type=float, default=2e-4)
parser.add_argument("-se", "--seed", help="Random seed", type=int, default=random.randint(1, int(1e6)))
parser.add_argument("-fd", "--full_dataset", help="Whether to use full dataset", action='store_true')
parser.add_argument("-iid", "--image_id", help="Image ID to train on, if not the full dataset", type=int, default=15)
parser.add_argument("-lss", "--layer_size", help="Layer sizes as list of ints", type=int, default=28)
parser.add_argument("-nl", "--num_layers", help="Number of layers", type=int, default=10)
parser.add_argument("-w0", "--w0", help="w0 parameter for SIREN model.", type=float, default=30.0)
parser.add_argument("-w0i", "--w0_initial", help="w0 parameter for first layer of SIREN model.", type=float, default=30.0)

args = parser.parse_args()


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dtype = torch.float32
if not os.path.exists(args.logdir):
    os.makedirs(args.logdir)


func_rep = Siren(
    dim_in=2,
    dim_hidden=args.layer_size,
    dim_out=6,
    num_layers=args.num_layers,
    final_activation=torch.nn.Identity(),
    w0_initial=args.w0_initial,
    w0=args.w0
).to(device)
trainer = Trainer(func_rep, lr=args.learning_rate)


model_path = args.logdir +'/best_model_15.pt'
load_state= torch.load(model_path)
func_rep.load_state_dict(load_state)







from torch.nn import GaussianNLLLoss
import torch
import mvtechinr
import os
import numpy as np

from einops import rearrange
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve
from utility_fun import *

prdt = "01"
patch_size = 1


#Dataset
data = mvtechinr.Mvtec(1,product=prdt)



#### testing #####
loader = [data.train_loader,data.test_norm_loader,data.test_anom_loader]

def Patch_Overlap_Score(threshold, data_load = loader[1:], upsample =1):
    
    norm_loss_t = []
    normalised_score_t = []
    mask_score_t = []

    loss3_tn = []

    loss3_ta = []
    
    score_tn = []
    score_ta = []
    

    for n,data in enumerate(data_load):
        total_loss_all = []
        for c,(i, j) in enumerate(data):
            if i.size(1)==1:
                i = torch.stack([i,i,i]).squeeze(2).permute(1,0,2,3)
            reconstructions = func_rep(i.cuda())
            
            
            #Loss calculations
            predicted_mean, predicted_std = torch.split(reconstructions, 3, dim=1)
            var = torch.square(predicted_std)
            loss = GaussianNLLLoss(predicted_mean,i,var)
            
            if n == 0 :
                loss3_tn.append(loss.sum().detach().cpu().numpy())
            if n == 1:
                loss3_ta.append(loss.sum().detach().cpu().numpy())
            print(c,c,c,c,c)
            if upsample==0 :
                #Mask patch
                mask_patch = rearrange(j.squeeze(0).squeeze(0), '(h p1) (w p2) -> (h w) p1 p2', p1 = patch_size, p2 = patch_size)
                mask_patch_score = Binarization(mask_patch.sum(1).sum(1),0.)
                mask_score_t.append(mask_patch_score) # Storing all masks
                norm_score = Binarization(norm_loss_t[-1], threshold)
                m = torch.nn.UpsamplingNearest2d((512,512))
                score_map = m(torch.tensor(norm_score.reshape(-1,1,512//patch_size,512//patch_size)))
               
                
                normalised_score_t.append(norm_score)# Storing all patch scores
            elif upsample == 1:
                mask_score_t.append(j.squeeze(0).squeeze(0).cpu().numpy()) # Storing all masks
                
                m = torch.nn.UpsamplingBilinear2d((512,512))
                
                norm_score = norm_loss_t[-1].reshape(-1,1,512//patch_size,512//patch_size)
                score_map = m(torch.tensor(norm_score))
                score_map = Filter(score_map , type =1) 

                   
                normalised_score_t.append(score_map) # Storing all score maps
                
            ## Plotting
            if c%5 == 0:
                plot(i,j,score_map[0][0])
            if n == 0:
                score_tn.append(score_map.max())
            if n ==1:
                score_ta.append(score_map.max())
                
                
        if n == 0 :
            t_loss_all_normal = total_loss_all
        if n == 1:
            t_loss_all_anomaly = total_loss_all
        
    ## PRO Score            
    scores = np.asarray(normalised_score_t).flatten()
    masks = np.asarray(mask_score_t).flatten()
    PRO_score = roc_auc_score(masks, scores)
    
    ## Image Anomaly Classification Score (AUC)
    roc_data = np.concatenate((t_loss_all_normal, t_loss_all_anomaly))
    roc_targets = np.concatenate((np.zeros(len(t_loss_all_normal)), np.ones(len(t_loss_all_anomaly))))
    AUC_Score_total = roc_auc_score(roc_targets, roc_data)
    
    # AUC Precision Recall Curve
    precision, recall, thres = precision_recall_curve(roc_targets, roc_data)
    AUC_PR = auc(recall, precision)

    
    return PRO_score, AUC_Score_total, AUC_PR

if __name__=="__main__":
    
    thres = 0.5
    PRO, AUC, AUC_PR = Patch_Overlap_Score(threshold=thres)

    print(f'PRO Score: {PRO} \nAUC Total: {AUC} \nPR_AUC Total: {AUC_PR}')
