import argparse
import getpass
import imageio
import json
import os
import random
import torch
import util
from siren import Siren
from torchvision import transforms
from torchvision.utils import save_image
from training import Trainer


parser = argparse.ArgumentParser()
parser.add_argument("-ld", "--logdir", help="Path to save logs", default=f"/tmp/{getpass.getuser()}")
parser.add_argument("-ni", "--num_iters", help="Number of iterations to train for", type=int, default=1000)
parser.add_argument("-lr", "--learning_rate", help="Learning rate", type=float, default=2e-4)
parser.add_argument("-se", "--seed", help="Random seed", type=int, default=random.randint(1, int(1e6)))
parser.add_argument("-fd", "--full_dataset", help="Whether to use full dataset", action='store_true')
parser.add_argument("-iid", "--image_id", help="Image ID to train on, if not the full dataset", type=int, default=15)
parser.add_argument("-lss", "--layer_size", help="Layer sizes as list of ints", type=int, default=28)
parser.add_argument("-nl", "--num_layers", help="Number of layers", type=int, default=10)
parser.add_argument("-w0", "--w0", help="w0 parameter for SIREN model.", type=float, default=30.0)
parser.add_argument("-w0i", "--w0_initial", help="w0 parameter for first layer of SIREN model.", type=float, default=30.0)

args = parser.parse_args()

# Set up torch and cuda
dtype = torch.float32
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_tensor_type('torch.cuda.FloatTensor' if torch.cuda.is_available() else 'torch.FloatTensor')

# Set random seeds
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

if args.full_dataset:
    min_id, max_id = 1, 24  # Kodak dataset runs from kodim01.png to kodim24.png
else:
    min_id, max_id = args.image_id, args.image_id

# Dictionary to register mean values (both full precision and half precision)
results = {'fp_bpp': [], 'hp_bpp': [], 'fp_psnr': [], 'hp_psnr': []}

# Create directory to store experiments
if not os.path.exists(args.logdir):
    os.makedirs(args.logdir)

# Fit images
for i in range(min_id, max_id + 1):
    print(f'Image {i}')

    # Load image
    img = imageio.imread(f"/workspace/eunho/BTech_Dataset_transformed/01/train/ok/0000.bmp")
    img = transforms.ToTensor()(img).float().to(device, dtype)
    print(img.size())
    # Setup model
    func_rep = Siren(
        dim_in=2,
        dim_hidden=args.layer_size,
        dim_out=6,
        num_layers=args.num_layers,
        final_activation=torch.nn.Identity(),
        w0_initial=args.w0_initial,
        w0=args.w0
    ).to(device)

    # Set up training
    trainer = Trainer(func_rep, lr=args.learning_rate)
    coordinates, features = util.to_coordinates_and_features(img)
    coordinates, features = coordinates.to(device, dtype), features.to(device, dtype)

    # Calculate model size. Divide by 8000 to go from bits to kB
    model_size = util.model_size_in_bits(func_rep) / 8000.
    print(f'Model size: {model_size:.1f}kB')
    fp_bpp = util.bpp(model=func_rep, image=img)
    print(f'Full precision bpp: {fp_bpp:.2f}')

    # Train model in full precision
    pre =trainer.train(coordinates, features, num_iters=args.num_iters)
    print("----------------",pre.size(),"-------------------")
    print(f'Best training psnr: {trainer.best_vals["psnr"]:.2f}')

    # Log full precision results
    results['fp_bpp'].append(fp_bpp)
    results['fp_psnr'].append(trainer.best_vals['psnr'])

    # Save best model
    torch.save(trainer.best_model, args.logdir + f'/best_model_{i}.pt')

    # Update current model to be best model
    func_rep.load_state_dict(trainer.best_model)

    # Save full precision image reconstruction
    with torch.no_grad():
        img_recon = func_rep(coordinates).reshape(img.shape[1], img.shape[2], 3).permute(2, 0, 1)
        print(img_recon.size())
        save_image(torch.clamp(img_recon, 0, 1).to('cpu'), args.logdir + f'/1fp_reconstruction_{i}.png')
        print(img_recon.size())
    # Convert model and coordinates to half precision. Note that half precision
    # torch.sin is only implemented on GPU, so must use cuda
    if torch.cuda.is_available():
        func_rep = func_rep.half().to('cuda')
        coordinates = coordinates.half().to('cuda')

        # Calculate model size in half precision
        hp_bpp = util.bpp(model=func_rep, image=img)
        results['hp_bpp'].append(hp_bpp)
        print(f'Half precision bpp: {hp_bpp:.2f}')

        # Compute image reconstruction and PSNR
        with torch.no_grad():
            img_recon = func_rep(coordinates).reshape(img.shape[1], img.shape[2], 3).permute(2, 0, 1).float()
            hp_psnr = util.get_clamped_psnr(img_recon, img)
            print(torch.clamp(img_recon, 0, 1).to('cpu').size)
            save_image(torch.clamp(img_recon, 0, 1).to('cpu'), args.logdir + f'/1hp_reconstruction_{i}.png')
            print(f'Half precision psnr: {hp_psnr:.2f}')
            results['hp_psnr'].append(hp_psnr)
    else:
        results['hp_bpp'].append(fp_bpp)
        results['hp_psnr'].append(0.0)

    # Save logs for individual image
    with open(args.logdir + f'/logs{i}.json', 'w') as f:
        json.dump(trainer.logs, f)

    print('\n')

print('Full results:')
print(results)
with open(args.logdir + f'/results.json', 'w') as f:
    json.dump(results, f)

# Compute and save aggregated results
results_mean = {key: util.mean(results[key]) for key in results}
with open(args.logdir + f'/results_mean.json', 'w') as f:
    json.dump(results_mean, f)

print('Aggregate results:')
print(f'Full precision, bpp: {results_mean["fp_bpp"]:.2f}, psnr: {results_mean["fp_psnr"]:.2f}')
print(f'Half precision, bpp: {results_mean["hp_bpp"]:.2f}, psnr: {results_mean["hp_psnr"]:.2f}')


import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve
import torch.nn.functional as F
#image load

thres = 0.5

# 원본 이미지 로드 및 텐서로 변환
img = imageio.imread(f"/workspace/eunho/BTech_Dataset_transformed/01/test/ko/0000.bmp")
img = transforms.ToTensor()(img).float().to(device, dtype)

# 이미지 재구성
with torch.no_grad():
    img_recon = func_rep(coordinates).reshape(img.shape[1], img.shape[2], 3).permute(2, 0, 1).to(device)
    img_recon_cpu = img_recon.cpu().detach().numpy()

# 손실 계산
loss = F.mse_loss(img_recon, img, reduction='mean')
mask_score_t = []
normalised_score_t = []
## PRO Score
# 그라운드 트루스 마스크 로드 및 텐서로 변환
mask_score_t.append(j.squeeze(0).squeeze(0).cpu().numpy()) # Storing all masks
m = torch.nn.UpsamplingBilinear2d((512,512))
norm_score = loss[-1].reshape(-1,1,512//patch_size,512//patch_size)
score_map = m(torch.tensor(norm_score))
score_map = Filter(score_map , type =0) 

                   
normalised_score_t.append(score_map)

# 재구성된 이미지에 임계값 적용
img_recon_thresholded = (img_recon_cpu > thres).astype(np.float32)

# 손실을 사용하여 PRO 점수 계산
scores = np.asarray(loss.cpu()).flatten()
masks = mask_cpu.flatten()
PRO_score = roc_auc_score(masks, scores)
print("PRO Score:", PRO_score)
    
## Image Anomaly Classification Score (AUC)
#roc_data = np.concatenate((t_loss_all_normal, t_loss_all_anomaly))
#roc_targets = np.concatenate((np.zeros(len(t_loss_all_normal)), np.ones(len(t_loss_all_anomaly))))
#AUC_Score_total = roc_auc_score(roc_targets, roc_data)
    
# AUC Precision Recall Curve
#precision, recall, thres = precision_recall_curve(roc_targets, roc_data)
#AUC_PR = auc(recall, precision)

    
#return PRO_score, AUC_Score_total, AUC_PR



