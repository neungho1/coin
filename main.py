import argparse
import getpass
import imageio
import json
import os
import random
import torch
import util
from siren_copy import Siren
from torchvision import transforms
from torchvision.utils import save_image
from training import Trainer
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("-ld", "--logdir", help="Path to save logs", default=f"/tmp/{getpass.getuser()}")
parser.add_argument("-ni", "--num_iters", help="Number of iterations to train for", type=int, default=500)
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
    """
    img = imageio.imread("/workspace/eunho/BTech_Dataset_transformed/01/test/ok/0000.bmp")
    img = transforms.ToTensor()(img).float().to(device, dtype)
    img_ano = imageio.imread("/workspace/eunho/BTech_Dataset_transformed/01/test/ko/0000.bmp")
    img_ano = transforms.ToTensor()(img_ano).float().to(device, dtype)
    img_ano
    """
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
    img_list = []
    for j in range(100):
        if j < 100:
            num = j+100
            img = imageio.imread(f"/workspace/eunho/BTech_Dataset_transformed/01/train/ok/{num:04d}.bmp")
            img = transforms.ToTensor()(img).float().to(device, dtype)
            img_list.append(img)
    c_list = []
    f_list = []
    for j in img_list:
        coordinates, features = util.to_coordinates_and_features(j)
        coordinates, features = coordinates.to(device, dtype), features.to(device, dtype)
        c_list.append(coordinates)
        f_list.append(features)
    print(f_list[0].size())
    f_tensor = torch.stack(f_list)
    print(f_tensor.size())
    mean = torch.mean(f_tensor, dim=0)
    std = torch.std(f_tensor, dim=0)

    print('평균:', mean, mean.size())
    print('표준편차:', std, std.size())
    result = torch.cat((mean, std), dim=1)
    print('결과:', result.size())
    # Calculate model size. Divide by 8000 to go from bits to kB
    model_size = util.model_size_in_bits(func_rep) / 8000.
    print(f'Model size: {model_size:.1f}kB')
    fp_bpp = util.bpp(model=func_rep, image=img)
    print(f'Full precision bpp: {fp_bpp:.2f}')

    # Train model in full precision
    trainer.train(c_list, f_list, num_iters=args.num_iters)
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
        img = imageio.imread("/workspace/eunho/BTech_Dataset_transformed/01/test/ko/0000.bmp")
        img = transforms.ToTensor()(img).float().to(device, dtype)
        c, f = util.to_coordinates_and_features(img)
        c, f = c.to(device, dtype), f.to(device, dtype)
        print(img.size())
        img_recon= func_rep(c)
        img_recon_mean, img_recon_std = torch.split(img_recon, 3, dim=1)
        print(img_recon_mean.size(),img_recon_std.size())
        img_recon_mean = img_recon_mean.reshape(3, 1600, 1600)
        img_recon_std = img_recon_std.reshape(3, 1600, 1600)
        save_image(torch.clamp(img_recon_mean,0,1).to('cpu'), args.logdir + f'/test_re.png')
        aaa = img_recon_mean - img_recon_std
        save_image(torch.clamp(aaa, 0, 1).to('cpu'),args.logdir + f'/test.png')
        threshold = 0.5

        mask = (img_recon_mean - 3 * img_recon_std > img) | (img_recon_mean + 3 * img_recon_std < img)

        # Creating an empty tensor with the same shape as img, but with the same number of channels as img
        masked_img = torch.zeros_like(img[0:1, ...])

        # Filling the masked_img tensor with values based on the mask
        masked_img[0, mask[0, ...]] = 1.0  # You can adjust the value based on your requirement

        # Saving the masked_img'
        print(masked_img.size(),"-----")
        save_image(masked_img.to('cpu'), args.logdir + f'/masked_image.png')
        #mask = np.where(aaa > threshold, 1., 0.)
        #masked_img = torch.where(mask, torch.tensor(1.0), torch.tensor(0.0))
        #save_image(torch.clamp(masked_img, 0, 1).to('cpu'),args.logdir + f'/test_mask.png')
    # Convert model and coordinates to half precision. Note that half precision
    # torch.sin is only implemented on GPU, so must use cuda
    """
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
            save_image(torch.clamp(img_recon, 0, 1).to('cpu'), args.logdir + f'/hp_reconstruction_{i}.png')
            print(f'Half precision psnr: {hp_psnr:.2f}')
            results['hp_psnr'].append(hp_psnr)
    else:
        results['hp_bpp'].append(fp_bpp)
        results['hp_psnr'].append(0.0)
    """
    # Save logs for individual image
    with open(args.logdir + f'/logs_re.json', 'w') as f:
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

from sklearn.metrics import roc_auc_score
import numpy as np

# Assuming ground_truth is a binary tensor with the same shape as img
# Convert ground_truth to a 1D array
ground_truth_flat = imageio.imread("/workspace/eunho/BTech_Dataset_transformed/01/ground_truth/ko/0000.png")
ground_truth_flat = ground_truth_flat.flatten()

# Threshold the masked_img to create a binary prediction
thresholded_masked_img = (masked_img.to('cpu') > threshold).view(-1).numpy()
print(ground_truth_flat.shape,thresholded_masked_img.shape)
# Calculate AUROC
auroc = roc_auc_score(ground_truth_flat, thresholded_masked_img)

print(f"Segmentation AUROC: {auroc}")