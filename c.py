from torchvision import transforms
import torch
from torchvision.utils import save_image
import imageio
import argparse
import os
from siren import Siren
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
    dim_out=3,
    num_layers=args.num_layers,
    final_activation=torch.nn.Identity(),
    w0_initial=args.w0_initial,
    w0=args.w0
).to(device)
trainer = Trainer(func_rep, lr=args.learning_rate)


model_path = args.logdir +'/best_model_1.pt'
load_state= torch.load(model_path)
func_rep.load_state_dict(load_state)


with torch.no_grad():
    img = imageio.imread("/workspace/eunho/BTech_Dataset_transformed/01/test/ko/0000.bmp")
    img = transforms.ToTensor()(img).float().to(device, dtype)
    c, f = util.to_coordinates_and_features(img)
    c, f = c.to(device, dtype), f.to(device, dtype)
    img_recon = func_rep(c).reshape(img.shape[1], img.shape[2], 3).permute(2, 0, 1)
    save_image(torch.clamp(img_recon, 0, 1).to('cpu'), args.logdir+ f'/test_re1.png')
    aaa = img - img_recon
    save_image(torch.clamp(aaa, 0, 1).to('cpu'),args.logdir + f'/test1.png')
    thread = 0.5#torch.mean(aaa)
    #ano_p = 평균 + 3 * 표준편차
    #ano_n = 평균 - 3 * 표준편차
    thresholded_img = torch.where(aaa > thread, torch.tensor(1.0), torch.tensor(0.0))
    #mask = np.where(ano_p > aaa > ano_n, 1., 0.)
    print(thresholded_img.size())
    save_image(torch.clamp(thresholded_img, 0, 1).to('cpu'), args.logdir + f'/test_thresholded.png')
