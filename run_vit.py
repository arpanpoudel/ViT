import argparse
from pathlib import Path
import torch
import torchvision.transforms as standard_transforms
import numpy as np
from modular.predictions import pred_and_plot_image

from PIL import Image
from Vit_Transfer import build_model
import os
import warnings
warnings.filterwarnings('ignore')

def get_args_parser():
    parser = argparse.ArgumentParser('Set params for the Vit', add_help=False)
    

    parser.add_argument('--weight_path', default='./pretrained_weights/Vit_weights_B16.pth',
                        help='path where the trained weights saved')

    parser.add_argument('--gpu_id', default=0, type=int, help='the gpu used for evaluation')
    parser.add_argument('--train', default= False, type=bool, help=' used to train the model')
    parser.add_argument('--disp', default= False, type=bool, help=' display the loss and accuracy curves')
    parser.add_argument('--pretrained', default= True, type=bool, help=' display the loss and accuracy curves')
    return parser

def main(args, debug=False):


    print(args)
    device=f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu'
    print(device)
    # get the P2PNet
    model = build_model(args)
    
    # load trained model
    if args.weight_path is not None:
        checkpoint = torch.load(args.weight_path, map_location='cpu')
        model.load_state_dict(checkpoint)

    img_path = "data/pizza_steak_sushi/04-pizza-dad.jpeg"
    class_names=sorted([entry.name for entry in list(os.scandir("data/pizza_steak_sushi/train"))])
    
    result=pred_and_plot_image(model=model,image_path=img_path,class_names=class_names,device=device)
    print(f"Given image belongs to class {result}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser('VIT script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)   