import torch
import torchvision
from torch import nn
import matplotlib.pyplot as plt
from torchinfo import summary
from modular import engine,data_setup
from helper_functions import set_seeds,plot_loss_curves
import argparse
from modular import utils

def get_args_parser():
    parser = argparse.ArgumentParser('Set parameters for Vit', add_help=False)
    

    parser.add_argument('--gpu_id', default=0, type=int, help='the gpu used for evaluation')
    parser.add_argument('--train', default= False, type=bool, help=' used to train the model')
    parser.add_argument('--disp', default= False, type=bool, help=' display the loss and accuracy curves')
    parser.add_argument('--pretrained', default= True, type=bool, help=' display the loss and accuracy curves')
    return parser

class ViT:
    
    """
    Class that implements the transfer learning to create a pretrained Vit model to foodvision problem.
    """

    
    def get_model(self,class_names,device):
        
        self.weights=torchvision.models.ViT_B_16_Weights.DEFAULT
        model=torchvision.models.vit_b_16(weights=self.weights).to(device)
        self.transforms=self.weights.transforms()
        # Freeze all base layers in the "features" section of the model (the feature extractor) by setting requires_grad=False
        for param in model.parameters():
            param.requires_grad = False 
        
        model.heads = nn.Linear(in_features=768, out_features=len(class_names)).to(device)
        return  model
    
    def train(self,model,train_dataloader,test_dataloader,loss_fn,optimizer,epochs,device):
        
        result= engine.train(model= model,train_dataloader=train_dataloader,test_dataloader=test_dataloader,
                             optimizer=optimizer,
                             loss_fn=loss_fn,epochs=epochs,device=device)
        return  result
    
    def get_transforms(self):
        self.weights=torchvision.models.ViT_B_16_Weights.DEFAULT
        self.transforms=self.weights.transforms()
        return self.transforms

def build_model(args):
    # device agnostic code
    #print (args)
    device=f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu'
    #print(device)
    
    vit=ViT()
    #paths
    train_dir='data/pizza_steak_sushi/train'
    test_dir='data//pizza_steak_sushi/test'   
    train_dataloader_pretrained, test_dataloader_pretrained, class_names = data_setup.create_dataloaders(train_dir=train_dir,
                                                                                                     test_dir=test_dir,
                                                                                                     transform=vit.get_transforms(),
                                                                                                     batch_size=32)
    
    if args.train==True:
        #model setup
        model=vit.get_model(class_names=class_names,device=device)
        # Create optimizer and loss function
        optimizer = torch.optim.Adam(params=model.parameters(), 
                             lr=1e-3)
        loss_fn = torch.nn.CrossEntropyLoss()
        pretrained_vit_results = vit.train(model=model,
                                      train_dataloader=train_dataloader_pretrained,
                                      test_dataloader=test_dataloader_pretrained,
                                      optimizer=optimizer,
                                      loss_fn=loss_fn,
                                      epochs=10,
                                      device=device)
        utils.save_model(model=model,target_dir='pretrained_weights/',model_name='Vit_weights_B16.pth')
        if args.disp==True:
            plot_loss_curves(pretrained_vit_results)
        
    if args.pretrained==True:
        model=vit.get_model(class_names=class_names,device=device)
        model.load_state_dict(torch.load(f='pretrained_weights/Vit_weights_B16.pth'))
        return model
        
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Vit Script', parents=[get_args_parser()])
    args=parser.parse_args()
    model=build_model(args)
    
