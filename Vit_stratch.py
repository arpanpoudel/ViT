
import matplotlib.pyplot as plt
import torch
import torchvision

from torch import nn
from torchvision import transforms
from torchinfo import summary

# Create a class which subclasses nn.Module
class PatchEmbedding(nn.Module):
    """Turns a 2D input image into a 1D sequence learnable embedding vector.
    
    Args:
        in_channels (int): Number of color channels for the input images. Defaults to 3.
        patch_size (int): Size of patches to convert input image into. Defaults to 16.
        embedding_dim (int): Size of embedding to turn image into. Defaults to 768.
    """ 
    # 2. Initialize the class with appropriate variables
    def __init__(self,in_channels=3, patch_size=16, embedding_dim=768):
        
        super().__init__()
        
        #create a conv2d layer to convert image into patches
        self.patcher=nn.Conv2d(in_channels=in_channels,
                              out_channels=embedding_dim,
                              kernel_size=patch_size,
                              stride=patch_size,
                              padding=0)
        #create a flatten layer to flatten the patch feature maps into a single dimensions
        self.flatten=nn.Flatten(start_dim=2,end_dim=3)
        
        #set the patch size
        self.patch_size=patch_size
        
    #define a forward model
    def forward(self,x):
         # Create assertion to check that inputs are the correct shape
        image_resolution = x.shape[-1]
        assert image_resolution % self.patch_size == 0, f"Input image size must be divisble by patch size, image shape: {image_resolution}, patch size: {patch_size}"
        # Perform the forward pass
        x_patched = self.patcher(x)
        x_flattened = self.flatten(x_patched) 
        
        # 6. Make sure the output shape has the right order 
        return x_flattened.permute(0, 2, 1) # adjust so the embedding is on the final dimension [batch_size, P^2•C, N] -> [batch_size, N, P^2•C]

class MultiHeadAttention(nn.Module):
    """
    Creates a multi-head self-attention block ("MSA block" for short)
    """
    def __init__(self,embed_dim=768,num_heads=12,attn_dropout:float=0):
        super().__init__()
        self.embed_dim=embed_dim
        self.num_heads=num_heads
        
        #create a Layer norm
        self.layer_norm=nn.LayerNorm(self.embed_dim)
        
        # Create the Multi-Head Attention (MSA) layer
        self.multihead_attn = nn.MultiheadAttention(embed_dim=embed_dim,
                                                    num_heads=num_heads,
                                                    dropout=attn_dropout,
                                                    batch_first=True)
    # 5. Create a forward() method to pass the data throguh the layers
    def forward(self, x):
        x = self.layer_norm(x)
        attn_output, _ = self.multihead_attn(query=x, # query embeddings 
                                             key=x, # key embeddings
                                             value=x, # value embeddings
                                             need_weights=False) # do we need the weights or just the layer outputs?
        return attn_output

class MLP(nn.Module):
    '''
    MLP block that takes input from the MSA block and passes through the linear layers
    '''
    def __init__(self,embedding_dim=768,mlp_size=3072, dropout=0.1):
        
        super().__init__()
        # define a norm layer
        self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim)
        
        # define a mlp block
        self.mlp=nn.Sequential(nn.Linear(in_features=embedding_dim,out_features=mlp_size),
                              nn.GELU(),
                              nn.Dropout(p=dropout),
                              nn.Linear(in_features=mlp_size,out_features=embedding_dim),
                              nn.GELU(),
                              nn.Dropout(p=dropout))
        
        
    # define the forward method
    def forward(self,x):
        x=self.layer_norm(x)
        out=self.mlp(x)
        return out

class TransformerEncoder(nn.Module):
    """ Class to perform the  MLA, MLP with skip connection"""
    def __init__(self, embedding_dim=768, mlp_size=3072, dropout=0.1,num_heads=12):
        
        super().__init__()
        
        #define the MHSA block
        self.MSA_block=MultiHeadAttention(embed_dim=embedding_dim)
        
        # define the MLP block
        self.MLP_block=MLP(embedding_dim=embedding_dim,mlp_size=mlp_size,dropout=dropout)
        
    def forward(self,x):
        msa_out=self.MSA_block(x)
        mlp_in=msa_out+x
        mlp_out=self.MLP_block(mlp_in)
        out=mlp_in+mlp_out
        return out
