import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F

# Define a convolution neural network
class Network(nn.module):
    def __init__(self,num_users,num_movies):
        super().__init__()
        self.user_embed = nn.Embedding(num_users,32,max_norm=True)
        self.movie_embed = nn.Embedding(num_movies,32,max_norm=True)
        self.out= nn.Linear(64,1)

    

    def forward(self, userId,movieId,rating=None):
        user_embeds = self.user_embed(userId)
        movie_embeds = self.movie_embed(movieId)
        output = torch.cat([user_embeds,movie_embeds], dim=1)
        output = self.out(output)
        
        return output

