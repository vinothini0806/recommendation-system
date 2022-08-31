import numpy as np
from data import get_data
from model import Network
from loss import get_loss
from train import *
from torch.optim import Adam
import wandb
import os

def main():
    os.environ['WANDB_API_KEY']='b935180e09d0a1aa25645ee7615514ec84c50ad5'
    os.environ['WANDB_MODE']='offline'
    wandb.init(project='cnn')
    
    train_loader,test_loader,num_users,num_movies = get_data()
    model = Network(num_users,num_movies)
    # Define your execution device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("The model will be running on", device, "device")
    # Convert model parameters and buffers to CPU or Cuda
    model.to(device)
    criterion = get_loss()
    optimizer = Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    num_epochs = 10
    
    
    wandb.config.update({'lr':0.001,'weight_decay':0.0001,'num_epochs':10})
    
    for epoch in range(num_epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        

        train_loss = train(train_loader,model, criterion,optimizer,device)
        test_loss = test(test_loader,model, criterion,device)

        print('epoch:%d train_loss: %.3f  test_loss: %.3f ' % (epoch + 1, train_loss,test_loss))

        
            
        wandb.log({
            'epoch':epoch,
            'train_loss':train_loss,
            'test_loss':test_loss
        })

if __name__ == "__main__":
    main()
        