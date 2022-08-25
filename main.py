# import matplotlib.pyplot as plt
import numpy as np
from data import get_data
from model import Network
from loss import get_loss
from train import *
from torch.optim import Adam
# Function to show the images
# def imageshow(img):
#     img = img / 2 + 0.5     # unnormalize
#     npimg = img.numpy()
#     plt.imshow(np.transpose(npimg, (1, 2, 0)))
#     plt.show()


# Function to test the model with a batch of images and show the labels predictions
# def testBatch():
#     # get batch of images from the test DataLoader  
#     images, labels = next(iter(test_loader))

#     # show all images as one image grid
#     # imageshow(torchvision.utils.make_grid(images))
   
#     # Show the real labels on the screen 
#     print('Real labels: ', ' '.join('%5s' % classes[labels[j]] 
#                                for j in range(batch_size)))
  
#     # Let's see what if the model identifiers the  labels of those example
#     outputs = model(images)
    
#     # We got the probability for every 10 labels. The highest (max) probability should be correct label
#     _, predicted = torch.max(outputs, 1)
    
#     # Let's show the predicted labels on the screen to compare with the real ones
#     print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] 
#                               for j in range(batch_size)))

def main():
    train_loader,test_loader,classes = get_data()
    model = Network()
    criterion = get_loss()
    optimizer = Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    num_epochs = 5
    train(train_loader, test_loader,model, criterion,optimizer, num_epochs)

if __name__ == "__main__":
    main()
        
        