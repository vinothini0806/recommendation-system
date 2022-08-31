import torch
from torch.autograd import Variable

# Function to save the model
def saveModel(model):
    path = "./myFirstModel.pth"
    torch.save(model.state_dict(), path)

# Function to test the model with the test dataset 
def test(test_loader,model,criterion,device):
    
    model.eval()
    total = 0.0
    running_loss = 0.0
    
    with torch.no_grad():
        for data in test_loader:
            userId, movieId,rating = data['userId'], data['movieId'], data['rating']
            userId, movieId,rating = userId.to(device), movieId.to(device),rating.to(device)
            # run the model on the test set to predict labels
            outputs = model(userId, movieId)
            # the label with the highest energy will be our prediction
            _, predicted = torch.max(outputs.data, 1)
            total += rating.size(0)
            # compute the loss based on model output and real labels
            loss = criterion(outputs, rating)
            running_loss += loss.item()     # extract the loss value
    

    return(running_loss)


# Training function. 
def train(train_loader, model, criterion,optimizer, device):
    
        total = 0.0
        running_loss = 0.0
        
        for i, data in enumerate(train_loader, 0):
            
            # get the inputs
            userId = Variable(data['userId'].to(device))
            movieId = Variable(data['movieId'].to(device))
            rating = Variable(data['rating'].to(device))

            # zero the parameter gradients
            optimizer.zero_grad()
            # predict 
            outputs = model(userId, movieId)
            # compute the loss based on model output and real labels
            loss = criterion(outputs, rating)
            # backpropagate the loss
            loss.backward()
            # adjust parameters based on the calculated gradients
            optimizer.step()

           
            running_loss += loss.item()     # extract the loss value
            total += rating.size(0)

        
        return(running_loss)