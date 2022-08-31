import torch
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from sklearn import metrics, preprocessing
import pandas as pd
from sklearn.model_selection import train_test_split

class MovieDataset:
    def __init__(self, userId,movieId,rating):
        self.userId = userId
        self.movieId = movieId
        self.rating = rating

    def __len__(self):
        return len(self.userId)

    def __getitem__(self,item):
        # item is item index
        userId = self.userId[item]
        movieId = self.movieId[item]
        rating = self.rating[item]

        return{"userId":torch.tensor(userId,dtype=torch.long),
        "movieId":torch.tensor(movieId,dtype=torch.long),
        "rating":torch.tensor(rating,dtype=torch.float)}

def get_data():
    df = pd.read_csv("../../input/train-v2/train_v2.csv")
    df=df.reset_index(drop=True)

    lbl_user = preprocessing.LabelEncoder()
    lbl_movie= preprocessing.LabelEncoder()
    
    df.user = lbl_user.fit_transform(df.userId.values)
    df.movie = lbl_movie.fit_transform(df.movieId.values)

    df_train, df_valid = train_test_split(
        df,test_size=0.1, random_state=42, stratify=df.rating.values
    )

    train_dataset = MovieDataset(userId = df_train.userId.values,
    movieId = df_train.movieId.values,
    rating = df_train.rating.values)

    test_dataset = MovieDataset(userId = df_valid.userId.values,
    movieId = df_valid.movieId.values,
    rating = df_valid.rating.values)
    
    num_users = len(lbl_user.classes_)
    num_movies = len(lbl_movie.classes_)
    batch_size=128
    
    # Create a loader for the training set which will read the data within batch size and put into memory.
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    print("The number of data in a training set is: ", len(train_loader)*batch_size)


    # Create a loader for the test set which will read the data within batch size and put into memory. 
    # Note that each shuffle is set to false for the test loader.
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    print("The number of data in a test set is: ", len(test_loader)*batch_size)

    

    return train_loader,test_loader,num_users,num_movies