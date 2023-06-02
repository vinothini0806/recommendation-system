

# Recommendation System using Neural Network

This project implements a recommendation system using a neural network model trained on the MovieLens dataset. The goal of the system is to predict movie ratings for users based on their historical preferences and provide personalized recommendations.

## Dataset

The MovieLens dataset is a widely used benchmark dataset in the field of recommender systems. It contains ratings and movie metadata from users on the MovieLens website. The dataset consists of several files, including:

- `ratings.csv`: Contains user ratings for movies.
- `movies.csv`: Contains movie metadata such as title and genres.

You can download the MovieLens dataset from the following link: [MovieLens Dataset](https://grouplens.org/datasets/movielens/)

## Installation

1. Clone the repository:

```
git clone https://github.com/vinothini0806/recommendation-system.git
cd recommendation-system
```

## Usage

1. Prepare the dataset:

- Download the MovieLens dataset from the provided link.
- Extract the dataset and place the relevant files (e.g., `ratings.csv` and `movies.csv`) in the `data/` directory of this project.

2. Train the neural network model:

```
python train.py
```

This script will load the dataset, preprocess it, and train the neural network model. The trained model will be saved in the `models/` directory.

3. Generate recommendations:

## Acknowledgements

- The MovieLens dataset is provided by the GroupLens Research Project at the University of Minnesota.
- The neural network implementation is based on the TensorFlow library.


# Recommendation System using Neural Network

This project implements a recommendation system using a neural network model trained on the MovieLens dataset. The goal of the system is to predict movie ratings for users based on their historical preferences and provide personalized recommendations.

## Dataset

The MovieLens dataset is a widely used benchmark dataset in the field of recommender systems. It contains ratings and movie metadata from users on the MovieLens website. The dataset consists of several files, including:

- `ratings.csv`: Contains user ratings for movies.
- `movies.csv`: Contains movie metadata such as title and genres.

You can download the MovieLens dataset from the following link: [MovieLens Dataset](https://grouplens.org/datasets/movielens/)


## Usage

1. Prepare the dataset:

- Download the MovieLens dataset from the provided link.
- Extract the dataset and place the relevant files (e.g., `ratings.csv` and `movies.csv`) in the `data/` directory of this project.

2. Train the neural network model:

```
python train.py
```




## Acknowledgements

- The MovieLens dataset is provided by the GroupLens Research Project at the University of Minnesota.
- The neural network implementation is based on the TensorFlow library.
