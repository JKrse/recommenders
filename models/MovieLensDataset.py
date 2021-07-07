# ========================================================================

import torch.utils.data
import pandas as pd
import os

# ========================================================================

class MovieLensDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path):

        ratings = pd.read_csv(f"{dataset_path}/ratings.csv")
        movies  = pd.read_csv(f"{dataset_path}/movies.csv")
        ratings = ratings.merge(movies, on="movieId", how="left")

        self.items = ratings.loc[:, ratings.columns != "rating"]
        
        self.target = ratings["rating"]
        self.target_binary = self.__binary_target_movieLens(ratings["rating"].copy())
        

    def __binary_target_movieLens(self, target):
        """
        Preprocess the ratings into negative and positive samples
        Args:
            target ([int]): ratings
        Returns:
            [int]: binary ratings (0 or 1)
        """
        target[target <= 3] = 0  # ratings less than or equal to 3 classified as 0
        target[target > 3] = 1  # ratings bigger than 3 classified as 1
        return target