# recommenders
repo for recommender systems

**Note Apple's M1 does not allow the usage of the libary deepCTR.**


# Download datasets

## Download MIND dataset
Use the function in ```MIND_exploration.download_mind.py``` to download dataset (*"small_train", "small_dev", "large_train", "large_dev"*).

To load the MIND files use the functions in ```load_mind``` class (```MIND_exploration.load_mind.py```)

Example of the downloading and loading dataset can be seen in ```MIND_exploration.mind_explore.py```. 

Some of the code has been taken from [Microsoft](https://docs.microsoft.com/en-us/azure/open-datasets/dataset-microsoft-news?tabs=azureml-opendatasets). 

MIND dataset are also avaiblie [https://msnews.github.io/](https://msnews.github.io/)

## Download MovieLens-1M

No function has been implemented but the dataset can be downloaded using the following commands: 

``` 
wget https://files.grouplens.org/datasets/movielens/ml-latest-small.zip

unzip ml-latest-small.zip -d datasets/

rm ml-latest-small.zip

```

Thanks to [grouplens](https://grouplens.org/datasets/movielens/) for making datasets avaible. 