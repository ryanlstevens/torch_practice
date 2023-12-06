import os 
import torch 
from typing import Callable 
from torch import utils 
import torchvision.datasets as datasets
from torchvision import transforms as torch_transforms
from .general import generate_file_paths
from .mnist_helpers import flatten, generate_train_test_dataset
from loguru import logger as log 

def create_train_val_test_data(
    raw_train_data: torch.utils.data.Dataset,
    outlier_class: int=1,
    train_val_test_split: list[float] = [0.6, 0.2, 0.2]
):
    # Check train_val_test_split input, must be length 3
    if len(train_val_test_split)!=3:
        raise(ValueError(f'train_val_test_split must be length 3, current length is {len(train_val_test_split)}'))

    if sum(train_val_test_split) != 1.0:
        raise(ValueError(f'train_val_test_split should sum to 1.0, current sum is {sum(train_val_test_split)}'))
    
    # Generate inliers and outliers data
    (
        inliers_data,
        outliers_data
    ) = split_into_inlier_and_outlier_datasets(
        raw_train_data,
        outlier_class,
    )

    # Generate train-test split from inliers data only
    random_seed = torch.Generator().manual_seed(42)
    inliers_train, inliers_test = torch.utils.data.random_split(
        inliers_data,
        [sum(train_val_test_split[:2]), train_val_test_split[-1]],
        random_seed
    )

    # Train data is only inliers
    # Test data includes both inliers + outliers,
    # we combine the test dataste
    train_data, val_data = torch.utils.data.random_split(
        inliers_train,
        [
            train_val_test_split[0]/sum(train_val_test_split[:2]),
            train_val_test_split[1]/sum(train_val_test_split[:2])
        ],
        random_seed
    )
    test_data = torch.utils.data.ConcatDataset([
        inliers_test,
        outliers_data
    ])

    return(
        train_data,
        val_data,
        test_data
    )

def split_into_inlier_and_outlier_datasets(
    raw_train_data: torch.utils.data.Dataset,
    outlier_class: int=1,
):
    ix=-1
    outliers_ix=[]
    for _, label in raw_train_data:
        ix+=1
        if label == outlier_class:
            outliers_ix.append(ix)
            continue 

    inliers_ix = [iy for iy in range(ix) if iy not in outliers_ix]
    
    return(
        torch.utils.data.Subset(raw_train_data, inliers_ix),
        torch.utils.data.Subset(raw_train_data, outliers_ix)
    )

def load_and_preprocess_data(
    path_to_raw_mnist_data: str = None,
    outlier_class: int = 0,
    train_val_test_split: list[float] = [0.6, 0.2, 0.2],
    data_preprocessing: Callable=None,
    batch_size: int=32,
    debug: bool=False
):
    
    if path_to_raw_mnist_data is None: 
        _, path_to_raw_mnist_data, _ = generate_file_paths()

    # If no preprocessing passed, do "standard" preprocessing:
    #  -Normalize elements to be between 0 and 1
    #  -Vertically stack each column 
    if data_preprocessing is None:
        data_preprocessing=torch_transforms.Compose([
            torch_transforms.ToTensor(),
            flatten
        ])

    # Download mnist datasets
    mnist_train_set = datasets.MNIST(
        root=path_to_raw_mnist_data, 
        train=True, 
        download=True, 
        transform=data_preprocessing
    )

    mnist_test_set = datasets.MNIST(
        root=path_to_raw_mnist_data, 
        train=False, 
        download=True, 
        transform=data_preprocessing
    )

    # Concatenate train and test datasets
    mnist_data = utils.data.ConcatDataset((
        mnist_train_set,
        mnist_test_set
    ))

    # Form 3 datasets:
    #  Train and val dataset includes only inliers
    #  Test dataset includes both inliers and outliers
    train, val, test = create_train_val_test_data(
        mnist_data,
        outlier_class,
        train_val_test_split
    )

    if debug:
        train = torch.utils.data.Subset(
            train,
            list(range(1000))
        )
        val = torch.utils.data.Subset(
            val,
            list(range(3000))
        )
        test = torch.utils.data.Subset(
            test,
            list(range(1000))
        )
    else:
        train = torch.utils.data.DataLoader(
            train, 
            batch_size=batch_size,
            shuffle=True
        )
        val = torch.utils.data.DataLoader(
            val, 
            batch_size=batch_size,
        )
        test = torch.utils.data.DataLoader(
            test, 
        )


    return train, val, test 

