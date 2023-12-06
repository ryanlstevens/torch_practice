import torch
from typing import Optional
import matplotlib.pyplot as plt 
import numpy as np 
import os 
from static_autoencoder_tutorial.src.helpers import general 

def generate_outlier_dataset(
    raw_train_data: torch.utils.data.Dataset,
    outlier_class: int=1,
    inlier_class_exclude: Optional[list[int]]=None
):
    ix=-1
    outliers_ix=[]
    inliers_ix=[]
    if inlier_class_exclude is not None:
        inlier_class_exclude=set(inlier_class_exclude)
    else:
        inlier_class_exclude=set([])
    for _, label in raw_train_data:
        ix+=1
        if label == outlier_class:
            outliers_ix.append(ix)
            continue 

        if label not in inlier_class_exclude:
            inliers_ix.append(ix)
    
    return(
        torch.utils.data.Subset(raw_train_data, inliers_ix),
        torch.utils.data.Subset(raw_train_data, outliers_ix)
    )

def generate_train_test_dataset(
    raw_train_data: torch.utils.data.Dataset,
    outlier_class: int=1,
    inlier_class_exclude: Optional[list[int]]=None,
    train_test_split: list[float] = [0.8, 0.2],
    train_val_split: list[float] = [0.8, 0.2]
):
    # Generate inliers and outliers data
    (
        inliers_data,
        outliers_data
    ) = generate_outlier_dataset(
        raw_train_data,
        outlier_class,
        inlier_class_exclude
    )

    # Generate train-test split from inliers data only
    random_seed = torch.Generator().manual_seed(42)
    inliers_train, inliers_test = torch.utils.data.random_split(
        inliers_data,
        train_test_split,
        random_seed
    )

    # Train data is only inliers
    # Test data includes both inliers + outliers,
    # we combine the test dataste
    train_data = inliers_train 
    train_data, val_data = torch.utils.data.random_split(
        inliers_train,
        train_val_split,
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

def plot_batch_sample(
    data: torch.utils.data.Dataset,
    batch_size= 6
):
    ncols=3
    nrows=(batch_size//ncols) + ((batch_size % ncols)!=0)
    fig, axes = plt.subplots(ncols=ncols, nrows=nrows)
    i, j = (0, 0)
    for ix in range(batch_size):
        img, label = data[ix]
        if len(img.shape)==3:
            img = np.squeeze(img, 0)
            
        ax=axes[i, j]
        ax.imshow(
            img,
        )
        ax.set_title(f'Target={label}')
        j+=1
        if j==(ncols):
            i+=1
            j=0
    
    plt.show()

def plot_batch_prediction(
    data: torch.utils.data.Dataset,
    model, 
    loss_fn,
    batch_size= 6
):
    ncols=2
    nrows=batch_size
    fig, axes = plt.subplots(ncols=ncols, nrows=nrows)
    for ix in range(batch_size):
        img, label = data[ix]
        pred = model(img)
        error = loss_fn(model(img), img)
        if len(img.shape)==3:
            img = np.squeeze(img, 0)
            pred = np.squeeze(pred.detach().numpy(), 0)
            
        ax1=axes[ix, 0]
        ax2=axes[ix, 1]
        ax1.imshow(
            img,
        )
        ax2.imshow(
            pred,
        )
        ax1.set_title(f'Target={label}\nTrue', fontsize=10)
        ax2.set_title(f'Target={label}\nError={error:2.5f}', fontsize=10)
       
    
    plt.show()

def flatten(tensor):
    """Convert from 2d to 1d tensor"""
    return torch.flatten(tensor)