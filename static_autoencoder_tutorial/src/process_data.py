import os 
import torch 
from torch import utils 
import torchvision.datasets as datasets
from torchvision import transforms as torch_transforms
from helpers import general, mnist_helpers
from loguru import logger as log 

def process_mnist_data(
    path_to_raw_mnist_data: str,
    path_to_processed_mnist_data: str
):

    data_preprocessing=torch_transforms.Compose([
        torch_transforms.ToTensor(),
        mnist_helpers.flatten
    ])

    log.info(f'Path to raw mnist folder: {path_to_raw_mnist_data}')

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
    mnist_data = utils.data.ConcatDataset([
        mnist_train_set,
        mnist_test_set
    ])

    # Generate series of outlier datasets 
    for OUTLIER_CLASS in range(0, 10):

        # Form 3 datasets:
        #  Train and val dataset includes only inliers
        #  Test dataset includes both inliers and outliers
        train, val, test = mnist_helpers.generate_train_test_dataset(
            mnist_data,
            OUTLIER_CLASS
        )

        # Save datasets to folders
        path_to_outlier_class_folders=os.path.join(
            path_to_processed_mnist_data,
            f'{OUTLIER_CLASS}'
        )
        log.info(f'Path to outlier class folders: {path_to_outlier_class_folders}')
        if not os.path.exists(path_to_outlier_class_folders):
            log.info(f'Path does not exist, creating it...')
            os.makedirs(path_to_outlier_class_folders)

        torch.save(
            train, 
            os.path.join(
                path_to_outlier_class_folders,
                'train.pt'
            )
        )
        torch.save(
            val, 
            os.path.join(
                path_to_outlier_class_folders,
                'val.pt'
            )
        )
        torch.save(
            test, 
            os.path.join(
                path_to_outlier_class_folders,
                'test.pt'
            )
        )

if __name__ == '__main__':
    # Get file paths
    (
        _,
        PATH_TO_RAW_MNIST_DATA,
        PATH_TO_PROCESSED_MNIST_DATA
    ) = general.generate_file_paths()

    # Process mnist data
    process_mnist_data(
        path_to_raw_mnist_data=PATH_TO_RAW_MNIST_DATA,
        path_to_processed_mnist_data=PATH_TO_PROCESSED_MNIST_DATA
    )