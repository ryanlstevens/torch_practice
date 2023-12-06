
import pathlib
import os 
from typing import Tuple 

def generate_file_paths() -> Tuple[str, str, str]:
    PATH_TO_FILE_DIR=pathlib.Path(__file__).parent.resolve()
    PATH_TO_RAW_MNIST_DATA=os.path.join(
        PATH_TO_FILE_DIR,
        '..',
        '..',
        'data',
        'mnist',
        'raw'
    )
    PATH_TO_PROCESSED_MNIST_DATA=os.path.join(
        PATH_TO_FILE_DIR,
        '..',
        '..',
        'data',
        'mnist',
        'processed'
    )

    return(
        PATH_TO_FILE_DIR,
        PATH_TO_RAW_MNIST_DATA,
        PATH_TO_PROCESSED_MNIST_DATA
    )