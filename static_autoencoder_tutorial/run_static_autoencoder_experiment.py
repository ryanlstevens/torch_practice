import sys; sys.path.insert(0, '/Users/ryanstevens/Documents/github/torch_practice/')
from src.helpers import data
from src.helpers import models, training
import torch.nn as nn
import torch
from importlib import reload
from pathlib import Path
from loguru import logger

path_to_root = Path(__file__).parent

if __name__ == '__main__':

    # Set parameters
    INPUT_DIM=28**2
    LOSS_FN=nn.CrossEntropyLoss()
    EPOCHS=200
    LR=1e-4

    losses=[]
    for batch_size in [
        4, 
        16, 
        32,
        64
    ]:
        
        # Load dataset 
        train, val, test = data.load_and_preprocess_data(debug=False, batch_size=batch_size)

        for latent_dim, compression_per_layer in [
            (1, 16), 
            (64, 2), 
            (128, 2),
            (256, 2), 
            (28*28, 2)
        ]:
            
            # Create path to save best model 
            path_to_early_stopping_folder=(
                path_to_root / 
                'data' / 
                'models' / 
                'static' /
                f'latentDim={latent_dim}_batchSize={batch_size}'
            )
            path_to_early_stopping_folder.mkdir(parents=True, exist_ok=True)

            logger.info(f"Path to early stopping folder: f{path_to_early_stopping_folder}")

            # Create encoder-decoder
            encoder_decoder_model = models.StaticEncoderDecoder(
                input_dim=INPUT_DIM,
                latent_dim=latent_dim,
                head_layer=nn.Sigmoid(),
                compression_per_layer=compression_per_layer,
            )

            # Initialize early stopping
            early_stopping=training.EarlyStopping(
                patience=5, 
                verbose=True, 
                path=path_to_early_stopping_folder
            )

            # Initialize model training
            model_trainer=training.ModelTrainer(
                model=encoder_decoder_model,
                optimizer=torch.optim.Adam(encoder_decoder_model.parameters(), lr=LR),
                loss_fn=LOSS_FN,
                train_loader=train,
                val_loader=val,
                epochs=EPOCHS,
                early_stopping=early_stopping,
            )

            # Train model
            model_trainer.train()

            # Persist training and validation losses
            losses.append(
                {
                    'train_losses': model_trainer.train_losses,
                    'val_losses': model_trainer.val_losses,
                    'test_losses': [LOSS_FN(model_trainer.best_model(X), X) for _, (X, _) in enumerate(test)],
                    'latent_dim': latent_dim
                }
            )