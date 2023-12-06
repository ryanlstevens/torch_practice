from typing import Optional
import torch.nn as nn
import numpy as np
from loguru import logger

class StaticEncoderDecoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        use_dropout: bool = True,
        p_dropout: float = 0.5,
        head_layer: nn.Module = nn.Sigmoid(),
        compression_per_layer: int = 2,
    ):
        super().__init__()
        self.encoder = self._build_simple_nn(
            input_dim = input_dim,
            output_dim = latent_dim,
            use_dropout = use_dropout, 
            p_dropout = p_dropout,
            compression_per_layer = compression_per_layer
        )
        self.decoder = self._build_simple_nn(
            input_dim = latent_dim,
            output_dim = input_dim,
            use_dropout = use_dropout, 
            p_dropout = p_dropout,
            compression_per_layer = compression_per_layer,
            head_layer = head_layer
        )

    @staticmethod
    def _create_hidden_layer_dims(
        input_dim: int,
        output_dim: int,
        compression_per_layer: int=2
    ):
        if input_dim > output_dim:
            start_dim = output_dim
            end_dim = input_dim
        
        else:
            start_dim = input_dim
            end_dim = output_dim

        layers_dim=[]
        layers_dim.append(start_dim)
        while start_dim < end_dim:
            start_dim *= compression_per_layer
            layers_dim.append(min(start_dim, end_dim))
        
        if input_dim > output_dim:
            return zip(layers_dim[::-1][:-1], layers_dim[::-1][1:])
        
        else:
            return zip(layers_dim[:-1], layers_dim[1:])

    @staticmethod
    def _build_simple_nn(
        input_dim: int,
        output_dim: int,
        use_dropout: bool = True,
        p_dropout: float = 0.5,
        compression_per_layer: int = 2,
        head_layer: Optional[nn.Module()] = None
    ):
        layers=[]
        hidden_layer_dims = StaticEncoderDecoder._create_hidden_layer_dims(
            input_dim=input_dim,
            output_dim=output_dim,
            compression_per_layer=compression_per_layer
        )
        for layer_input_dim, layer_output_dim in hidden_layer_dims:
            layers.extend(
                [
                    nn.Linear(
                        layer_input_dim,
                        layer_output_dim
                    ),
                ]
            )

            if (
                (input_dim < output_dim) & 
                (layer_output_dim == output_dim) &
                (head_layer is not None)
            ):
                layers.append(head_layer)
            else:
                layers.append(nn.ReLU())
                if use_dropout:
                    layers.append(
                        nn.Dropout(
                            p_dropout
                        )
                    )

        return nn.Sequential(*layers)

    
    def forward(self, x):
        return self.decoder(
            self.encoder(x)
        )
    
class _StaticEncoderDecoder_BottleNeckLayer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        use_dropout: bool = True,
        p_dropout: float = 0.5,
        head_layer: nn.Module = nn.Sigmoid(),
        compression_per_layer: int = 2,
    ):
        super().__init__()
        self.encoder = self._build_simple_nn(
            input_dim = input_dim,
            output_dim = latent_dim,
            use_dropout = use_dropout, 
            p_dropout = p_dropout,
            compression_per_layer = compression_per_layer
        )
        self.decoder = self._build_simple_nn(
            input_dim = latent_dim,
            output_dim = input_dim,
            use_dropout = use_dropout, 
            p_dropout = p_dropout,
            compression_per_layer = compression_per_layer,
            head_layer = head_layer
        )

    @staticmethod
    def _create_hidden_layer_dims(
        input_dim: int,
        output_dim: int,
        compression_per_layer: int=2
    ):
        if input_dim > output_dim:
            start_dim = output_dim
            end_dim = input_dim
        
        else:
            start_dim = input_dim
            end_dim = output_dim

        layers_dim=[]
        layers_dim.append(start_dim)
        while start_dim < end_dim:
            start_dim *= compression_per_layer
            layers_dim.append(min(start_dim, end_dim))
        
        if input_dim > output_dim:
            return zip(layers_dim[::-1][:-1], layers_dim[::-1][1:])
        
        else:
            return zip(layers_dim[:-1], layers_dim[1:])

    @staticmethod
    def _build_simple_nn(
        input_dim: int,
        output_dim: int,
        use_dropout: bool = True,
        p_dropout: float = 0.5,
        compression_per_layer: int = 2,
        head_layer: Optional[nn.Module()] = None
    ):
        layers=[]
        hidden_layer_dims = StaticEncoderDecoder._create_hidden_layer_dims(
            input_dim=input_dim,
            output_dim=output_dim,
            compression_per_layer=compression_per_layer
        )
        for layer_input_dim, layer_output_dim in hidden_layer_dims:
            layers.extend(
                [
                    nn.Linear(
                        layer_input_dim,
                        layer_output_dim
                    ),
                ]
            )

            if (
                (input_dim < output_dim) & 
                (layer_output_dim == output_dim) &
                (head_layer is not None)
            ):
                layers.append(head_layer)
            else:
                layers.append(nn.ReLU())
                if use_dropout:
                    layers.append(
                        nn.Dropout(
                            p_dropout
                        )
                    )

        return nn.Sequential(*layers)

    
    def forward(self, x):
        return self.encoder(x)