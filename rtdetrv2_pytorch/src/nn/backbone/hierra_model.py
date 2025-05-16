"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.

https://towardsdatascience.com/getting-started-with-pytorch-image-models-timm-a-practitioners-guide-4e77b4bf9055#0583
"""

import torch
from torchvision.models.feature_extraction import get_graph_node_names, create_feature_extractor

from .utils import IntermediateLayerGetter
from ...core import register



@register()
class TimmModel(torch.nn.Module):
    def __init__(self, \
                 name,
                 return_layers=None,
                 pretrained=True,
                 exportable=True,
                 features_only=True,
                 **kwargs) -> None:
        super().__init__()

        import timm
        # Tworzymy model z TIMM
        model = timm.create_model(
            name,
            pretrained=pretrained,
            exportable=exportable,
            features_only=features_only,
            **kwargs
        )

        # Pobieramy listę dostępnych nazw warstw w modelu
        available_layers = model.feature_info.module_name()
        print(f"Dostępne warstwy: {available_layers}")

        return_layers = ['layers.1', 'layers.2', 'layers.3']

        # Filtrujemy tylko istniejące warstwy
        self.return_layers = [layer for layer in return_layers if layer in available_layers]
        if not self.return_layers:
            raise ValueError(f"Żadne z warstw {return_layers} nie są dostępne w modelu.")

        # Tworzymy indeksy dla wybranych warstw
        return_idx = [available_layers.index(name) for name in self.return_layers]

        self.model = model
        self.return_idx = return_idx

        # Informacje o kanałach i stride dla wybranych warstw
        self.strides = [model.feature_info.reduction()[i] for i in return_idx]
        self.channels = [model.feature_info.channels()[i] for i in return_idx]

    def forward(self, x: torch.Tensor):
        # Pobieramy wszystkie cechy modelu
        features = self.model(x)

        # Wybieramy tylko te warstwy, które są w return_layers
        outputs = [features[i] for i in self.return_idx]
        return outputs


if __name__ == '__main__':

    model = TimmModel('swinv2_large_window12to16_192to256', pretrained=False, features_only=True, return_layers="layers_3 \
    SwinTransformerV2Stage")
    data = torch.rand(1, 3, 640, 640)
    outputs = model(data)

    for output in outputs:
        print(output.shape)

    """
    model:
        type: TimmModel
        name: resnet34
        return_layers: ['layer2', 'layer4']
    """
