import torch
import torchvision.models as models
from collections import OrderedDict

# Załaduj model Swin Transformer Small (Swin-S) z pretrenowanymi wagami
model = models.swin_t(pretrained=True)
model.eval()  # Przełącz model na tryb ewaluacji

# Lista nazw warstw, które chcemy zwrócić
return_layers = ['patch_embed', 'layers.1', 'layers.2', 'layers.3']

# Losowy tensor odpowiadający obrazowi 3x224x224
input_tensor = torch.randn(1, 3, 224, 224)

# Funkcja do sprawdzania warstw i wybierania wybranych
def inspect_return_layers(model, input_tensor, return_layers):
    outputs = OrderedDict()

    # Pobierz wszystkie cechy zwracane przez model
    with torch.no_grad():
        features = model(input_tensor)

    # Mapujemy nazwy warstw na ich indeksy
    feature_names = ['patch_embed', 'layers.0', 'layers.1', 'layers.2', 'layers.3']
    layer_to_index = {name: idx for idx, name in enumerate(feature_names)}

    # Pobieramy wybrane warstwy na podstawie return_layers
    for layer_name in return_layers:
        if layer_name in layer_to_index:
            # Uzyskujemy cechy z odpowiednich warstw
            outputs[layer_name] = features[layer_to_index[layer_name]]
        else:
            print(f"Warstwa {layer_name} nie istnieje w modelu.")

    return outputs

# Wywołaj funkcję i wypisz wymiary wybranych warstw
layer_outputs = inspect_return_layers(model, input_tensor, return_layers)
for layer_name, feature in layer_outputs.items():
    print(f"{layer_name}: {feature.shape}")
