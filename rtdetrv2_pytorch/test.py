# import os
# import json
# import cv2  # Używane do wczytywania rozmiaru obrazów
# from pathlib import Path
# from datetime import datetime
#
# # Ścieżki do folderów z obrazami i adnotacjami
# images_folder = Path("C:/Users/Kamil/PycharmProjects/transferLearning/GOPRO_NEW/images/val")
# annotations_folder = Path("C:/Users/Kamil/PycharmProjects/transferLearning/GOPRO_NEW/labels/val")
# output_json = r"C:\Users\Kamil\PycharmProjects\transferLearning\GOPRO_NEW\labels\val.json"
#
# # Inicjalizacja struktury COCO
# coco_format = {
#     "info": {
#         "description": "Converted YOLO/Mask to COCO dataset",
#         "version": "1.0",
#         "year": 2024,
#         "contributor": "",
#         "date_created": datetime.now().strftime("%Y-%m-%d")
#     },
#     "licenses": [],
#     "images": [],
#     "annotations": [],
#     "categories": [
#         {"id": 1, "name": "person", "supercategory": "person"}
#     ]
# }
#
# annotation_id = 1  # unikalny ID dla każdej adnotacji
# image_id = 1  # unikalny ID dla każdego obrazu
#
# # Przetwarzanie każdego pliku .txt z adnotacjami
# for txt_file in annotations_folder.glob("*.txt"):
#     # Wczytanie nazwy pliku i rozmiaru obrazu
#     image_name = txt_file.stem + ".png"
#     image_path = images_folder / image_name
#     print(image_path)
#
#     # Wczytaj obraz, aby uzyskać jego rozmiar
#     image = cv2.imread(str(image_path))
#     height, width = image.shape[:2]
#
#     # Dodaj informacje o obrazie do sekcji "images"
#     coco_format["images"].append({
#         "id": image_id,
#         "file_name": image_name,
#         "height": height,
#         "width": width
#     })
#
#     # Wczytanie adnotacji z pliku tekstowego
#     with open(txt_file, "r") as f:
#         for line in f:
#             parts = line.strip().split()
#             class_id = int(parts[0]) + 1  # Przekształć klasę z YOLO (0 -> 1)
#             x_center, y_center, bbox_width, bbox_height = map(float, parts[1:])
#
#             # Przekształć znormalizowane współrzędne YOLO do formatu COCO
#             x = (x_center - bbox_width / 2) * width
#             y = (y_center - bbox_height / 2) * height
#             w = bbox_width * width
#             h = bbox_height * height
#             area = w * h
#
#             # Dodaj adnotację do sekcji "annotations"
#             coco_format["annotations"].append({
#                 "id": annotation_id,
#                 "image_id": image_id,
#                 "category_id": class_id,  # ID kategorii zaktualizowane
#                 "bbox": [x, y, w, h],
#                 "area": area,
#                 "iscrowd": 0
#             })
#             annotation_id += 1
#
#     # Inkrementuj ID obrazu po przetworzeniu pliku adnotacji
#     image_id += 1
#
# # Zapisz wynikowy plik JSON
# with open(output_json, "w") as outfile:
#     json.dump(coco_format, outfile, indent=4)


# -------------------------#
# import timm
#
# model = timm.create_model('swinv2_large_window12to16_192to256', pretrained=False, features_only=True)
# for name, module in model.named_children():
#     print(name, module)

# -------------------------#

import torch
import torch.nn as nn

from torch_intermediate_layer_getter import IntermediateLayerGetter as MidGetter


class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(2, 2)
        self.fc2 = nn.Linear(2, 2)
        self.nested = nn.Sequential(
            nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 3)),
            nn.Linear(3, 1),
        )
        self.interaction_idty = nn.Identity()  # Simple trick for operations not performed as modules

    def forward(self, x):
        x1 = self.fc1(x)
        x2 = self.fc2(x)

        interaction = x1 * x2
        self.interaction_idty(interaction)

        x_out = self.nested(interaction)

        return x_out


model = Model()
return_layers = {
    'fc2': 'fc2',
    'nested.0.1': 'nested',
    'interaction_idty': 'interaction',
}
mid_getter = MidGetter(model, return_layers=return_layers, keep_output=True)
mid_outputs, model_output = mid_getter(torch.randn(1, 2))

print(model_output)

print(mid_outputs)


# model_output is None if keep_ouput is False
# if keep_output is True the model_output contains the final model's output