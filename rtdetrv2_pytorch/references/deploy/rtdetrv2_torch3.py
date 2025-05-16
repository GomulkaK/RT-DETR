import os
import json
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image, ImageDraw
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils
from src.core.yaml_config import YAMLConfig
import pandas as pd

def draw(images, labels, boxes, scores, thrh=0.6):
    for i, im in enumerate(images):
        draw = ImageDraw.Draw(im)

        scr = scores
        lab = labels
        box = boxes
        scrs = scores

        for j, b in enumerate(box):
            draw.rectangle(list(b), outline='red')
            draw.text((b[0], b[1]), text=f"{lab[j].item()} {round(scrs[j].item(), 2)}", fill='blue')

        im.save(f'results_{i}.jpg')


def create_coco_annotations(coco_annotations,images, labels, boxes, scores, image_file,image_id_start=1):
    annotation_id = len(coco_annotations["annotations"]) + 1  # Utrzymanie ciągłości ID anotacji
    image_id = image_id_start

    for i, (im, label, box, score) in enumerate(zip(images, labels, boxes, scores)):
        image_info = {
            "id": image_id,
            "file_name": f"{image_file}",
            "width": im.size[0],
            "height": im.size[1],
        }
        coco_annotations["images"].append(image_info)

        for j, b in enumerate(box):
                annotation_info = {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": label[j].item()+1,
                    "bbox": [b[0].item(), b[1].item(), b[2].item() - b[0].item(), b[3].item() - b[1].item()],
                    "score": score[j].item(),
                    "area": (b[2].item() - b[0].item()) * (b[3].item() - b[1].item()),
                    "iscrowd": 0
                }
                coco_annotations["annotations"].append(annotation_info)
                annotation_id += 1

        image_id += 1

    return coco_annotations


def main(config, resume, im_folder, model_name,device='gpu'):
    cfg = YAMLConfig(config, resume=resume)

    if resume:
        checkpoint = torch.load(resume, map_location='cpu')
        print(checkpoint.keys())  # Wyświetl klucze w pliku kontrolnym
        if 'ema' in checkpoint:
            state = checkpoint['ema']['module']
        else:
            state = checkpoint['model']
        filtered_state = {
            k: v for k, v in state.items()
            if k in cfg.model.state_dict() and cfg.model.state_dict()[k].shape == v.shape
        }
        cfg.model.load_state_dict(filtered_state, strict=False)
    else:
        raise AttributeError('Only support resume to load model.state_dict by now.')

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()
        def forward(self, images, orig_target_sizes, score_threshold=0.01):
            outputs = self.model(images)
            outputs = self.postprocessor(outputs, orig_target_sizes)
            labels, boxes, scores = outputs
            mask = scores > score_threshold
            labels = labels[mask]
            boxes = boxes[mask]
            scores = scores[mask]
            return labels, boxes, scores

    model = Model().to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Liczba parametrów modelu: {total_params}")
    coco_annotations = {
        "info": {
            "description": "Converted YOLO/Mask to COCO dataset",
            "version": "1.0",
            "year": 2024,
            "contributor": "",
            "date_created": "2024-11-09"
        },
        "licenses": [],
        "images": [],
        "annotations": [],
            "categories": [
        {
            "id": 1,
            "name": "person",
            "supercategory": "person"
        }
    ]
    }

    image_files = [f for f in os.listdir(im_folder) if f.endswith(('.png'))]
    for i, image_file in enumerate(image_files):  # Limitujemy do 500 obrazów
        print(i)
        im_path = os.path.join(im_folder, image_file)
        im_pil = Image.open(im_path).convert('RGB')
        w, h = im_pil.size
        orig_size = torch.tensor([w, h])[None].to(device)

        transforms = T.Compose([T.Resize((640, 640)), T.ToTensor()])
        im_data = transforms(im_pil)[None].to(device)

        labels, boxes, scores = model(im_data, orig_size, score_threshold=0.1)
        draw([im_pil], labels, boxes, scores)
        coco_annotations = create_coco_annotations(coco_annotations,[im_pil], [labels], [boxes], [scores], image_file, image_id_start=i+1)
        #print(coco_annotations)
    with open(f'detection_{model_name}.json', 'w') as json_file:
        json.dump(coco_annotations, json_file)

    # Ładowanie pliku COCO z wynikami
    coco_gt = COCO(r"D:\Detection\Datasets\Custom\8020_seq\test.json")  # Zakładając, że masz plik GT w formacie COCO
    with open(f'detection_{model_name}.json', 'r') as f:
        data_det = json.load(f)
    coco_dt = coco_gt.loadRes(data_det['annotations'])

    # Obliczanie metryk
    coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # Dane do DataFrame
    eval_data = {
        "Metric": ["AP", "AP50", "AP75", "AR@1", "AR@10", "AR@100"],
        "Value": [
            coco_eval.stats[0],
            coco_eval.stats[1],
            coco_eval.stats[2],
            coco_eval.stats[6],
            coco_eval.stats[7],
            coco_eval.stats[8],
        ],
    }
    # Tworzenie DataFrame
    df = pd.DataFrame(eval_data)
    # Zapis do CSV
    df.to_csv(f'eval_{model_name}.csv', index=False)
if __name__ == '__main__':
    config_path = r"C:\Users\Kamil\PycharmProjects\people_detection_310\RT-DETR\rtdetrv2_pytorch\configs\rtdetrv2\efficient.yml"
    resume_path = r"D:\Pobrane\OneDrive_1_11.12.2024\efficientnet_50_346\checkpoint0019.pth"
    model_name = "conv_20"
    im_folder = r"D:\Detection\Datasets\Custom\images_test" # Folder z obrazami
    device = "cuda"
    main(config=config_path, resume=resume_path, im_folder=im_folder, model_name=model_name, device=device)
