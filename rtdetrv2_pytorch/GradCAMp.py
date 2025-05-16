import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from torchvision import transforms
import glob
import yaml
from src.core.yaml_config import YAMLConfig

transform = transforms.Compose([
    transforms.Resize((640, 640)),  # Zmiana rozmiaru obrazu
    transforms.ToTensor(),          # Konwersja obrazu na tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalizacja
])
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Hook to get the activations and gradients of the target layer
        self.hook_handles = []
        self._register_hooks()

    def _register_hooks(self):
        # Hook for activations
        def save_activation(module, input, output):
            self.activations = output

        # Hook for gradients
        def save_gradient(module, input, output):
            self.gradients = output[0]

        self.hook_handles.append(self.target_layer.register_forward_hook(save_activation))
        self.hook_handles.append(self.target_layer.register_full_backward_hook(save_gradient))

    def remove_hooks(self):
        # Remove all hooks
        for handle in self.hook_handles:
            handle.remove()

    def generate_cam(self, input_image, target_class=None):
        self.model.eval()

        # Perform forward pass
        print(input_image.shape)
        input_image = input_image.to('cuda')

        _, _, H, W = input_image.shape

        orig_target_sizes = torch.tensor([H, W]).unsqueeze(0).to(input_image.device)

        output = self.model(input_image, orig_target_sizes=orig_target_sizes)

        if target_class is None:
            target_class = torch.tensor(2).to(device)

        # Rozpakuj wyjścia
        labels, boxes, scores = output

        # Filtruj detekcje dla danej klasy
        mask = labels == target_class
        class_scores = scores[mask]

        if class_scores.numel() == 0:
            raise ValueError(f"No detections found for class {target_class.item()}")

        # Znajdź indeks najlepszego score dla tej klasy
        best_idx = torch.argmax(class_scores)
        global_idx = torch.nonzero(mask, as_tuple=False)[best_idx]

        best_box = boxes[global_idx]
        best_label = labels[global_idx]# Zero out the gradients
        self.model.zero_grad()

        # Perform backward pass tylko dla najlepszego detection
        target_score = scores[global_idx]
        target_score.backward(retain_graph=True)

        # Get the gradients and activations
        gradients = self.gradients
        activations = self.activations

        # Pool the gradients across the channels (global average pooling)
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * activations, dim=1, keepdim=True)
        cam = F.relu(cam)  # Apply ReLU to the cam

        # Resize the CAM to the input image size
        cam = cam.squeeze().detach().cpu().numpy()
        cam = cv2.resize(cam, (input_image.shape[2], input_image.shape[3]))
        cam = np.maximum(cam, 0)  # Relu operation to remove negative values

        return cam, best_box.detach().cpu().numpy(), best_label.item()

    def overlay_cam_on_image(self, image, cam, alpha=0.5):
        # Normalize CAM and apply it on top of the image
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        cam = np.uint8(255 * cam)

        # Convert image to uint8
        image = image.permute(1, 2, 0).numpy()
        image = np.uint8(image * 255)

        heatmap = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(image, alpha, heatmap, 1 - alpha, 0)
        overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)  # Dodaj tę linię
        return overlay


# Wewnątrz Twojego kodu:

# Dodaj ten argument:
def main(config, resume, input_image_path, model_name, device='cuda'):
    cfg = YAMLConfig(config, resume=resume)

    if resume:
        checkpoint = torch.load(resume, map_location='cuda')
        print(checkpoint.keys())
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
            print("wynik", scores)
            mask = scores > score_threshold
            labels = labels[mask]
            boxes = boxes[mask]
            scores = scores[mask]

            return labels, boxes, scores

    model = Model().to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Liczba parametrów modelu: {total_params}")

    # Grad-CAM target layer

    backbone = model.model._modules['backbone']

    print(backbone._modules)

    #nested_model = backbone._modules['modules']
    #for name, layer in nested_model.named_children():
    #    print(f"Name: {name}, Layer: {layer}")
    target_layer = backbone._modules['res_layers'][3].blocks[1]
    grad_cam = GradCAM(model, target_layer)

    # Wczytanie pojedynczego obrazu
    image = Image.open(input_image_path).convert('RGB')
    input_image = transform(image).unsqueeze(0).to(device)
    print(input_image.shape)

    # Grad-CAM
    cam, box, label = grad_cam.generate_cam(input_image)
    overlay = grad_cam.overlay_cam_on_image(input_image[0].cpu(), cam)

    x1, y1, x2, y2 = map(int, box.flatten())
    overlay = cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 255, 255), 2)
    overlay = cv2.putText(overlay, f"Class {label}", (x1, y1 - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    # Pokaż samą heatmapę przed nałożeniem
    plt.figure(figsize=(6, 6))
    plt.imshow(cam, cmap='jet')
    plt.colorbar()
    plt.title("Grad-CAM Heatmap")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    # Wizualizacja
    plt.imshow(overlay)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    grad_cam.remove_hooks()


if __name__ == '__main__':
    config_path = r"/home/kgomulka/PycharmProjects/RT-DETR/rtdetrv2_pytorch/configs/rtdetrv2/weapons_r18.yml"
    resume_path = r"/home/kgomulka/PycharmProjects/RT-DETR/rtdetrv2_pytorch/output/weapons_r18/best.pth"
    model_name = "r18"
    input_image_path = r"/home/kgomulka/Pobrane/Weapons/test/1397_depositphotos_383756230-stock-photo-lockdownart-bandit-tools-loot-grenade_jpg.rf.cfe20cadac864c7ee5eaab4e0ffc48cf.jpg"  # Zmień na ścieżkę do konkretnego obrazu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    main(config=config_path, resume=resume_path, input_image_path=input_image_path, model_name=model_name, device=device)