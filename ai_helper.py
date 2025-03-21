import os
import io
import torch
import torch.nn as nn
import skimage.io
import numpy as np
from efficientnet_pytorch import model as enet

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Constants
tile_size = 256
image_size = 256
n_tiles = 36
batch_size = 8
num_workers = 4


# EfficientNet Model Wrapper
class enetv2(nn.Module):
    def __init__(self, backbone, out_dim):
        super(enetv2, self).__init__()
        self.enet = enet.EfficientNet.from_name(backbone)
        self.myfc = nn.Linear(self.enet._fc.in_features, out_dim)
        self.enet._fc = nn.Identity()

    def extract(self, x):
        return self.enet(x)

    def forward(self, x):
        x = self.extract(x)
        x = self.myfc(x)
        return x


# Load Pretrained Models
def load_models(model_files):
    models = []
    for model_f in model_files:
        model_path = os.path.join(model_f)
        backbone = "efficientnet-b0"
        model = enetv2(backbone, out_dim=5)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        models.append(model)
        print(f"Loaded model: {model_f}")
    return models


model_files = ["base_implementation.pth"]  # Ensure correct file extension
models = load_models(model_files)


# Get Tiles from Image
def get_tiles(img, mode=0):
    result = []
    h, w, c = img.shape
    pad_h = (tile_size - h % tile_size) % tile_size + ((tile_size * mode) // 2)
    pad_w = (tile_size - w % tile_size) % tile_size + ((tile_size * mode) // 2)

    img_padded = np.pad(
        img, [[pad_h // 2, pad_h - pad_h // 2], [pad_w // 2, pad_w - pad_w // 2], [0, 0]],
        constant_values=255
    )

    img_reshaped = img_padded.reshape(
        img_padded.shape[0] // tile_size, tile_size,
        img_padded.shape[1] // tile_size, tile_size, 3
    ).transpose(0, 2, 1, 3, 4).reshape(-1, tile_size, tile_size, 3)

    n_tiles_with_info = (img_reshaped.reshape(img_reshaped.shape[0], -1).sum(1) < tile_size**2 * 3 * 255).sum()
    idxs = np.argsort(img_reshaped.reshape(img_reshaped.shape[0], -1).sum(-1))[:n_tiles]
    img_reshaped = img_reshaped[idxs]

    for i, tile in enumerate(img_reshaped):
        result.append({"img": tile, "idx": i})

    return result, n_tiles_with_info >= n_tiles


# Convert Image to Tensor
def getitem(img, tile_mode):
    tiles, OK = get_tiles(img, tile_mode)
    n_row_tiles = int(np.sqrt(n_tiles))
    images = np.zeros((image_size * n_row_tiles, image_size * n_row_tiles, 3), dtype=np.uint8)

    for h in range(n_row_tiles):
        for w in range(n_row_tiles):
            i = h * n_row_tiles + w
            if i < len(tiles):
                this_img = tiles[i]["img"]
            else:
                this_img = np.ones((image_size, image_size, 3), dtype=np.uint8) * 255  # White padding

            h1, w1 = h * image_size, w * image_size
            images[h1:h1 + image_size, w1:w1 + image_size] = this_img

    images = images.astype(np.float32) / 255  # Normalize
    images = images.transpose(2, 0, 1)  # Change channel order for PyTorch
    return torch.tensor(images).unsqueeze(0)  # Add batch dimension


# Predict Image Label
def predict_label(img):
    data1 = getitem(img, 0).to(device)
    data2 = getitem(img, 2).to(device)

    LOGITS1, LOGITS2 = [], []

    with torch.no_grad():
        logits1 = models[0](data1)
        logits2 = models[0](data2)

    LOGITS1.append(logits1)
    LOGITS2.append(logits2)

    LOGITS1 = torch.cat(LOGITS1).sigmoid().cpu()
    LOGITS2 = torch.cat(LOGITS2).sigmoid().cpu()
    FINAL_LOGITS = (LOGITS1 + LOGITS2) / 2

    pred = FINAL_LOGITS.sum(1).round().numpy()[0]  # Extract final prediction
    return int(pred)  # Convert NumPy to int


# Classify Uploaded Image
def classify_images(im):
    image = skimage.io.imread(io.BytesIO(im))  # Read from bytes
    pred = predict_label(image)
    return f"Your submitted case has Prostate cancer of ISUP Grade {pred}"
