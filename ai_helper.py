import os
import sys
import skimage.io
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from efficientnet_pytorch import model as enet
import io


tile_size = 256
image_size = 256
n_tiles = 36
batch_size = 8
num_workers = 4


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


def load_models(model_files):
    models = []
    for model_f in model_files:
        model_f = os.path.join(model_f)
        backbone = 'efficientnet-b0'
        model = enetv2(backbone, out_dim=5)
        model.load_state_dict(torch.load(model_f, map_location=lambda storage, loc: storage), strict=True)
        model.eval()
        model.to(device)
        models.append(model)
        print(f'{model_f} loaded!')
    return models


model_files = [
    'base_implementation'
]

models = load_models(model_files)


def get_tiles(img, mode=0):
    result = []
    h, w, c = img.shape
    pad_h = (tile_size - h % tile_size) % tile_size + ((tile_size * mode) // 2)
    pad_w = (tile_size - w % tile_size) % tile_size + ((tile_size * mode) // 2)

    img2 = np.pad(img, [[pad_h // 2, pad_h - pad_h // 2], [pad_w // 2, pad_w - pad_w // 2], [0, 0]],
                  constant_values=255)
    img3 = img2.reshape(
        img2.shape[0] // tile_size,
        tile_size,
        img2.shape[1] // tile_size,
        tile_size,
        3
    )

    img3 = img3.transpose(0, 2, 1, 3, 4).reshape(-1, tile_size, tile_size, 3)
    n_tiles_with_info = (img3.reshape(img3.shape[0], -1).sum(1) < tile_size ** 2 * 3 * 255).sum()
    if len(img) < n_tiles:
        img3 = np.pad(img3, [[0, N - len(img3)], [0, 0], [0, 0], [0, 0]], constant_values=255)
    idxs = np.argsort(img3.reshape(img3.shape[0], -1).sum(-1))[:n_tiles]
    img3 = img3[idxs]
    for i in range(len(img3)):
        result.append({'img': img3[i], 'idx': i})
    return result, n_tiles_with_info >= n_tiles


def getitem(img, tile_mode):
    sub_imgs = False

    tiff_file = img
    image = skimage.io.MultiImage(tiff_file)[0]
    tiles, OK = get_tiles(image, tile_mode)

    idxes = n_tiles
    idxes = np.asarray(idxes) + n_tiles if sub_imgs else idxes

    n_row_tiles = int(np.sqrt(n_tiles))
    images = np.zeros((image_size * n_row_tiles, image_size * n_row_tiles, 3))
    for h in range(n_row_tiles):
        for w in range(n_row_tiles):
            i = h * n_row_tiles + w

            if len(tiles) > idxes[i]:
                this_img = tiles[idxes[i]]['img']
            else:
                this_img = np.ones((image_size, image_size, 3)).astype(np.uint8) * 255
            this_img = 255 - this_img
            h1 = h * image_size
            w1 = w * image_size
            images[h1:h1 + image_size, w1:w1 + image_size] = this_img

    #         images = 255 - images
    images = images.astype(np.float32)
    images /= 255
    images = images.transpose(2, 0, 1)

    return torch.tensor(images)


def predict_label(im):
    data1 = getitem(im, 0)
    data2 = getitem(im, 2)
    LOGITS = []
    LOGITS2 = []
    with torch.no_grad():
        data1 = data1.to(device)
        logits = models[0](data1)
        LOGITS.append(logits)

        data = data.to(device)
        logits = models[0](data)
        LOGITS2.append(logits)

    LOGITS = (torch.cat(LOGITS).sigmoid().cpu() + torch.cat(LOGITS2).sigmoid().cpu()) / 2
    PREDS = LOGITS.sum(1).round().numpy()
    return PREDS


def classify_images(im):
    image = skimage.io.MultiImage(io.BytesIO(im))[0]
    pred, idx, probs = predict_label(image)
    s = 'Your submitted case has Prostate cancer of ISUP Grade ' + pred
    return s
