from multiprocessing import Pool
import os.path
import time
from itertools import repeat
import torchmetrics
import matplotlib.pyplot as plt
import cv2
import torch
from torch import nn, tensor
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import DataLoader

from dataset.sp_gdph_dataset import GDPHSuperpixelsDataset
from loss.fl import focal_loss

from models.gcn import GraphNet
import numpy as np
import torch.nn.functional as F
from loss_schedule import loss_schedule
from models.graph_transformer import Mlp
from models.medical_transformer import GraphTransformer
from models.transformer import Encoder
from utils.utils_single import cluster_loss
gdph_root = "/mnt/s3/lhm/GDPH/"
gdph_patch_root = "/mnt/s3/lhm/GDPH/patch/"
writer = SummaryWriter('./log')
dataset = GDPHSuperpixelsDataset(root='/mnt/s3/lhm/GDPH/pre')
loader = DataLoader(dataset, batch_size=1, shuffle=True)
device = torch.device('cuda:1')
weight = tensor([1, 0.8]).to(device)
criterion = nn.CrossEntropyLoss()
# criterion = focal_loss
clsfr = Mlp(in_features=256, hidden_features=128, out_features=13).to(device)

graph_embedding = GraphNet(c_in=768, hidden_size=256, nc=13).to(device)
# encoder = Encoder().to(device)
encoder = GraphTransformer(num_nodes=1024).to(device)
opt = torch.optim.SGD(
    [{"params": graph_embedding.parameters(), 'lr': 3e-3}, {"params": encoder.parameters(), 'lr': 1e-3},
     {"params": clsfr.parameters(), 'lr': 1e-3}])
scheduler = CosineAnnealingLR(opt, T_max=200, eta_min=0)

print("Loading model state dict")
graph_embedding.load_state_dict(torch.load('/home/lhm/medical-gnn/model_save/GDPH/graph_embedding.pth'))
encoder.load_state_dict(torch.load('/home/lhm/medical-gnn/model_save/GDPH/encoder.pth'))
clsfr.load_state_dict(torch.load('/home/lhm/medical-gnn/model_save/GDPH/clsfr.pth'))

print("model state dict Loaded")

for index, batch in enumerate(loader):
    batch.x = batch.x.to(device)

    batch.edge_index = batch.edge_index.to(device)
    code = batch.code[0]
    prefix = batch.prefix[0]
    node_num = torch.max(batch.edge_index) + 1
    segments = np.load("/mnt/s3/lhm/GDPH" + f"/seg_tissue/{code}/{prefix}.npy", allow_pickle=True)
    rebuild = np.zeros_like(segments, dtype=np.int32)
    if node_num > 1024:
        continue
    node_y = batch.node_y.long().to(device)
    logit, feat = graph_embedding(batch)
    # pred_all = F.softmax(logit, dim=1)

    '''
    transformer 处理
    '''
    if node_num < 1024:
        padding_vector = [False] * node_num + [True] * (1024 - node_num)
    else:
        padding_vector = [False] * node_num
    # 将这个行向量复制seq_length行，形成所需的mask矩阵
    padding_mask = torch.from_numpy(np.tile(padding_vector, (1024, 1))).to(device)
    feat = feat.unsqueeze(0)  # fake batch
    padding_mask = padding_mask.unsqueeze(0)
    # node_emb, *attn = encoder(feat, padding_mask)
    node_emb = encoder(feat, padding_mask)
    node_emb = node_emb.squeeze(0)
    pred_all = F.softmax(clsfr(node_emb), dim=1)
    pred = pred_all[:node_num, :]
    target = node_y[:node_num]
    for index, y in enumerate(target):
        rebuild[segments == index] = int(y)
    plt.subplot(1, 3, 1)
    plt.imshow(rebuild)
    img = cv2.imread(gdph_root + f"/patch/{code}/{prefix}.ori.jpg")
    plt.subplot(1, 3, 2)
    plt.imshow(img)
    gt = cv2.imread(gdph_patch_root + f"{code}/{prefix}.mask.png", cv2.IMREAD_GRAYSCALE)
    plt.subplot(1, 3, 3)
    plt.imshow(gt)
    plt.show()
    print(target)
