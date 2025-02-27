import os
import argparse
import logging
import cv2
import torch
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import DataLoader
from torch_geometric.utils import add_self_loops

from dataset.sp_dataset import SuperpixelsDataset, SuperpixelsTestDataset
from loss.fl import focal_loss
from models.gcn import GCN, GCNSimple, GraphNet
from models.graph_sage import GraphSAGE
import numpy as np
from loss_schedule import loss_schedule
from models.graph_transformer import GraphTransformer
from utils.utils import _norm, cut_patch_stage_2, convert_numpy_img_to_superpixel_graph_2, read_points_from_xml, \
    generate_transformer_input_and_its_groups_id_sequence, get_nodes_group_composition, merge_t_out_and_k_out

# 配置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def parse_args():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description='Training script for HCC superpixel graphs')
    # 数据路径参数
    parser.add_argument('--slide_root', type=str, default='/path/to/your/slide',
                        help='Root directory for slide images')
    parser.add_argument('--dst_root', type=str, default='/path/to/your/dst/from/preprocessed',
                        help='Root directory for dataset')
    # 模型参数
    parser.add_argument('--num_node_features', type=int, default=512,
                        help='Number of node features')
    parser.add_argument('--activation', type=str, default='relu',
                        help='Activation function')
    parser.add_argument('--num_conv_layers', type=int, default=3,
                        help='Number of convolutional layers')
    parser.add_argument('--conv_size', type=int, default=128,
                        help='Convolution size')
    parser.add_argument('--pool_method', type=str, default='add',
                        help='Pooling method')
    parser.add_argument('--lin1_size', type=int, default=64,
                        help='Size of the first linear layer')
    parser.add_argument('--lin2_size', type=int, default=32,
                        help='Size of the second linear layer')
    parser.add_argument('--output_size', type=int, default=3,
                        help='Output size')
    # 训练参数
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size')
    parser.add_argument('--lr1', type=float, default=0.001,
                        help='Learning rate for model1')
    parser.add_argument('--lr2', type=float, default=0.001,
                        help='Learning rate for model2')
    parser.add_argument('--lr_transformer2', type=float, default=0.01,
                        help='Learning rate for graph_transformer_2')
    parser.add_argument('--T_max', type=int, default=200,
                        help='T_max for CosineAnnealingLR')
    parser.add_argument('--eta_min', type=float, default=0,
                        help='eta_min for CosineAnnealingLR')
    parser.add_argument('--K', type=int, default=2,
                        help='Number of selected indices')
    # 设备参数
    parser.add_argument('--device', type=str, default='cuda:1',
                        help='Device to use for training')
    # 模型保存路径
    parser.add_argument('--model_save_dir', type=str, default='model_save/',
                        help='Directory to save trained models')
    return parser.parse_args()


def train_stage1(model, opt, scheduler, loader, dst_root, device, criterion, epoch, train_acc1):
    """
    训练第一阶段
    """
    model.train()
    for index, batch in enumerate(loader):
        batch.x = batch.x.to(device)
        batch.pos = batch.pos * 64
        batch.edge_index = batch.edge_index.to(device)
        code = batch.code[0]
        seg_map_stage_1 = np.load(f'{dst_root}/seg/{code}.npy')
        pred, feat = model(batch)
        y = (batch.y > 1).long().to(device)
        loss = criterion(pred, y)
        pred_y = pred.argmax(dim=1)
        acc = [(((pred_y == y) & (y == cls)).float().sum() / (
                (y == cls).float().sum() + 1e-6)).item() // 0.0001 / 100 for cls in range(2)]
        logging.info(
            f'stage1 acc {acc}  with gt {y.sum()}/{y.shape[0]}  pred {pred_y.sum()}/{pred_y.shape[0]}')
        train_acc1.append(acc)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return train_acc1


def train_stage2(model, opt, scheduler, graph_transformer, opt_graph_transformer, loader, slide_root, dst_root, device,
                 criterion, epoch, train_acc2, K, loss_schedule):
    """
    训练第二阶段
    """
    model.train()
    graph_transformer.train()
    for index, batch in enumerate(loader):
        if loss_schedule[epoch][1][0] == 0:
            continue
        batch.x = batch.x.to(device)
        batch.pos = batch.pos * 64
        batch.edge_index = batch.edge_index.to(device)
        code = batch.code[0]
        # 第一阶段的预测和处理
        seg_map_stage_1 = np.load(f'{dst_root}/seg/{code}.npy')
        pred, feat = model(batch)
        y = (batch.y > 1).long().to(device)
        loss = criterion(pred, y)
        pred_y = pred.argmax(dim=1)
        acc = [(((pred_y == y) & (y == cls)).float().sum() / (
                (y == cls).float().sum() + 1e-6)).item() // 0.0001 / 100 for cls in range(2)]
        logging.info(
            f'stage1 acc {acc}  with gt {y.sum()}/{y.shape[0]}  pred {pred_y.sum()}/{pred_y.shape[0]}')

        # 选择正样本和负样本的中心
        front_bit_map = (batch.x.mean(dim=1).mean(dim=1) < 220) & (batch.x.mean(dim=1).mean(dim=1) > 0)
        front_idx = torch.arange(batch.x.shape[0]).to(device)[front_bit_map]
        pos_selected_idx_stage_2 = torch.sort(pred[front_bit_map][:, 1], descending=True)[1][:K]
        neg_selected_idx_stage_2 = torch.sort(pred[front_bit_map][:, 0], descending=True)[1][:K]
        pos_selected_idx_stage_2 = front_idx[pos_selected_idx_stage_2].cpu()
        neg_selected_idx_stage_2 = front_idx[neg_selected_idx_stage_2].cpu()
        pos_centers_stage_2 = batch.pos[pos_selected_idx_stage_2]
        neg_centers_stage_2 = batch.pos[neg_selected_idx_stage_2]

        # 裁剪正样本和负样本的补丁
        pos_patches_2 = cut_patch_stage_2(pos_centers_stage_2, slide_root + code + '.svs')
        neg_patches_2 = cut_patch_stage_2(neg_centers_stage_2, slide_root + code + '.svs')

        # 生成正样本和负样本的图和分割图
        polygonsX8 = read_points_from_xml(liver_name=f'{code}.xml', scale=8,
                                          xml_path=slide_root,
                                          dataset='HCC')
        pos_graphs_2 = []
        neg_graphs_2 = []
        pos_seg_maps_2 = []
        neg_seg_maps_2 = []
        for idx, patch in enumerate(pos_patches_2):
            g, seg_map = convert_numpy_img_to_superpixel_graph_2(patch, polygonsX8,
                                                                 pos_centers_stage_2[idx] - (1024 * 8 // 2),
                                                                 seg_map_stage_1)
            pos_graphs_2.append(g)
            pos_seg_maps_2.append(seg_map)
        for idx, patch in enumerate(neg_patches_2):
            g, seg_map = convert_numpy_img_to_superpixel_graph_2(patch, polygonsX8,
                                                                 neg_centers_stage_2[idx] - (1024 * 8 // 2),
                                                                 seg_map_stage_1)
            neg_graphs_2.append(g)
            neg_seg_maps_2.append(seg_map)

        # 训练正样本图
        accumulate_steps = len(pos_graphs_2) + len(neg_graphs_2)
        for graph in pos_graphs_2:
            graph.x = graph.x.to(device)
            graph.edge_index = graph.edge_index.to(device)
            pred, feat = model(graph)
            tx, group_ids = generate_transformer_input_and_its_groups_id_sequence(graph.group_id, feat)
            t_out, t_attn = graph_transformer(tx)
            cls_tokens = t_out[:16]
            k_out = get_nodes_group_composition(feat, graph.group_id, 10, device)
            feat_final = merge_t_out_and_k_out(t_out, k_out, group_ids)
            graph.y = graph.y.to(device)
            loss = criterion(pred, graph.y) / accumulate_steps
            pred_y = pred.argmax(dim=1)
            acc = [(((pred_y == graph.y) & (graph.y == cls)).float().sum() / (
                    (graph.y == cls).float().sum() + 1e-6)).item() // 0.0001 / 100 for cls in range(2)]
            logging.info(
                f'pos stage2 acc {acc}  with gt {graph.y.sum()}/{graph.y.shape[0]}  pred {pred_y.sum()}/{pred_y.shape[0]}')
            train_acc2.append(acc)
            loss.backward()

        # 训练负样本图
        for graph in neg_graphs_2:
            graph.x = graph.x.to(device)
            graph.edge_index = graph.edge_index.to(device)
            pred, feat = model(graph)
            tx, group_ids = generate_transformer_input_and_its_groups_id_sequence(graph.group_id, feat)
            t_out, t_attn = graph_transformer(tx)
            cls_tokens = t_out[:16]
            k_out = get_nodes_group_composition(feat, graph.group_id, 10, device)
            feat_final = merge_t_out_and_k_out(t_out, k_out, group_ids)
            graph.y = graph.y.to(device)
            loss = criterion(pred, graph.y) / accumulate_steps
            pred_y = pred.argmax(dim=1)
            acc = [(((pred_y == graph.y) & (graph.y == cls)).float().sum() / (
                    (graph.y == cls).float().sum() + 1e-6)).item() // 0.0001 / 100 for cls in range(2)]
            logging.info(
                f'neg stage2 acc {acc}  with gt {graph.y.sum()}/{graph.y.shape[0]}  pred {pred_y.sum()}/{pred_y.shape[0]}')
            train_acc2.append(acc)
            loss.backward()

        opt.step()
        opt.zero_grad()
        opt_graph_transformer.step()
        opt_graph_transformer.zero_grad()
    return train_acc2


def main():
    args = parse_args()
    writer = SummaryWriter('./log')
    dataset = SuperpixelsDataset(
        root=args.data_root,
        raw_root="./dataset"
    )
    dataset_test = SuperpixelsTestDataset(
        root=args.data_root,
        raw_root="./dataset"
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    loader_test = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=True)

    hparams = {"num_node_features": args.num_node_features,
               "activation": args.activation,
               "num_conv_layers": args.num_conv_layers,
               "conv_size": args.conv_size,
               "pool_method": args.pool_method,
               "lin1_size": args.lin1_size,
               "lin2_size": args.lin2_size,
               "output_size": args.output_size}

    device = torch.device(args.device)
    criterion = nn.CrossEntropyLoss()

    model1 = GraphNet(c_in=1024, hidden_size=512, nc=2).to(device)
    opt1 = torch.optim.SGD(params=model1.parameters(), lr=args.lr1)
    scheduler1 = CosineAnnealingLR(opt1, T_max=args.T_max, eta_min=args.eta_min)

    model2 = GraphNet(c_in=1024, hidden_size=512, nc=2).to(device)
    opt2 = torch.optim.SGD(params=model2.parameters(), lr=args.lr2)
    scheduler2 = CosineAnnealingLR(opt2, T_max=args.T_max, eta_min=args.eta_min)

    graph_transformer_2 = GraphTransformer()
    opt_graph_transformer_2 = torch.optim.SGD(params=graph_transformer_2.parameters(), lr=args.lr_transformer2)

    train_acc1 = []
    train_acc2 = []
    loss_schedule_data = loss_schedule(100)

    for epoch in range(args.epochs):
        train_acc1 = train_stage1(model1, opt1, scheduler1, loader, args.dst_root, device, criterion, epoch, train_acc1)
        train_acc2 = train_stage2(model2, opt2, scheduler2, graph_transformer_2, opt_graph_transformer_2, loader,
                                  args.slide_root, args.dst_root, device, criterion, epoch, train_acc2, args.K,
                                  loss_schedule_data)

        logging.info(f'epoch {epoch} total train acc1 {np.mean(train_acc1, axis=0)}')
        logging.info(f'epoch {epoch} total train acc2 {np.mean(train_acc2, axis=0)}')
        train_acc1 = []
        train_acc2 = []

        scheduler1.step()
        scheduler2.step()

        model_save_dir = args.model_save_dir
        if not os.path.exists(model_save_dir):
            os.makedirs(model_save_dir)
        torch.save(model1.state_dict(), os.path.join(model_save_dir, '1.pth'))
        torch.save(model2.state_dict(), os.path.join(model_save_dir, '2.pth'))


if __name__ == "__main__":
    main()
