import collections
import os.path
import os
import networkx as nx
import openslide
import numpy as np
import matplotlib.pyplot as plt
import glob
import cv2
import collections
import tqdm
from skimage.segmentation import slic, mark_boundaries
import pandas as pd
from utils.kfbslide import KFBSlide
import csv
thy_root = "/mnt/medical-data/hkw/FengShi/预后分析/HE stain/"
dst_root = "/mnt/s3/lhm/thy_zheyi/"

import concurrent.futures
from tqdm import tqdm
from multiprocessing import Pool, Pipe, freeze_support

#=============================================================#
# 接口                                                        #
#-------------------------------------------------------------#
#   multi_process_exec 多进程执行                             #
#   multi_thread_exec  多线程执行                             #
#-------------------------------------------------------------#
# 参数：                                                      #
#   f         (function): 批量执行的函数                      #
#   args_mat  (list)    : 批量执行的参数                      #
#   pool_size (int)     : 进程/线程池的大小                   #
#   desc      (str)     : 进度条的描述文字                    #
#-------------------------------------------------------------#
# 例子：                                                      #
# >>> def Pow(a,n):        ← 定义一个函数（可以有多个参数）   #
# ...     return a**n                                         #
# >>>                                                         #
# >>> args_mat=[[2,1],     ← 批量计算 Pow(2,1)                #
# ...           [2,2],                Pow(2,2)                #
# ...           [2,3],                Pow(2,3)                #
# ...           [2,4],                Pow(2,4)                #
# ...           [2,5],                Pow(2,5)                #
# ...           [2,6]]                Pow(2,6)                #
# >>>                                                         #
# >>> results=multi_thread_exec(Pow,args_mat,desc='计算中')   #
# 计算中: 100%|█████████████| 6/6 [00:00<00:00, 20610.83it/s] #
# >>>                                                         #
# >>> print(results)                                          #
# [2, 4, 8, 16, 32, 64]                                       #
#-------------------------------------------------------------#

ToBatch = lambda arr, size: [arr[i * size:(i + 1) * size] for i in range((size - 1 + len(arr)) // size)]


def batch_exec(f, args_batch, w):
    results = []
    for i, args in enumerate(args_batch):
        try:
            if isinstance(args, (list, tuple, dict)):
                ans = f(*args)
            else:
                ans = f(args)
            results.append(ans)
        except Exception as e:
            print(e)
            results.append(None)
        w.send(1)
    return results


def multi_process_exec(f, args_mat, pool_size=5, desc=None):
    if len(args_mat) == 0: return []
    batch_size = max(1, int(len(args_mat) / 4 / pool_size))
    results = []
    args_batches = ToBatch(args_mat, batch_size)
    with tqdm(total=len(args_mat), desc=desc) as pbar:
        with Pool(processes=pool_size) as pool:
            r, w = Pipe(duplex=False)
            pool_rets = []
            for i, args_batch in enumerate(args_batches):
                pool_rets.append(pool.apply_async(batch_exec, (f, args_batch, w)))
            cnt = 0
            while cnt < len(args_mat):
                try:
                    msg = r.recv()
                    pbar.update(1)
                    cnt += 1
                except EOFError:
                    print('EOFError')
                    break
            for ret in pool_rets:
                for r in ret.get():
                    results.append(r)
    return results


def multi_thread_exec(f, args_mat, pool_size=5, desc=None):
    if len(args_mat) == 0: return []
    results = [None for _ in range(len(args_mat))]
    with tqdm(total=len(args_mat), desc=desc) as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=pool_size) as executor:
            futures = {executor.submit(f, *args): i for i, args in enumerate(args_mat)}
            for future in concurrent.futures.as_completed(futures):
                i = futures[future]
                ret = future.result()
                results[i] = ret
                pbar.update(1)
    return results
def convert_numpy_img_to_superpixel_graph(img, code, n=2048):
    lbp = np.load(f"{dst_root}patch/{code}.lbp.npy")
    # mask = (img[:, :, 0] > 0) & (img[:, :, 1] > 0) & (img[:, :, 2] > 0)
    segments = slic(img, n_segments=n, compactness=8, sigma=0.9, start_label=0,convert2lab=True)
    np.save(f"{dst_root}segments/{code}.npy", segments)
    num_of_nodes = np.max(segments) + 1
    nodes = {node: {"rgb_list": [],"lbp_list":[], "r": [], "g": [], "b": [], "l": []} for node in range(num_of_nodes)}
    # get rgb values and positions
    for y in range(segments.shape[0]):
        for x in range(segments.shape[1]):
            node = segments[y, x]
            if node < 0:
                continue
            rgb = img[y, x, :]
            nodes[node]["r"].append(rgb[2])
            nodes[node]["g"].append(rgb[1])
            nodes[node]["b"].append(rgb[0])
            nodes[node]["l"].append(lbp[y][x])

    for node in nodes:
        r_bin = np.bincount(nodes[node]["r"])
        r_bin = np.pad(r_bin, (0, 256 - len(r_bin)), 'constant', constant_values=(0, 0))
        g_bin = np.bincount(nodes[node]["g"])
        g_bin = np.pad(g_bin, (0, 256 - len(g_bin)), 'constant', constant_values=(0, 0))
        b_bin = np.bincount(nodes[node]["b"])
        b_bin = np.pad(b_bin, (0, 256 - len(b_bin)), 'constant', constant_values=(0, 0))
        l_bin = np.bincount(nodes[node]["l"])
        l_bin = np.pad(l_bin, (0, 256 - len(l_bin)), 'constant', constant_values=(0, 0))
        nodes[node]["rgb_list"] = np.stack([r_bin, g_bin, b_bin]).ravel()
        nodes[node]["lbp_list"] = l_bin.ravel()
    G = nx.Graph()
    # compute node positions
    segments_ids = np.unique(segments)
    segments_ids = np.delete(segments_ids, np.where(segments_ids == -1))
    pos = np.array([np.mean(np.nonzero(segments == i), axis=1) for i in segments_ids])
    pos = pos.astype(int)
    #pos[0]为height_y pos[1]为width_x
    for node in nodes:
        feature = nodes[node]['rgb_list']
        feature = feature / np.sum(feature)
        mean = np.mean(feature)
        std = np.std(feature)
        feature = (feature - mean) / std
        lbp_feat = nodes[node]['lbp_list']
        lbp_feat = lbp_feat / np.sum(lbp_feat)
        mean = np.mean(lbp_feat)
        std = np.std(lbp_feat)
        lbp_feat = (lbp_feat - mean) / std
        feature = np.concatenate((feature,lbp_feat))
        G.add_node(node, features=feature, label=0)
    # add edges
    vs_right = np.vstack([segments[:, :-1].ravel(), segments[:, 1:].ravel()])
    vs_below = np.vstack([segments[:-1, :].ravel(), segments[1:, :].ravel()])
    bneighbors = np.unique(np.hstack([vs_right, vs_below]), axis=1)
    for i in range(bneighbors.shape[1]):
        if bneighbors[0, i] == -1 or bneighbors[1, i] == -1:
            continue
        if bneighbors[0, i] != bneighbors[1, i]:
            G.add_edge(bneighbors[0, i], bneighbors[1, i])
    # add self loops
    for node in nodes:
        G.add_edge(node, node)
    # get edge_index
    m = len(G.edges)
    edge_index = np.zeros([2 * m, 2]).astype(np.int64)
    for e, (s, t) in enumerate(G.edges):
        edge_index[e, 0] = s
        edge_index[e, 1] = t
        edge_index[m + e, 0] = t
        edge_index[m + e, 1] = s
    # get features
    num_of_features = 1024
    x = np.zeros([n, num_of_features]).astype(np.float32)
    # y = np.zeros(n).astype(np.float32)
    for node in G.nodes:
        if node >= n:
            continue
        x[node] = G.nodes[node]["features"]
        # y[node] = G.nodes[node]["label"]
    return x, edge_index



def process_img(code,y):
    y = int(y)
    # code = i.split('/')[-1].split('.svs')[0]
    img = cv2.imread(f"{dst_root}/patch/{code}.ori.png")
    # img_hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    # plt.imshow(img)
    # plt.show()
    # plt.imshow(img_hsv)
    # plt.show()
    x, edge_index = convert_numpy_img_to_superpixel_graph(img, code, n=3072)
    res = dict({})
    res['x'] = x
    res['edge'] = edge_index
    res['code'] = code
    res['y'] = y
    np.save(f'{dst_root}graph/{code}.npy', res, allow_pickle=True)

df = pd.read_csv(f'{dst_root}labels.csv')
# 遍历DataFrame
data_list = []
for index, row in df.iterrows():
    data_list.append((row["code"], row["label"]))
multi_process_exec(process_img, data_list)