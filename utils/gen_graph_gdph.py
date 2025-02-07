import os.path
import os
import networkx as nx
import numpy as np
import glob
import cv2
import tqdm
from skimage.segmentation import slic, mark_boundaries
import concurrent.futures
from tqdm import tqdm
from multiprocessing import Pool, Pipe, freeze_support
import skimage.feature as feature

# =============================================================#
# 接口                                                        #
# -------------------------------------------------------------#
#   multi_process_exec 多进程执行                             #
#   multi_thread_exec  多线程执行                             #
# -------------------------------------------------------------#
# 参数：                                                      #
#   f         (function): 批量执行的函数                      #
#   args_mat  (list)    : 批量执行的参数                      #
#   pool_size (int)     : 进程/线程池的大小                   #
#   desc      (str)     : 进度条的描述文字                    #
# -------------------------------------------------------------#
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
# -------------------------------------------------------------#

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


gdph_root = "/mnt/s3/lhm/GDPH/"
gdph_patch_root = "/mnt/s3/lhm/GDPH/patch/"

images = glob.glob(gdph_patch_root + '*/*.ori.jpg')
masks = glob.glob(gdph_patch_root + '*/*.mask.png')


def convert_numpy_img_to_superpixel_graph(img, code, prefix, n=1024):
    height = img.shape[0]
    width = img.shape[1]
    scale = 2
    # white_pixels = (img[:, :, 2] > 254) & (img[:, :, 1] > 254) & (img[:, :, 0] > 254)
    # white_count = np.sum(white_pixels)
    # white_ratio = (white_count * 100) / (width * height)
    # if white_ratio > 97.5:
    #     return
    # print(white_ratio)
    gt = cv2.imread(gdph_patch_root + f"{code}/{prefix}.mask.png", cv2.IMREAD_GRAYSCALE)

    img = cv2.resize(img, (width // scale, height // scale))
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lbp = feature.local_binary_pattern(img_gray, 10, 10, method="uniform")
    # mask = (img[:, :, 0] > 0) & (img[:, :, 1] > 0) & (img[:, :, 2] > 0)
    # gt = cv2.resize(gt, (width // scale, height // scale))
    # plt.imshow(img)
    # plt.show()
    # plt.imshow(gt)
    # plt.colorbar()
    # plt.show()

    segments = slic(img, n_segments=n, compactness=8, sigma=0.9, start_label=0)
    # ,mask=mask)
    if not os.path.exists(f"/mnt/s3/lhm/GDPH/seg_tissue/{code}"):
        os.mkdir(f"/mnt/s3/lhm/GDPH/seg_tissue/{code}")
    np.save(f"/mnt/s3/lhm/GDPH/seg_tissue/{code}/{prefix}.npy", segments)
    num_of_nodes = np.max(segments) + 1
    nodes = {node: {"rgb_list": [], "lbp_list": [], "r": [], "g": [], "b": [], "y": [], "l": []} for node in
             range(num_of_nodes)}
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
            nodes[node]["y"].append(gt[y][x])
            nodes[node]["l"].append(lbp[y][x])

    for node in nodes:
        r_bin = np.bincount(nodes[node]["r"])
        r_bin = np.pad(r_bin, (0, 256 - len(r_bin)), 'constant', constant_values=(0, 0))
        g_bin = np.bincount(nodes[node]["g"])
        g_bin = np.pad(g_bin, (0, 256 - len(g_bin)), 'constant', constant_values=(0, 0))
        b_bin = np.bincount(nodes[node]["b"])
        b_bin = np.pad(b_bin, (0, 256 - len(b_bin)), 'constant', constant_values=(0, 0))
        l_bin = np.bincount(nodes[node]["l"])
        l_bin = np.pad(l_bin, (0, 12 - len(l_bin)), 'constant', constant_values=(0, 0))
        nodes[node]["rgb_list"] = np.stack([r_bin, g_bin, b_bin]).ravel()
        nodes[node]["lbp_list"] = l_bin.ravel()
    G = nx.Graph()
    # compute node positions
    segments_ids = np.unique(segments)
    segments_ids = np.delete(segments_ids, np.where(segments_ids == -1))
    pos = np.array([np.mean(np.nonzero(segments == i), axis=1) for i in segments_ids])
    pos = pos
    pos = pos.astype(int)
    # pos[0]为height_y pos[1]为width_x
    for node in nodes:
        feat = nodes[node]['rgb_list']
        feat = feat / np.sum(feat)
        mean = np.mean(feat)
        std = np.std(feat)
        feat = (feat - mean) / std
        lbp_feat = nodes[node]['lbp_list']
        lbp_feat = lbp_feat / np.sum(lbp_feat)
        mean = np.mean(lbp_feat)
        std = np.std(lbp_feat)
        lbp_feat = (lbp_feat - mean) / std
        feat = np.concatenate((feat, lbp_feat))
        counts = np.bincount(nodes[node]['y'])
        most_common_element = np.argmax(counts)
        G.add_node(node, features=feat, label=most_common_element)
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
    num_of_features = 780
    x = np.zeros([n, num_of_features]).astype(np.float32)
    y = np.zeros(n).astype(np.float32)
    for node in G.nodes:
        if node >= n:
            continue
        x[node] = G.nodes[node]["features"]
        y[node] = G.nodes[node]["label"]
    return x, y, edge_index, pos


def process_img(i):
    img = cv2.imread(i)
    code = i.split('/')[-2]
    prefix = i.split('/')[-1].split('.ori.jpg')[0]
    if not os.path.exists(gdph_patch_root + f"{code}/{prefix}.mask.png"):
        return
    x, y, edge_index, pos = convert_numpy_img_to_superpixel_graph(img, code, prefix)
    res = dict({})
    res['x'] = x
    res['edge'] = edge_index
    res['pos'] = pos
    res['code'] = code
    res['nodey'] = y
    if not os.path.exists(f'{gdph_root}/graph_tissue/{code}'):
        os.mkdir(f'{gdph_root}/graph_tissue/{code}')
    np.save(f'{gdph_root}/graph_tissue/{code}/{prefix}.npy', res, allow_pickle=True)


# args_mat = glob.glob(gdph_patch_root + '*/*.ori.jpg')[49500:]
args_mat = glob.glob(gdph_patch_root + '*/*.ori.jpg')
# print(args_mat)
multi_process_exec(process_img, args_mat, 48, desc='processing')
