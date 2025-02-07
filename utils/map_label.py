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
gs = glob.glob(f"{gdph_root}/graph_tissue/*/*.npy")
pixel_2_label = {59: 1, 70: 2, 73: 3, 82: 4, 109: 5, 117: 6, 120: 7, 160: 8, 161: 9, 170: 10, 217: 11,
                 255: 12}


def replace_label(g):
    a = np.load(g, allow_pickle=True).item()
    # print(g)
    nodey = a['nodey']
    # print(np.unique(nodey))
    newy = np.zeros_like(nodey)
    for index, y in enumerate(nodey):
        if int(y) in pixel_2_label:
            newy[index] = pixel_2_label[int(y)]
    a['nodey_mapped'] = newy
    # print(np.unique(newy))
    np.save(g, a)


multi_process_exec(replace_label, gs, 48)
