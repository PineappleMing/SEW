import collections
import os
import networkx as nx
import numpy as np
import cv2
from skimage.segmentation import slic
from skimage.feature import local_binary_pattern
import pandas as pd
import argparse

# Define LBP parameters
radius = 3
n_points = 8 * radius


def generate_lbp(img, code, dst_root):
    """
    Generate the Local Binary Pattern (LBP) features of an image and save them as a .npy file.
    :param img: Input image.
    :param code: Identifier for the image.
    :param dst_root: Root directory for data storage.
    :return: Generated LBP features.
    """
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
        lbp_path = os.path.join(dst_root, 'patch', f'{code}.lbp.npy')
        np.save(lbp_path, lbp)
        return lbp
    except Exception as e:
        print(f"Error generating LBP features for {code}: {e}")
        return None


def convert_numpy_img_to_superpixel_graph(img, code, n, dst_root):
    """
    Convert a numpy image to a superpixel graph.
    :param img: Input image.
    :param code: Identifier for the image.
    :param n: Number of segments for superpixel segmentation.
    :param dst_root: Root directory for data storage.
    :return: Node features and edge index of the superpixel graph.
    """
    try:
        lbp_path = os.path.join(dst_root, 'patch', f'{code}.lbp.npy')
        if not os.path.exists(lbp_path):
            lbp = generate_lbp(img, code, dst_root)
        else:
            lbp = np.load(lbp_path)

        segments = slic(img, n_segments=n, compactness=8, sigma=0.9, start_label=0, convert2lab=True)
        np.save(os.path.join(dst_root, 'segments', f'{code}.npy'), segments)
        num_of_nodes = np.max(segments) + 1
        nodes = {node: {"rgb_list": [], "lbp_list": [], "r": [], "g": [], "b": [], "l": []} for node in
                 range(num_of_nodes)}
        # Get RGB values and positions
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
        # Compute node positions
        segments_ids = np.unique(segments)
        segments_ids = np.delete(segments_ids, np.where(segments_ids == -1))
        pos = np.array([np.mean(np.nonzero(segments == i), axis=1) for i in segments_ids])
        pos = pos.astype(int)

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
            feature = np.concatenate((feature, lbp_feat))
            G.add_node(node, features=feature, label=0)

        # Add edges
        vs_right = np.vstack([segments[:, :-1].ravel(), segments[:, 1:].ravel()])
        vs_below = np.vstack([segments[:-1, :].ravel(), segments[1:, :].ravel()])
        bneighbors = np.unique(np.hstack([vs_right, vs_below]), axis=1)
        for i in range(bneighbors.shape[1]):
            if bneighbors[0, i] == -1 or bneighbors[1, i] == -1:
                continue
            if bneighbors[0, i] != bneighbors[1, i]:
                G.add_edge(bneighbors[0, i], bneighbors[1, i])

        # Add self-loops
        for node in nodes:
            G.add_edge(node, node)

        # Get edge index
        m = len(G.edges)
        edge_index = np.zeros([2 * m, 2]).astype(np.int64)
        for e, (s, t) in enumerate(G.edges):
            edge_index[e, 0] = s
            edge_index[e, 1] = t
            edge_index[m + e, 0] = t
            edge_index[m + e, 1] = s

        # Get features
        num_of_features = 1024
        x = np.zeros([n, num_of_features]).astype(np.float32)
        for node in G.nodes:
            if node >= n:
                continue
            x[node] = G.nodes[node]["features"]

        return x, edge_index
    except Exception as e:
        print(f"Error processing {code}: {e}")
        return None, None


def process_img(img_path, y, n, dst_root):
    """
    Process an image to generate a superpixel graph and save the result.
    :param img_path: Path to the input image.
    :param y: Label of the image.
    :param n: Number of segments for superpixel segmentation.
    :param dst_root: Root directory for data storage.
    """
    y = int(y)
    try:
        img = cv2.imread(img_path)
        if img is None:
            print(f"Failed to read image: {img_path}")
            return
        code = img_path.split("/")[-1].split(".")[0]
        x, edge_index = convert_numpy_img_to_superpixel_graph(img, code, n, dst_root)
        if x is None or edge_index is None:
            return
        res = {
            'x': x,
            'edge': edge_index,
            'code': code,
            'y': y
        }
        output_path = os.path.join(dst_root, 'graph', f'{code}.npy')
        np.save(output_path, res, allow_pickle=True)
    except Exception as e:
        print(f"Error processing image {img_path}: {e}")


def main():
    """
    Main function to parse command-line arguments and process images.
    """
    parser = argparse.ArgumentParser(description='Script for generating superpixel graphs from images.')
    parser.add_argument('--dst_root', type=str, default='/path/to/dst',
                        help='Root directory for data storage.')
    parser.add_argument('--label_csv', type=str, default='labels.csv',
                        help='Path to the CSV file containing image label information (relative to dst_root).')
    parser.add_argument('--n_segments', type=int, default=3072,
                        help='Number of segments for superpixel segmentation.')

    args = parser.parse_args()
    dst_root = args.dst_root
    label_csv_path = os.path.join(dst_root, args.label_csv)
    n_segments = args.n_segments

    # Create necessary folders
    for folder in ['patch', 'segments', 'graph']:
        os.makedirs(os.path.join(dst_root, folder), exist_ok=True)

    df = pd.read_csv(label_csv_path)
    for index, row in df.iterrows():
        process_img(row["img"], row["label"], n_segments, dst_root)


if __name__ == "__main__":
    main()