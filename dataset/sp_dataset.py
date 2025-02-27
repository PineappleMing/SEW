import glob
import os
from typing import Callable, Optional

import numpy as np
import torch
from torch_geometric.data import InMemoryDataset, Data


class SuperpixelsDataset(InMemoryDataset):
    def __init__(
            self,
            root,
            raw_root,
            transform: Optional[Callable] = None,
            pre_transform: Optional[Callable] = None,
            pre_filter: Optional[Callable] = None,
            **kwargs,
    ):
        self.slic_kwargs = kwargs
        self.raw_root = raw_root
        super().__init__(os.path.join(root, "DATA"), transform,
                         pre_transform, pre_filter)

        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_dir(self) -> str:
        return os.path.join(self.root, 'processed')

    @property
    def processed_file_names(self):
        """Dynamically generates filename for processed dataset based on superpixel parameters."""
        return ['data.pt']

    def process(self):
        data_list = []
        gs = glob.glob(self.raw_root + '*.npy')
        for g in gs:
            graph = np.load(g, allow_pickle=True).item()
            x, y, edge_index, code = graph['x'], graph['y'], graph[
                'edge'], graph['code']
            x = torch.as_tensor(x, dtype=torch.float32)
            edge_index = torch.as_tensor(edge_index, dtype=torch.long).T
            node_y = torch.as_tensor(y, dtype=torch.long)
            data_list.append(
                Data(x=x,
                     y=y,
                     edge_index=edge_index,
                     code=code, ))
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


class SuperpixelsTestDataset(InMemoryDataset):
    def __init__(
            self,
            root,
            raw_root,
            transform: Optional[Callable] = None,
            pre_transform: Optional[Callable] = None,
            pre_filter: Optional[Callable] = None,
            **kwargs,
    ):
        self.slic_kwargs = kwargs
        self.raw_root = raw_root
        super().__init__(os.path.join(root, "DATA_test"), transform,
                         pre_transform, pre_filter)

        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_dir(self) -> str:
        return os.path.join(self.root, 'processed')

    @property
    def processed_file_names(self):
        """Dynamically generates filename for processed dataset based on superpixel parameters."""
        return ['data.pt']

    def process(self):
        data_list = []

        gs = glob.glob(self.raw_root + '*.npy')
        gs = gs[int(len(gs) * 0.8):]
        for g in gs:
            graph = np.load(g, allow_pickle=True).item()
            x, edge_index, pos, y, code = graph['x'], graph['edge'], graph[
                'pos'], graph['y'], graph['code']
            x = torch.as_tensor(x, dtype=torch.float32)
            edge_index = torch.as_tensor(edge_index, dtype=torch.long).T
            pos = torch.as_tensor(pos, dtype=torch.float32)
            y = torch.as_tensor(y, dtype=torch.long)
            data_list.append(
                Data(x=x, edge_index=edge_index, pos=pos, y=y, code=code))
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])