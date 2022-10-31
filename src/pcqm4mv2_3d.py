import os
import os.path as osp
import shutil
import pickle



import pandas as pd
from tqdm import tqdm
import torch

from ogb.lsc import PygPCQM4Mv2Dataset
from ogb.utils.torch_util import replace_numpy_with_torchtensor


from utils.smiles_trans import smiles2graph

class PCQM4Mv2Dataset_3d(PygPCQM4Mv2Dataset):
    def __init__(self, root='./data/',
                 smiles2graph=smiles2graph,
                 transform=None,
                 pre_transform=None):
        super(PCQM4Mv2Dataset_3d, self).__init__(root, smiles2graph, transform, pre_transform)
        self.smiles2graph = smiles2graph
        self.transform, self.pre_transform = transform, pre_transform
        # self.folder = osp.join(root, 'pcqm4m-v2')   # self.folder from parent
        self.version = 1
        self.url = 'https://dgl-data.s3-accelerate.amazonaws.com/dataset/OGB-LSC/pcqm4m-v2.zip'
        # check version and update if necessary
        if osp.isdir(self.folder) and (not osp.exists(osp.join(self.folder, f'RELEASE_v{self.version}.txt'))):
            print('PCQM4Mv2 dataset has been updated.')
            shutil.rmtree(self.folder)
        self.data, self.slices = torch.load(self.processed_paths[0])


    @property
    def raw_file_names(self):
        # sdf + rdkit position.pkl
        return ['data.csv', 'position_all.pkl']

    @property
    def processed_dir(self) -> str:
        # processed_3d
        return osp.join(self.root, 'processed_3d_position')

    @property
    def processed_file_names(self):
        return ['geometric_data_processed.pt']

    def download(self):
        print('Please checkout if the raw file is downloaded!')

    def process(self):
        csv_data = pd.read_csv(self.raw_paths[0])
        homolumogap_list = csv_data['homolumogap']
        smiles_list = csv_data['smiles']
        with open(self.raw_paths[1], "rb") as f:
            position_data = pickle.load(f)
        data_list = []
        print("Converting SMILES strings into graphs...(with 3D position)")
        split_dict = self.get_idx_split()
        for i in tqdm(range(len(smiles_list))):
            smiles = smiles_list[i]
            homolumogap = homolumogap_list[i]

            data = smiles2graph(smiles)
            pos = torch.from_numpy(position_data[i]).to(torch.float)  # change dtype here, origin torch.float64, now, torch.float32
            assert data['x'].shape[0] == pos.shape[0]
            data.pos = pos
            data.y = torch.Tensor([homolumogap])

            data_list.append(data)

        # double-check prediction target
        assert (all([not torch.isnan(data_list[i].y)[0] for i in split_dict['train']]))
        assert (all([not torch.isnan(data_list[i].y)[0] for i in split_dict['valid']]))
        assert (all([torch.isnan(data_list[i].y)[0] for i in split_dict['test-dev']]))
        assert (all([torch.isnan(data_list[i].y)[0] for i in split_dict['test-challenge']]))

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)

        print('Saving...')
        torch.save((data, slices), self.processed_paths[0])



if __name__ == '__main__':
    dataset = PCQM4Mv2Dataset_3d(root='./data/')
    print(dataset)
    # print(dataset.data.edge_index)
    # print(dataset.data.edge_index.shape)
    # print(dataset.data.x.shape)
    # print(dataset[100])
    # print(dataset[100].y)
    # print(dataset.get_idx_split())
