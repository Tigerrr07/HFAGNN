import torch
from torch_geometric.loader import DataLoader
import pandas as pd
import os
from tqdm import tqdm
import argparse
import numpy as np
import random

### importing OGB-LSC
from ogb.lsc import PCQM4Mv2Evaluator
from utils.smiles_trans import smiles2graph, smile2pos

def test(model, device, loader):
    model.eval()
    y_pred = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        with torch.no_grad():
            pred = model(batch).view(-1,)

        y_pred.append(pred.detach().cpu())

    y_pred = torch.cat(y_pred, dim = 0)

    return y_pred

class OnTheFlyPCQMDataset(object):
    def __init__(self, smiles_list, smiles2graph=smiles2graph):
        super(OnTheFlyPCQMDataset, self).__init__()
        self.smiles_list = smiles_list 
        self.smiles2graph = smiles2graph

    def __getitem__(self, idx):
        '''Get datapoint with index'''
        smiles= self.smiles_list[idx]
        data = self.smiles2graph(smiles)
        pos = smile2pos(smiles)
        data.pos = torch.from_numpy(pos).to(torch.float)
        return data

    def __len__(self):
        '''Length of the dataset
        Returns
        -------
        int
            Length of Dataset
        '''
        return len(self.smiles_list)

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='GNN baselines on pcqm4m with Pytorch Geometrics')
    parser.add_argument('--raw_path', type=str, default='./data/pcqm4m-v2/raw/data.csv.gz',
                        help='dataset raw path')
    parser.add_argument('--num_output_layers', type=int, default=3)
    parser.add_argument('--split_path', type=str, default='./data/pcqm4m-v2/split_dict.pt',
                        help='dataset split path')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--gnn', type=str, default='gin-virtual',
                        help='GNN gin, gin-virtual, or gcn, or gcn-virtual (default: gin-virtual)')
    parser.add_argument('--dropout', type=float, default=0,
                        help='dropout ratio (default: 0)')
    parser.add_argument('--num_layers', type=int, default=4,
                        help='number of GNN message passing layers (default: 4)')
    parser.add_argument('--emb_dim', type=int, default=256,
                        help='dimensionality of hidden units in GNNs (default: 256)')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='input batch size for training (default: 256)')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='number of workers (default: 0)')
    parser.add_argument('--checkpoint_dir', type=str, default = '', help='directory to save checkpoint')
    parser.add_argument('--test_type', type=str, default = 'test-dev', help='test-dev or test-challenge')
    parser.add_argument('--save_test_dir', type=str, default = 'saved', help='directory to save test submission file')
    args = parser.parse_args()

    print(args)

    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    random.seed(0)

    if args.device == -1:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:" + str(args.device))

    ### automatic dataloading and splitting
    ### Read in the raw SMILES strings
    csv_data = pd.read_csv(args.raw_path)
    smiles_list = csv_data['smiles']
    split_idx = torch.load(args.split_path)
    test_smiles_dataset = list(smiles_list[split_idx[args.test_type]])
    onthefly_dataset = OnTheFlyPCQMDataset(test_smiles_dataset)
    test_loader = DataLoader(onthefly_dataset, batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)

    ### automatic evaluator. takes dataset name as input
    evaluator = PCQM4Mv2Evaluator()

    if args.gnn == 'HFAGNN':
        from model.HFAGNN import HFAGNN
        model = HFAGNN(cutoff=8.0,
                        num_layers=args.num_layers,
                        hidden_channels=args.emb_dim,
                        middle_channels=args.emb_dim // 2,
                        out_channels=1,
                        dropout=args.dropout,
                        num_radial=3,
                        num_spherical=2,
                        norm='layer')  
    else:
        raise ValueError('Invalid GNN type')
    model = model.to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f'#Params: {num_params}')

    checkpoint_path = os.path.join(args.checkpoint_dir, 'checkpoint.pt')
    if not os.path.exists(checkpoint_path):
        raise RuntimeError(f'Checkpoint file not found at {checkpoint_path}')
    
    ## reading in checkpoint
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    print('Predicting on test data...')
    y_pred = test(model, device, test_loader)
    print('Saving test submission file...')
    evaluator.save_test_submission({'y_pred': y_pred}, args.save_test_dir, mode = args.test_type)


if __name__ == "__main__":
    main()