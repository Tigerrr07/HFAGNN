from rdkit import Chem
from rdkit.Chem import AllChem
from ogb.utils.features import atom_to_feature_vector, bond_to_feature_vector
import torch
import numpy as np
from torch_geometric.data import Data
from tqdm import tqdm
import pandas as pd
from multiprocessing import Pool
import os
import pickle

def smiles2graph(smiles_string, sdf_mol=None):
    """
    Converts SMILES string to graph Data object
    :input: SMILES string (str)
    :return: Data object
    """
    rdkit_mol = Chem.MolFromSmiles(smiles_string)
    mol = sdf_mol if sdf_mol is not None else rdkit_mol

    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_features_list.append(atom_to_feature_vector(atom))
    x = torch.from_numpy(np.array(atom_features_list)).to(torch.long)

    num_bond_features = 3
    if len(mol.GetBonds()) > 0: # mol has bonds
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_feature = bond_to_feature_vector(bond)
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)
        # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        edge_index = torch.from_numpy(np.array(edges_list).T).to(torch.long)
        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = torch.from_numpy(np.array(edge_features_list)).to(torch.long)
    else:   # mol has no bonds
        # print('mol has no bonds')
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, num_bond_features), dtype=torch.long)
    
    nnodes = x.shape[0]
    # add nnodes for consturct complete edge_index when model called
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, nnodes=nnodes)
    
    return data

def worker_func(csv_path, worker_id, num_workers):
    """
    return 3D postion for all data using rdkit.
    """
    csv_data = pd.read_csv(csv_path)
    # sdf_data = Chem.SDMolSupplier(sdf_path)
    smiles_list = csv_data['smiles']
    size = len(smiles_list)
    chunk_size = size // num_workers
    offset = worker_id * chunk_size
    # if worker_id == 0:  # 1 process for train examples
    #     offset = 0
    #     end = num_train
    if worker_id == num_workers-1:
        end = size
    else:
        end = offset + chunk_size
    print("From smiles to position")
    pos_list = []
    for i in tqdm(range(offset, end)):
        smiles = smiles_list[i]
        position = smile2pos(smiles)
        pos_list.append(position)
    return pos_list

def smile2pos(smiles):
    """
    Read SMILES, return 3D mol with 3D position
    """
    mol = Chem.MolFromSmiles(smiles)
    try:
        temp_mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(temp_mol, randomSeed=5)
        AllChem.MMFFOptimizeMolecule(temp_mol)
        mol = Chem.RemoveHs(temp_mol)
        position = mol.GetConformer().GetPositions()
    except ValueError:  # only 2d position
        AllChem.Compute2DCoords(mol)
        position = mol.GetConformer().GetPositions()
    return position

def get_position(csv_path, workers=20):
    """
    Read files from raw 2d data.
    Multi Processes get 3d position using rdkit from smiles.
    """
    worker_thread = []
    pool = Pool(processes=workers)
    for i in range(workers):
        w = pool.apply_async(worker_func, args=(
            csv_path, i, workers
        ))
        worker_thread.append(w)
    pool.close()
    pool.join()

    pos_all = []
    for w in worker_thread:
        pos_list = w.get()
        pos_all += pos_list

    # ../data/pcqm4m-v2/raw/position_all.pkl
    store_path = os.path.join(os.path.dirname(csv_path), 'position_all.pkl')
    with open(store_path, "wb") as f:
        pickle.dump(pos_all, f)

if __name__ == '__main__':
    csv_path = '../data/pcqm4m-v2/raw/data.csv.gz'
    get_position(csv_path, workers=10)

    store_path = os.path.join(os.path.dirname(csv_path), 'position_all.pkl')
    with open(store_path, "rb") as g:
        pos = pickle.load(g)
    print(len(pos))
