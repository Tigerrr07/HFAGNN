from contextlib import contextmanager
import os
import random
import argparse
import numpy as np
from tqdm import tqdm


import torch
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
from torch.optim.lr_scheduler import StepLR
from torch_geometric.loader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.distributed import DistributedSampler

from ogb.lsc import PCQM4Mv2Evaluator
from utils.smiles_trans import smiles2graph, get_position

from pcqm4mv2_3d import PCQM4Mv2Dataset_3d

parser = argparse.ArgumentParser(description='OGB LSC2022')
parser.add_argument('--root', type=str, default='./data/',
                    help='dataset root')
parser.add_argument('--raw_path', type=str, default='./data/pcqm4m-v2/raw/data.csv.gz',
                    help='raw smiles file path')
parser.add_argument('--graph_pooling', type=str, default='sum',
                    help='graph pooling strategy mean or sum (default: sum)')
parser.add_argument('--dropout', type=float, default=0,
                    help='dropout ratio (default: 0)')
parser.add_argument('--num_layers', type=int, default=4,
                    help='num of conv layers')
parser.add_argument('--emb_dim', type=int, default=256,
                    help='dimensionality of hidden units in GNNs (default: 256)')
parser.add_argument('--train_subset', action='store_true')
parser.add_argument('--subset_ratio', type=float, default=0.1, 
                    help='set train subset ratio for evaluate model quickly')
parser.add_argument('--batch_size', type=int, default=256,
                    help='input batch size for training (default: 256)')
parser.add_argument('--epochs', type=int, default=100,
                    help='number of epochs to train (default: 100)')
parser.add_argument('--lr', default=0.0005, type=float, help='initial learning rate')
parser.add_argument('--num_workers', type=int, default=0,
                    help='number of workers (default: 0)')
parser.add_argument('--log_dir', type=str, default="log",
                    help='tensorboard log directory')
parser.add_argument('--checkpoint_dir', type=str, default = "ckpt", help='directory to save checkpoint')
parser.add_argument('--save_test_dir', type=str, default = "saved", help='directory to save test submission file')
parser.add_argument('--seed', default=0, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--nnodes', default=1, type=int,
                    help='number of work nodes for distributed training')
parser.add_argument('--node_rank', default=0, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist_url', default='tcp://127.0.0.1:28765', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--gnn', type=str, default='HFAGNN')
parser.add_argument('--processes', type=int, default=10, help='num of cpu to process smiles.')
parser.add_argument('--norm', type=str, default='layer', help='which norm type: {batch, layer, instance}')

def main():
    args = parser.parse_args()
    fmt_template = '/{}_layers{}_emb{}_dr{}'.format(args.gnn, args.num_layers, args.emb_dim, args.dropout)
    args.log_dir = args.log_dir + fmt_template
    args.checkpoint_dir = args.checkpoint_dir + fmt_template
    args.save_test_dir = args.save_test_dir + fmt_template

    if args.train_subset:
        args.log_dir += f'_sr{args.subset_ratio}'
        args.checkpoint_dir += f'_sr{args.subset_ratio}'
        args.save_test_dir += f'_sr{args.subset_ratio}'
    print(args)

    # os.environ['NCCL_SHM_DISABLE'] = '1'

    # Process first if position file and processed file doesn't exsit 
    get_position(args.raw_path, workers=20) # root/pcqm4m-v2/raw/position_all.pkl
    PCQM4Mv2Dataset_3d(root=args.dataset_root, smile2graph=smiles2graph) # root/pcqm4m-v2/processed_3d/geometric_data_processed.pt
    ngpus_per_node = torch.cuda.device_count()

    args.world_size = args.nnodes * ngpus_per_node # default one process one gpu
    print(f'Use {args.world_size} GPUs.')
    # mp.spawn was is called as main_worker(i, *args)
    # for i in  range(nprocs) as local_rank
    if args.world_size > 1:
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        main_worker(0, 1, args)

def main_worker(local_rank, ngpus_per_node, args):
    args.local_rank = local_rank
    # global rank
    rank = args.node_rank * ngpus_per_node + local_rank
    init_seeds(1+rank)  # random seed

    dist.init_process_group('nccl', init_method=args.dist_url, rank=rank, world_size=args.world_size)
    device = torch.device("cuda", local_rank)

    # Devide the total batch size and total number of workers
    args.batch_size = int(args.batch_size / ngpus_per_node)
    args.num_workers = int((args.num_workers + ngpus_per_node - 1) / ngpus_per_node)
    # Import model
    model = import_model(args)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
    model = DistributedDataParallel(model, device_ids=[local_rank])

    # Define loss function (criterion), optimizer, and learning rate scheduler
    criterion = torch.nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    if args.train_subset:
        scheduler = StepLR(optimizer, step_size=300, gamma=0.25)
        args.epochs = 1000
    else:
        scheduler = StepLR(optimizer, step_size=30, gamma=0.25)
    
    # Load dataset and get train_dataset only in main process
    with torch_distributed_zero_first(rank):
        dataset = PCQM4Mv2Dataset_3d(root=args.root, smiles2graph=smiles2graph)
        split_idx = dataset.get_idx_split()
        # training with subset samples
        if args.train_subset:
            subset_ratio = args.subset_ratio
            subset_idx = torch.randperm(len(split_idx["train"]))[:int(subset_ratio*len(split_idx["train"]))]
            train_dataset = dataset[split_idx['train'][subset_idx]]
        else:  # full samples
            train_dataset = dataset[split_idx['train']]

    train_sampler = DistributedSampler(train_dataset, num_replicas=ngpus_per_node, rank=rank)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              num_workers=args.num_workers, sampler=train_sampler)

    if rank == 0:
        evaluator = PCQM4Mv2Evaluator()
        valid_loader = DataLoader(dataset[split_idx['valid']], batch_size=args.batch_size,
                                         shuffle=False, num_workers=args.num_workers)

        if args.checkpoint_dir != '':
            os.makedirs(args.checkpoint_dir, exist_ok=True)

        if args.log_dir != '':
            writer = SummaryWriter(log_dir=args.log_dir)  # automatic mkdir

        best_valid_mae = 1000

        print(f"Number of training samples: {len(dataset[split_idx['train']])}, Number of validation samples: {len(dataset[split_idx['valid']])}")
        print(f"Number of test-dev samples: {len(dataset[split_idx['test-dev']])}, Number of test-challenge samples: {len(dataset[split_idx['test-challenge']])}")

        num_params = sum(p.numel() for p in model.parameters())
        print(f'#Params: {num_params}', f'#GPU Memory Used for parameters: {torch.cuda.memory_allocated() / (1024 * 1024 * 1024):.4f} G')

    dist.barrier()
    
    for epoch in range(1, args.epochs + 1):
        train_loader.sampler.set_epoch(epoch)
        print("=====Epoch {}".format(epoch))
        print(f'Training in GPU {local_rank}')
        train_mae = train(model, device, train_loader, criterion, optimizer)
        dist.barrier()

        if rank == 0:
            print('Evaluating...')
            valid_mae = eval(model, device, valid_loader, evaluator)

            print(f'Epoch: {epoch:03d}, Train: {train_mae:.4f}, \
            Validation: {valid_mae:.4f}, lr: {scheduler.get_last_lr()[0]:.2e}')

            if args.log_dir != '':
                writer.add_scalar('valid/mae', valid_mae, epoch)
                writer.add_scalar('train/mae', train_mae, epoch)

            if valid_mae < best_valid_mae:
                best_valid_mae = valid_mae
                if args.checkpoint_dir != '':
                    print('Saving checkpoint...')
                    checkpoint = {'epoch': epoch, 'model_state_dict': model.module.state_dict(),
                                  'optimizer_state_dict': optimizer.state_dict(),
                                  'scheduler_state_dict': scheduler.state_dict(), 'best_val_mae': best_valid_mae,
                                  'num_params': num_params}
                    torch.save(checkpoint, os.path.join(args.checkpoint_dir, 'checkpoint.pt'))

            print(f'Best validation MAE so far: {best_valid_mae:.4f}')

        scheduler.step()

    # save submission file use best model in main process
    if rank == 0:
        # close log file
        if args.log_dir != '':
            writer.close()
        if args.save_test_dir != '' and args.checkpoint_dir != '':
            testdev_loader = DataLoader(dataset[split_idx["test-dev"]], batch_size=args.batch_size, shuffle=False)
            testchallenge_loader = DataLoader(dataset[split_idx["test-challenge"]], batch_size=args.batch_size,
                                        shuffle=False)
            best_model = import_model(args)
            best_model.to(device)
            best_model_ckpt = torch.load(os.path.join(args.checkpoint_dir, 'checkpoint.pt'))
            print(f"best val mae: {best_model_ckpt['best_val_mae']:.4f}")
            best_model.load_state_dict(best_model_ckpt['model_state_dict'])
            testdev_pred = test(best_model, device, testdev_loader)
            testdev_pred = testdev_pred.cpu().detach().numpy()

            testchallenge_pred = test(best_model, device, testchallenge_loader)
            testchallenge_pred = testchallenge_pred.cpu().detach().numpy()

            print('Saving test submission file...')
            evaluator.save_test_submission({'y_pred': testdev_pred}, args.save_test_dir, mode='test-dev')
            evaluator.save_test_submission({'y_pred': testchallenge_pred}, args.save_test_dir,
                                        mode='test-challenge')

    dist.destroy_process_group()

def import_model(args):
    if args.gnn == 'HFAGNN':
        from model.HFAGNN import HFAGNN
        model =  HFAGNN(cutoff=8.0,
                        num_layers=args.num_layers,
                        hidden_channels=args.emb_dim,
                        middle_channels=args.emb_dim // 2,
                        out_channels=1,
                        dropout=args.dropout,
                        num_radial=3,
                        num_spherical=2,
                        norm='layer')                    
    else:
        raise ValueError('Invalid MODEL type')
    return model

def train(model, device, loader, criterion, optimizer):
    model.train()
    loss_accum = 0

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)
        pred = model(batch).view(-1,)
        optimizer.zero_grad()
        loss = criterion(pred, batch.y)
        loss.backward()
        optimizer.step()
        loss_accum += loss.detach().cpu().item()
        print(loss.detach().cpu().item())
    return loss_accum / (step + 1)


def eval(model, device, loader, evaluator):
    model.eval()
    y_true = []
    y_pred = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)
        with torch.no_grad():
            pred = model(batch).view(-1,)

        y_true.append(batch.y.view(pred.shape).detach().cpu())
        y_pred.append(pred.detach().cpu())

    y_true = torch.cat(y_true, dim=0)
    y_pred = torch.cat(y_pred, dim=0)

    input_dict = {"y_true": y_true, "y_pred": y_pred}

    return evaluator.eval(input_dict)["mae"]


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

@contextmanager
def torch_distributed_zero_first(local_rank):
    if local_rank not in [-1, 0]:
        dist.barrier()
    yield
    if local_rank == 0:
        dist.barrier()

def init_seeds(seed=0, cuda_deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    # if cuda_deterministic:  # slower, more reproducible
    #     cudnn.deterministic = True
    #     cudnn.benchmark = False
    # else:  # faster, less reproducible
    #     cudnn.deterministic = False
    #     cudnn.benchmark = True

if __name__ == '__main__':
    main()
