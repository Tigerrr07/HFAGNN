export NCCL_SHM_DISABLE=1
export CUDA_VISIBLE_DEVICES=0
python train.py --root=./data/ --gnn=HFAGNN --num_layers=6 --dropout=0.1 \
--batch_size=512 --emb_dim=256 --epochs=100 --lr=0.0005 --dist_url=tcp://127.0.0.1:28720 \
--num_workers=4
