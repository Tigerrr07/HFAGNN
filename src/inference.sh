python test_inference.py \
--checkpoint_dir=./ckpt/HFAGNN_layers6_emb256_dr0.1 \
--raw_path=./data/pcqm4m-v2/raw/data.csv.gz \
--split_path=./data/pcqm4m-v2/split_dict.pt \
--num_workers=2 --gnn=HFAGNN --dropout=0.1 --num_layers=6 \
--batch_size=128 --emb_dim=256 --device=0 --test_type=test-challenge