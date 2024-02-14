rye run torchrun --rdzv_endpoint localhost:29511  --nnodes=1 --node_rank=0 --nproc_per_node=2 src/what_transformer_looked_at/main.py --mode norms_abs --visible_gpu 5 6 
