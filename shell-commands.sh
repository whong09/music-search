python train.py \
  --data-dir /home/ec2-user/data/chunks \
  --epochs 10 \
  --batch-size 8 \
  --num-workers 4 \
  --lr 1e-3 \
  --checkpoint-path ae_checkpoint.pt \
  --resume-from-checkpoint \
  --limit 2000
