#!/bin/bash

# from scratch
OMP_NUM_THREADS=1 torchrun --nproc-per-node=3 src/train_vit.py

# from ckpt
# OMP_NUM_THREADS=1 torchrun --nproc-per-node=3 src/train_vit.py --resume-path ./checkpoints/my-vit-tiny-patch16-224/20250807-152950-vit_tiny_patch16_224/checkpoint-63.pth.tar