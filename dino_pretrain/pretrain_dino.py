# pretrain_dino.py
import argparse
import os
import subprocess
import sys

def parse_args():
    parser = argparse.ArgumentParser(description="DINO self-supervised pretraining")
    parser.add_argument(
        '--data_path',
        required=True,
        help="Path to your extracted patches (e.g. preprocess/patches/Fold1)"
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=5,
        help="Number of training epochs"
    )
    parser.add_argument(
        '--batch_size_per_gpu',
        type=int,
        default=64,
        help="Batch size per GPU for DINO training"
    )
    parser.add_argument(
        '--output_dir',
        default='checkpoints/dino',
        help="Where to save DINO checkpoints"
    )
    parser.add_argument(
        '--dist_url',
        type=str,
        default='tcp://127.0.0.1:29500',
        help="URL for torch.distributed init (must match init_distributed_mode)"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Clone repo if needed
    if not os.path.isdir('dino'):
        subprocess.run(
            ['git', 'clone', 'https://github.com/facebookresearch/dino.git', 'dino'],
            check=True
        )

    # Launch DINO training
    cmd = [
        sys.executable, 'main_dino.py',
        '--data_path',          args.data_path,
        '--epochs',             str(args.epochs),
        '--batch_size_per_gpu', str(args.batch_size_per_gpu),
        '--output_dir',         args.output_dir,
        '--dist_url',           args.dist_url,
    ]
    subprocess.run(cmd, cwd='dino', check=True)

if __name__ == '__main__':
    main()