#!/bin/bash
echo "running train_VAE.sh"
python3 train_HairRemovalVAE.py --num_epochs 200 --num_workers 4 --run_name "12082025"
