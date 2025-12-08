#!/bin/bash
echo "running train_VAE.sh"
python3 train_HairRemovalVAE.py --num_epochs 10 --num_workers 4 --run_name "12082025_test_loss_vae(0.5)_ssim(0.5)" --lambda_vae 0.5 --lambda_ssim 0.5
