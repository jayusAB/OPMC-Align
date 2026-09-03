DATASET_NAME="CUHK-PEDES"

CUDA_VISIBLE_DEVICES=0 \
python train.py \
--name opmc_align \
--img_aug \
--batch_size 64 \
--MLM \
--dataset_name ${DATASET_NAME} \
--loss_names 'sdm+id+mlm+wbcmcir' \
--root_dir 'path to your data' \
--wbcmcir_loss_weight 0.05 \
--wbcmcir_warmup_epochs 5 \
--circle_delta_p 0.63 \
--circle_delta_n 0.24 \
--circle_m 0.15 \
--lmc_eta 2.0 \
--lmc_norm minmax \
--optfa_iters 20 \
--optfa_lambda 0.1 \
--optfa_rho 0.5 \
--optfa_split 4 \
--optfa_dust_mode global \
--num_epoch 60
