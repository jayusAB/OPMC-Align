CUDA_VISIBLE_DEVICES=0 \
python train.py \
--name irra \
--img_aug \
--batch_size 64 \
--MLM \
--dataset_name RSTPReid \
--root_dir ./data \
--loss_names 'sdm+id+mlm' \
--root_dir  'path to your data' \
--num_epoch 60
