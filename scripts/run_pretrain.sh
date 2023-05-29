export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
DATA_PATH=/root
python -m torch.distributed.launch --nproc_per_node=8 --master_port 23903 \
main_pretrain.py --do_train --num_thread_reader=2 \
--epochs=3 --batch_size=2048 --n_display=50 \
--val_csv /data1/DATASET/MSRVTT/msrvtt_data/MSRVTT_JSFUSION_test.csv \
--data_path ${DATA_PATH}/HD_VILA_BLIP2 \
--features_path ${DATA_PATH}/HD_VILA \
--output_dir ckpts/ckpt_pretrain_base \
--lr 1e-4 --max_words 32 --max_frames 12 --batch_size_val 64 \
--datatype pretrain --expand_msrvtt_sentences  \
--feature_framerate 1 --coef_lr 1e-4 \
--freeze_layer_num 0  --slice_framepos 2 \
--loose_type --linear_patch 2d --sim_header seqTransf \
--pretrained_clip_name ViT-B/16