export CUDA_VISIBLE_DEVICES=4,5,6,7
python -m torch.distributed.launch --nproc_per_node=4 --master_port 23903 \
main_pretrain.py --do_train --num_thread_reader=8 \
--epochs=3 --batch_size=1024 --n_display=50 \
--val_csv /data1/DATASET/MSRVTT/msrvtt_data/MSRVTT_JSFUSION_test.csv \
--data_path /data2/lyq/data/avs/v3c1/blip2_anno \
--pretrain_features_path /datassd2/DATASET/v3c1/compress_videos \
--features_path /data1/DATASET/MSRVTT/compress_videos \
--output_dir ckpts/ckpt_pretrain_logsoftmax_tempsimsiam_0.1_AVS_1e-4 \
--lr 1e-4 --max_words 32 --max_frames 8 --batch_size_val 64 \
--datatype pretrain --expand_msrvtt_sentences  \
--feature_framerate 1 --coef_lr 1e-4 \
--freeze_layer_num 0  --slice_framepos 2 \
--loose_type --linear_patch 2d --sim_header seqTransf \
--pretrained_clip_name ViT-B/16