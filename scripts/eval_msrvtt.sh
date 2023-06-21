export CUDA_VISIBLE_DEVICES=0,1
DATA_PATH=/data1/DATASET/MSRVTT
python -m torch.distributed.launch --nproc_per_node=2 --master_port 23889 \
main_task_retrieval.py --do_eval --num_thread_reader=8 \
--epochs=5 --batch_size=128 --n_display=50 \
--train_csv ${DATA_PATH}/msrvtt_data/MSRVTT_train.9k.csv \
--val_csv ${DATA_PATH}/msrvtt_data/MSRVTT_JSFUSION_test.csv \
--data_path ${DATA_PATH}/msrvtt_data/MSRVTT_data.json \
--features_path ${DATA_PATH}/compress_videos \
--output_dir ckpts/ckpt_msrvtt_retrieval_test \
--lr 1e-4 --max_words 32 --max_frames 12 --batch_size_val 128 \
--datatype msrvtt --expand_msrvtt_sentences  \
--feature_framerate 1 --coef_lr 1e-3 \
--freeze_layer_num 0  --slice_framepos 2 \
--loose_type --linear_patch 2d --sim_header seqTransf \
--init_model /data2/lyq/code/clip_pretrain/ckpts/ckpt_pretrain_logsoftmax_tempsimsiam_0.1/pytorch_model.bin.2 \
--pretrained_clip_name ViT-B/16