export CUDA_VISIBLE_DEVICES=0
DATA_PATH=/data1/DATASET/MSVD
python -m torch.distributed.launch --nproc_per_node=1 --master_port 29663 \
main_task_retrieval.py --do_eval --num_thread_reader=12 \
--epochs=5 --batch_size=128 --n_display=50 \
--data_path ${DATA_PATH}/msvd_data \
--features_path ${DATA_PATH}/videos \
--output_dir ckpts/ckpt_pretrain_msvd_test \
--lr 1e-4 --max_words 32 --max_frames 12 --batch_size_val 512 \
--datatype msvd \
--feature_framerate 1 --coef_lr 1e-3 \
--freeze_layer_num 0 --slice_framepos 2 \
--loose_type --linear_patch 2d --sim_header seqTransf \
--init_model ckpts/ckpt_pretrain_logsoftmax_tempsimsiam_0.1_2M_1e-4_l14/pytorch_model.bin.2 \
--pretrained_clip_name ViT-L/14