export CUDA_VISIBLE_DEVICES=0,1,2,3
DATA_PATH=/data1/DATASET/LSMDC
python -m torch.distributed.launch --nproc_per_node=4 --master_port 29665 \
main_task_retrieval.py --do_train --num_thread_reader=12 \
--epochs=5 --batch_size=128 --n_display=50 \
--data_path ${DATA_PATH}  \
--features_path ${DATA_PATH} \
--output_dir ckpts/ckpt_pretrain_lsmdc_2M_1e-4_random0.1_logsoftmax \
--lr 1e-4 --max_words 32 --max_frames 12 --batch_size_val 128 \
--datatype lsmdc \
--feature_framerate 1 --coef_lr 1e-3 \
--freeze_layer_num 0 --slice_framepos 2 \
--loose_type --linear_patch 2d --sim_header seqTransf \
--init_model ckpts/ckpt_pretrain_logsoftmax_tempsimsiam_0.1/pytorch_model.bin.2 \
--pretrained_clip_name ViT-B/16