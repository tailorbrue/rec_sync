LAUNCH_TRAINING(){

# accelerate config default
cd .. 
cd training
pretrained_model_name_or_path='stabilityai/stable-diffusion-2'
pretrained_vae_name_or_path='stabilityai/sdxl-vae'
gt_datapath='/home/wangziyi/rec/Accelerator-Simple-Template/DIR-D/training/gt'
rgb_datapath='/home/wangziyi/rec/Accelerator-Simple-Template/DIR-D/training/input'
train_rgb_list='/home/wangziyi/rec/Accelerator-Simple-Template/DIR-D/filelist_input.txt'
train_depth_list='/home/wangziyi/rec/Accelerator-Simple-Template/DIR-D/filelist_gt.txt'
vallist='None'
output_dir='/mnt/disk1/wangziyi/RecDiff_out'
train_batch_size=1
num_train_epochs=200
gradient_accumulation_steps=8
learning_rate=1e-5
lr_warmup_steps=0
dataloader_num_workers=4
tracker_project_name='DIR-D'


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch --mixed_precision="fp16"  --multi_gpu depth2image_trainer.py \
                  --pretrained_model_name_or_path $pretrained_model_name_or_path \
                  --gt_datapath $gt_datapath \
                  --rgb_datapath $rgb_datapath\
                  --train_depth_list $train_depth_list \
                  --train_rgb_list $train_rgb_list \
                  --output_dir $output_dir \
                  --train_batch_size $train_batch_size \
                  --num_train_epochs $num_train_epochs \
                  --gradient_accumulation_steps $gradient_accumulation_steps\
                  --learning_rate $learning_rate \
                  --lr_warmup_steps $lr_warmup_steps \
                  --dataloader_num_workers $dataloader_num_workers \
                  --tracker_project_name $tracker_project_name \
                  --gradient_checkpointing \
                  --enable_xformers_memory_efficient_attention \
                  --checkpointing_steps 1000 \
                  --resume_from_checkpoint "latest" \
                  --use_ema \
                  --pretrained_vae_name_or_path $pretrained_vae_name_or_path

                  
}


LAUNCH_TRAINING
