inference_single_image(){
input_rgb_path="/home/wangziyi/rec/Accelerator-Simple-Template/vkitti/vkitti_1.3.1_rgb/0018/30-deg-left/00003.png"
output_dir="out_dir"
pretrained_model_path="stabilityai/stable-diffusion-2" # your checkpoint here
ensemble_size=10

cd ..
cd run

CUDA_VISIBLE_DEVICES=0 python run_inference.py \
    --input_rgb_path $input_rgb_path \
    --output_dir $output_dir \
    --pretrained_model_path $pretrained_model_path \
    --ensemble_size $ensemble_size
    }

inference_single_image


