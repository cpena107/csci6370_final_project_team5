# Prompt for wandb entity
echo "Enter your wandb entity (username):"
read wandb_entity

# Transunet
CUDA_VISIBLE_DEVICES=0 python train_final_project_transunet.py --kfold 0 --wandb_project "transunet_segmentation" --wandb_entity $wandb_entity --model_name "transunet"
CUDA_VISIBLE_DEVICES=0 python train_final_project_transunet.py --kfold 1 --wandb_project "transunet_segmentation" --wandb_entity $wandb_entity --model_name "transunet"
CUDA_VISIBLE_DEVICES=0 python train_final_project_transunet.py --kfold 2 --wandb_project "transunet_segmentation" --wandb_entity $wandb_entity --model_name "transunet"
CUDA_VISIBLE_DEVICES=0 python train_final_project_transunet.py --kfold 3 --wandb_project "transunet_segmentation" --wandb_entity $wandb_entity --model_name "transunet"
CUDA_VISIBLE_DEVICES=0 python train_final_project_transunet.py --kfold 4 --wandb_project "transunet_segmentation" --wandb_entity $wandb_entity --model_name "transunet"

# Deeplab
CUDA_VISIBLE_DEVICES=0 python train_final_project.py --kfold 0 --wandb_project "unet_segmentation_deeplab" --wandb_entity $wandb_entity --checkpoint "deeplab" --model_name "unet_deeplab"
CUDA_VISIBLE_DEVICES=0 python train_final_project.py --kfold 1 --wandb_project "unet_segmentation_deeplab" --wandb_entity $wandb_entity --checkpoint "deeplab" --model_name "unet_deeplab"
CUDA_VISIBLE_DEVICES=0 python train_final_project.py --kfold 2 --wandb_project "unet_segmentation_deeplab" --wandb_entity $wandb_entity --checkpoint "deeplab" --model_name "unet_deeplab"
CUDA_VISIBLE_DEVICES=0 python train_final_project.py --kfold 3 --wandb_project "unet_segmentation_deeplab" --wandb_entity $wandb_entity --checkpoint "deeplab" --model_name "unet_deeplab"
CUDA_VISIBLE_DEVICES=0 python train_final_project.py --kfold 4 --wandb_project "unet_segmentation_deeplab" --wandb_entity $wandb_entity --checkpoint "deeplab" --model_name "unet_deeplab"

# UNet
CUDA_VISIBLE_DEVICES=0 python train_final_project.py --kfold 0 --wandb_project "unet_segmentation" --wandb_entity $wandb_entity --model_name "unet"
CUDA_VISIBLE_DEVICES=0 python train_final_project.py --kfold 1 --wandb_project "unet_segmentation" --wandb_entity $wandb_entity --model_name "unet"
CUDA_VISIBLE_DEVICES=0 python train_final_project.py --kfold 2 --wandb_project "unet_segmentation" --wandb_entity $wandb_entity --model_name "unet"
CUDA_VISIBLE_DEVICES=0 python train_final_project.py --kfold 3 --wandb_project "unet_segmentation" --wandb_entity $wandb_entity --model_name "unet"
CUDA_VISIBLE_DEVICES=0 python train_final_project.py --kfold 4 --wandb_project "unet_segmentation" --wandb_entity $wandb_entity --model_name "unet"

