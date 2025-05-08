# CSCI 6370 COVID CT Scans Segmentation Project

## Group 5: Carlos Pena-Caballero, Christian Narcia-Macias, Tyler Morgan, Jorge Orta
## Professor: Dr. Haoteng Tang

## Overview
This project implements medical image segmentation using deep learning models. It includes implementations of UNet and TransUNet architectures for semantic segmentation tasks.

## Requirements

```bash
pip install -r requirements.txt
```

## Available Models

### 1. UNet
- Implementation of the classic UNet architecture
- Features:
  - DeepLabV3 pretrained weights integration
  - Customizable number of input channels and classes
  - Skip connections for better feature preservation

### 2. TransUNet
- Implementation of the TransUNet architecture
- Features:
  - ViT (Vision Transformer) backbone
    - R50-ViT-B_16
  - Hybrid CNN-Transformer architecture
  - Configurable patch sizes and transformer parameters

## Usage

### Training
1. Prepare your dataset in the following structure:
   ```
   ../data/
   ├── slices/
   │   ├── imgs/        # Input images
   │   ├── masks/       # Ground truth masks
   │   ├── train_*.txt  # Train split files
   │   └── valid_*.txt  # Validation split files
   ```

2. Train UNet model:
   ```bash
   python train_final_project.py [--checkpoint CHECKPOINT] [--kfold KFOLD] [--wandb_project WANDB_PROJECT] [--wandb_entity WANDB_ENTITY] [--model_name MODEL_NAME]

   Arguments:
   --checkpoint      Path to model checkpoint to restore from (default: None)
   --kfold          K-fold cross validation number (default: 0) 
   --wandb_project  Weights & Biases project name (default: 'unet_segmentation')
   --wandb_entity   Weights & Biases entity/username (default: None)
   --model_name     Name of the model (default: 'unet_final_project')
   ```

3. Train TransUNet model:
   ```bash
   python train_final_project_transunet.py [--checkpoint CHECKPOINT] [--kfold KFOLD] [--wandb_project WANDB_PROJECT] [--wandb_entity WANDB_ENTITY] [--model_name MODEL_NAME]

   Arguments:
   --checkpoint      Path to model checkpoint to restore from (default: None)
   --kfold          K-fold cross validation number (default: 0)
   --wandb_project  Weights & Biases project name (default: 'unet_segmentation')
   --wandb_entity   Weights & Biases entity/username (default: None)
   --model_name     Name of the model (default: 'unet_final_project')
   ```

### Evaluation
1. Evaluate UNet model:
   ```bash
   python dice_coefficient_from_kfold.py --model_path <path_to_model> --kfold <fold_number>
   ```

2. Evaluate TransUNet model:
   ```bash
   python dice_coefficient_from_kfold_transunet.py --model_path <path_to_model> --kfold <fold_number>
   ```

### Metrics
The evaluation includes:
- Dice Coefficient
- Hausdorff Distance 95th percentile

## Project Structure
- `networks/`: Contains model architectures for TransUnet 
- `model_out/`: Directory for saved model weights
- `slices/`: Dataset directory
- `utils.py`: Utility functions
- `dataset_final_project.py`: Dataset loading and preprocessing
- `train_final_project*.py`: Training scripts
- `dice_coefficient_from_kfold*.py`: Evaluation scripts
