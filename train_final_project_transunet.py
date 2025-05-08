# Import required libraries
import argparse
import torch
from torch import optim, nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import os
from utils import DiceLoss
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from dataset_final_project import Project_dataset as Project
from dataset_final_project import RandomGenerator
from torchvision import transforms
import numpy as np
import wandb

# Set up command line arguments for script configuration
parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', type=str, default=None,
                    help="whether to restore from a checkpoint")
parser.add_argument('--kfold', type=str, default=0,
                    help="kfold cross validation")
parser.add_argument('--wandb_project', type=str, default='unet_segmentation',
                    help="wandb project name")
parser.add_argument('--wandb_entity', type=str, default=None,
                    help="wandb entity/username")
parser.add_argument('--model_name', type=str, default="transunet_final_project",
                    help="model name")
args = parser.parse_args()

# Define training hyperparameters and configuration
LEARNING_RATE = 0.05
BATCH_SIZE = 6  # Smaller batch size for TransUNet due to higher memory requirements
EPOCHS = 100 
MODEL_NAME = f"{args.model_name}_{args.kfold}"
IMAGE_PATH = "../data/slices/imgs"
LABEL_PATH = "../data/slices/masks"
TRAIN_PATH = f"../data/slices/train_{args.kfold}.txt"
VAL_PATH = f"../data/slices/valid_{args.kfold}.txt"
SUMMARY_LOGS = f'./model_out/log_final_project_transunet_{args.kfold}'
MODEL_SAVE_PATH = os.path.join(os.getcwd(), f"models/{MODEL_NAME}")
NUM_CLASSES = 2

os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
print(f"MODEL_SAVE_PATH: {MODEL_SAVE_PATH}/{MODEL_NAME}")

# Set device (GPU if available, else CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize datasets with data augmentation for training
train_dataset = Project(image_dir=IMAGE_PATH, label_dir=LABEL_PATH, list_dir=TRAIN_PATH, transform=transforms.Compose(
                                   [RandomGenerator(output_size=[512, 512])]))

# Initialize validation dataset without augmentation
val_dataset = Project(image_dir=IMAGE_PATH, label_dir=LABEL_PATH, list_dir=VAL_PATH)

# Create data loaders for training and validation
train_dataloader = DataLoader(dataset=train_dataset,
                            batch_size=BATCH_SIZE,
                            shuffle=True)

val_dataloader = DataLoader(dataset=val_dataset,
                            batch_size=BATCH_SIZE)

# Get dataset sizes
train_N = len(train_dataloader.dataset)
valid_N = len(val_dataloader.dataset)

# Configure and initialize TransUNet model
config_vit = CONFIGS_ViT_seg['R50-ViT-B_16']
config_vit.n_classes = NUM_CLASSES
config_vit.n_skip = 3
config_vit.patches.grid = (int(512 / 16), int(512 / 16))  # Set patch size for 512x512 images
model = ViT_seg(config_vit, img_size=512, num_classes=config_vit.n_classes).cuda()
model.load_from(weights=np.load(config_vit.pretrained_path))  # Load pretrained weights

# Initialize optimizer and loss functions
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=0.0001)
loss_function = nn.CrossEntropyLoss()
dice = DiceLoss(NUM_CLASSES)
writer = SummaryWriter(SUMMARY_LOGS)

iter_num = 0

# Enable multi-GPU training if available
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs")
    model = nn.DataParallel(model)

def train():
    """
    Training function that implements one epoch of training.
    Includes forward pass, loss calculation, backpropagation, and logging.
    """
    global iter_num
    model.train()
    train_running_loss = 0
    dice_loss = 0
    
    for idx, img_mask in enumerate(tqdm(train_dataloader)):
        # Prepare input data
        img = img_mask['image'].float().to(device)
        mask = img_mask['label'].long().to(device)
        
        # Convert image to appropriate format for TransUNet
        img = img.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
        
        # Forward pass through TransUNet
        y_pred = model(img)

        # Calculate combined loss (Cross Entropy + Dice)
        batch_loss = loss_function(y_pred, mask)
        dice_loss = dice(F.softmax(y_pred, dim=1), mask)
        batch_loss += dice_loss
        
        train_running_loss += batch_loss.item()
        
        # Backpropagation
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

        # Learning rate scheduling
        lr_ = LEARNING_RATE * max(0.0, (1.0 - iter_num / (EPOCHS * len(train_dataloader)))) ** 0.9
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_

        iter_num += 1
        
        # Log training metrics
        writer.add_scalar('info/lr', lr_, iter_num)
        writer.add_scalar('info/total_loss', batch_loss, iter_num)
        writer.add_scalar('info/dice', dice_loss, iter_num)
        
        # Log sample images and predictions
        if iter_num%10 == 0:
            image = img[1, 0:1, :, :]
            image = (image - image.min()) / (image.max() - image.min())
            writer.add_image('train/Image', image, iter_num)
            outputs = torch.argmax(torch.softmax(y_pred, dim=1), dim=1, keepdim=True)
            writer.add_image('train/Prediction', outputs[1, ...] * 50, iter_num)
            labs = mask[1, ...].unsqueeze(0) * 50
            writer.add_image('train/GroundTruth', labs, iter_num)

    train_running_loss = train_running_loss/len(train_dataloader)
    print(f"Iteration: {iter_num} Train Loss: {train_running_loss:.4f} Dice Loss: {dice_loss:.4f}")

def validate():
    """
    Validation function that evaluates model performance on validation set.
    Includes forward pass, loss calculation, and logging of metrics and visualizations.
    """
    global iter_num
    model.eval()
    val_running_loss = 0
    dice_loss = 0
    with torch.no_grad():
        for idx, img_mask in enumerate(tqdm(val_dataloader)):
            # Prepare input data
            img = img_mask['image'].float().to(device)
            mask = img_mask['label'].long().to(device)
            
            # Forward pass through TransUNet
            y_pred = model(img)

            # Calculate combined loss
            batch_loss = loss_function(y_pred, mask)
            dice_loss = dice(F.softmax(y_pred, dim=1), mask)
            batch_loss += dice_loss

            val_running_loss += batch_loss.item()

            iter_num += 1
            
            # Log validation metrics
            writer.add_scalar('info/dice', dice_loss, iter_num)
            writer.add_scalar('info/loss_ce', batch_loss, iter_num)
            
            # Log sample images and predictions with error handling
            if iter_num%2 == 0:
                try:
                    # Try to log first sample in batch
                    image = img[1, 0:1, :, :]
                    image = (image - image.min()) / (image.max() - image.min())
                    writer.add_image('valid/Image', image, iter_num)
                    
                    # Convert predictions to float before visualization
                    outputs = torch.argmax(torch.softmax(y_pred, dim=1), dim=1, keepdim=True).float()
                    writer.add_image('valid/Prediction', outputs[1, ...] * 50, iter_num)
                    labs = mask[1, ...].unsqueeze(0).float() * 50
                    writer.add_image('valid/GroundTruth', labs, iter_num)
                    
                    # Log to wandb with proper type conversion
                    wandb.log({
                        "valid/Image": wandb.Image(image.cpu().numpy(), caption="Validation Image"),
                        "valid/Prediction": wandb.Image((outputs[1, ...] * 50).cpu().numpy(), caption="Validation Prediction"),
                        "valid/GroundTruth": wandb.Image((labs * 50).cpu().numpy(), caption="Validation Ground Truth")
                    })
                except Exception as e:
                    # Fallback to first sample if second sample fails
                    image = img[0, 0:1, :, :]
                    image = (image - image.min()) / (image.max() - image.min())
                    writer.add_image('valid/Image', image, iter_num)
                    
                    # Convert predictions to float before visualization
                    outputs = torch.argmax(torch.softmax(y_pred, dim=1), dim=1, keepdim=True).float()
                    writer.add_image('valid/Prediction', outputs[0, ...] * 50, iter_num)
                    labs = mask[0, ...].unsqueeze(0).float() * 50
                    writer.add_image('valid/GroundTruth', labs, iter_num)
                    
                    # Log to wandb with proper type conversion
                    wandb.log({
                        "valid/Image": wandb.Image(image.cpu().numpy(), caption="Validation Image"),
                        "valid/Prediction": wandb.Image((outputs[0, ...] * 50).cpu().numpy(), caption="Validation Prediction"),
                        "valid/GroundTruth": wandb.Image((labs * 50).cpu().numpy(), caption="Validation Ground Truth")
                    })

        val_running_loss = val_running_loss/len(val_dataloader)
        print(f"Iteration: {iter_num} Valid Loss: {val_running_loss:.4f} Dice Loss: {dice_loss:.4f}")

if __name__ == "__main__":
    # Initialize wandb for experiment tracking
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        config={
            "learning_rate": LEARNING_RATE,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "model_name": MODEL_NAME,
            "kfold": args.kfold,
            "architecture": "TransUNet",
            "num_classes": NUM_CLASSES,
            "optimizer": "SGD",
            "loss": "CrossEntropyLoss + DiceLoss"
        }
    )
    
    # Enable model tracking in wandb
    wandb.watch(model, log="all")
    
    # Main training loop
    for epoch in tqdm(range(EPOCHS)):
        print(f'Epoch: {epoch}')
        train()
        validate()

        # Save model checkpoint every 50 epochs with error handling
        if (epoch+1) % 50 == 0:
            try:    
                checkpoint_path = os.path.join(MODEL_SAVE_PATH, f"{MODEL_NAME}_{epoch}.pth")
                torch.save(model.state_dict(), checkpoint_path)
                wandb.save(checkpoint_path)
            except Exception as e:
                # Create directory if it doesn't exist and retry saving
                os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
                torch.save(model.state_dict(), checkpoint_path)
                wandb.save(checkpoint_path)

    # Save final model
    final_model_path = os.path.join(MODEL_SAVE_PATH, f"{MODEL_NAME}_final.pth")
    torch.save(model.state_dict(), final_model_path)
    wandb.save(final_model_path)
    
    # Close wandb run
    wandb.finish()
