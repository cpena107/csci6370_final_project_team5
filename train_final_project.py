import argparse
import torch
from torch import optim, nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import os
from utils import DiceLoss
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from UNet_vanilla import UNet
from dataset_final_project import Project_dataset as Project
from dataset_final_project import RandomGenerator
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights, deeplabv3_resnet50
import numpy as np
import wandb

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', type=str, default=None,
                    help="whether to restore from a checkpoint")
parser.add_argument('--kfold', type=str, default=0,
                    help="kfold cross validation")
parser.add_argument('--wandb_project', type=str, default='unet_segmentation',
                    help="wandb project name")
parser.add_argument('--wandb_entity', type=str, default=None,
                    help="wandb entity/username")
args = parser.parse_args()

LEARNING_RATE = 0.05
BATCH_SIZE = 12
EPOCHS = 100 
MODEL_NAME = "unet_final_project_vanilla_pretrained_deeplab_with_cross_entropy"
IMAGE_PATH = "../data/slices/imgs"
LABEL_PATH = "../data/slices/masks"
TRAIN_PATH = "../data/slices/train_" + args.kfold + ".txt"
VAL_PATH = "../data/slices/valid_" + args.kfold + ".txt"
SUMMARY_LOGS = './model_out/log_final_project_vanilla_pretrained_deeplab_with_cross_entropy_' + args.kfold
MODEL_SAVE_PATH = os.path.join(os.getcwd(), f"models/{MODEL_NAME}")
NUM_CLASSES = 2

print(f"MODEL_SAVE_PATH: {MODEL_SAVE_PATH}/{MODEL_NAME}_{args.kfold}")

device = "cuda" if torch.cuda.is_available() else "cpu"

train_dataset = Project(image_dir=IMAGE_PATH, label_dir=LABEL_PATH, list_dir=TRAIN_PATH, transform=transforms.Compose(
                                   [RandomGenerator(output_size=[512, 512])]))

val_dataset = Project(image_dir=IMAGE_PATH, label_dir=LABEL_PATH, list_dir=VAL_PATH)

train_dataloader = DataLoader(dataset=train_dataset,
                            batch_size=BATCH_SIZE,
                            shuffle=True)

val_dataloader = DataLoader(dataset=val_dataset,
                            batch_size=BATCH_SIZE)

train_N = len(train_dataloader.dataset)
valid_N = len(val_dataloader.dataset)

model = UNet(in_channels=1, num_classes=NUM_CLASSES).to(device)

if args.checkpoint == None:
    model.load_deeplab_weights()
else:
    checkpoint = torch.load(args.checkpoint, weights_only=False)
    model.load_state_dict(checkpoint)

optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=0.0001)
loss_function = nn.CrossEntropyLoss()
dice = DiceLoss(NUM_CLASSES)
writer = SummaryWriter(SUMMARY_LOGS)

iter_num = 0

if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs")
    model = nn.DataParallel(model)

def train():
    global iter_num
    model.train()
    train_running_loss = 0
    dice_running_loss = 0
    dice_loss = 0
    
    for idx, img_mask in enumerate(tqdm(train_dataloader)):
        # get image/label batch
        img = img_mask['image'].float().to(device)
        mask = img_mask['label'].long().to(device)
        
        assert img.shape[1] == model.n_channels, \
                    f'Network has been defined with {model.n_channels} input channels, ' \
                    f'but loaded images have {img.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'
        
        img = img.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
        
        # model prediction
        y_pred = model(img)
        
        if model.n_classes == 1:
            batch_loss = loss_function(y_pred.squeeze(1), mask.float())
            dice_loss = dice(F.sigmoid(y_pred), mask.float())
            batch_loss += dice_loss
        else:
            batch_loss = loss_function(y_pred, mask)
            dice_loss = dice(F.softmax(y_pred, dim=1), mask)
            batch_loss += dice_loss
        
        train_running_loss += batch_loss.item()
        
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

        lr_ = LEARNING_RATE * max(0.0, (1.0 - iter_num / (EPOCHS * len(train_dataloader)))) ** 0.9
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_

        iter_num += 1
        writer.add_scalar('info/lr', lr_, iter_num)
        writer.add_scalar('info/total_loss', batch_loss, iter_num)
        writer.add_scalar('info/dice', dice_loss, iter_num)
        
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
    global iter_num
    model.eval()
    val_running_loss = 0
    dice_running_loss = 0
    dice_loss = 0
    with torch.no_grad():
        for idx, img_mask in enumerate(tqdm(val_dataloader)):
            # get image/label batch
            img = img_mask['image'].float().to(device)
            mask = img_mask['label'].long().to(device)

            # image to tensor
            img = img.unsqueeze(1)
            
            # model predictions
            y_pred = model(img)

            # loss calculations
            if model.n_classes == 1:
                batch_loss = loss_function(y_pred.squeeze(1), mask.float())
                dice_loss = dice(F.sigmoid(y_pred), mask.float())
                batch_loss += dice_loss
            else:
                batch_loss = loss_function(y_pred, mask)
                dice_loss = dice(F.softmax(y_pred, dim=1), mask)
                batch_loss += dice_loss

            val_running_loss += batch_loss.item()

            iter_num += 1
            writer.add_scalar('info/dice', dice_loss, iter_num)
            writer.add_scalar('info/loss_ce', batch_loss, iter_num)
            
            if iter_num%2 == 0:
                image = img[1, 0:1, :, :]
                image = (image - image.min()) / (image.max() - image.min())
                writer.add_image('valid/Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(y_pred, dim=1), dim=1, keepdim=True)
                writer.add_image('valid/Prediction', outputs[1, ...] * 50, iter_num)
                labs = mask[1, ...].unsqueeze(0) * 50
                writer.add_image('valid/GroundTruth', labs, iter_num)

        val_running_loss = val_running_loss/len(val_dataloader)
        
        print(f"Iteration: {iter_num} Valid Loss: {val_running_loss:.4f} Dice Loss: {dice_loss:.4f}")

if __name__ == "__main__":
    # Initialize wandb
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        config={
            "learning_rate": LEARNING_RATE,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "model_name": MODEL_NAME,
            "kfold": args.kfold,
            "architecture": "UNet",
            "num_classes": NUM_CLASSES,
            "optimizer": "SGD",
            "loss": "CrossEntropyLoss + DiceLoss"
        }
    )
    
    # Log model architecture
    wandb.watch(model, log="all")
    
    for epoch in tqdm(range(EPOCHS)):
        print(f'Epoch: {epoch}')
        train()
        validate()

        if (epoch+1) % 50 == 0:
            # Save model checkpoint
            checkpoint_path = os.path.join(MODEL_SAVE_PATH, f"{MODEL_NAME}_{args.kfold}_{epoch}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            # Log model checkpoint to wandb
            wandb.save(checkpoint_path)

    # Save final model
    final_model_path = os.path.join(MODEL_SAVE_PATH, f"{MODEL_NAME}_{args.kfold}_final.pth")
    torch.save(model.state_dict(), final_model_path)
    wandb.save(final_model_path)
    
    # Close wandb run
    wandb.finish()
