import torch
from torch.utils.data import DataLoader
from UNet_vanilla import UNet
from dataset_final_project import Project_dataset as Project
from tqdm import tqdm
import numpy as np
from torchvision import transforms
from dataset_final_project import RandomGenerator
import argparse
from scipy.spatial.distance import directed_hausdorff
import os
from pathlib import Path
import cv2

def save_visualization(original_img, true_mask, pred_mask, output_path):
    """
    Save only the prediction mask
    
    Args:
        original_img (numpy.ndarray): Original input image (not used)
        true_mask (numpy.ndarray): Ground truth mask (not used)
        pred_mask (numpy.ndarray): Predicted mask
        output_path (str): Path to save the visualization
    """
    # Convert prediction mask to uint8
    pred_mask = pred_mask.astype(np.uint8) * 255
    
    # Save the prediction mask
    cv2.imwrite(output_path, pred_mask)

def calculate_hausdorff_distance(pred_mask, true_mask):
    """
    Calculate the Hausdorff Distance between predicted and ground truth masks
    
    Args:
        pred_mask (numpy.ndarray): Binary predicted mask
        true_mask (numpy.ndarray): Binary ground truth mask
    
    Returns:
        float: Hausdorff Distance
    """
    # Convert to binary masks
    pred_mask = pred_mask.astype(bool)
    true_mask = true_mask.astype(bool)
    
    # Get boundary points
    pred_boundary = np.argwhere(pred_mask)
    true_boundary = np.argwhere(true_mask)
    
    # Check if either mask is empty
    if len(pred_boundary) == 0 or len(true_boundary) == 0:
        # If both masks are empty, return 0 (perfect match)
        if len(pred_boundary) == 0 and len(true_boundary) == 0:
            return 0.0
        # If only one mask is empty, return a large but finite value
        return 1000.0
    
    # Calculate directed Hausdorff distances
    d1 = directed_hausdorff(pred_boundary, true_boundary)[0]
    d2 = directed_hausdorff(true_boundary, pred_boundary)[0]
    
    # Return the maximum of the two directed distances
    return max(d1, d2)

def evaluate_model(model_path, kfold, device="cuda"):
    """
    Evaluate a trained UNet model using Dice coefficient and Hausdorff Distance on validation dataset
    
    Args:
        model_path (str): Path to the saved model weights
        kfold (str): K-fold cross validation number
        device (str): Device to run evaluation on ('cuda' or 'cpu')
    
    Returns:
        tuple: (Average Dice coefficient, Average Hausdorff Distance) across validation set
    """
    # Initialize paths and parameters
    IMAGE_PATH = "../data/slices/imgs"
    LABEL_PATH = "../data/slices/masks" 
    VAL_PATH = "../data/slices/valid_" + kfold + ".txt"
    NUM_CLASSES = 2
    BATCH_SIZE = 1

    # Create output directory for predictions
    model_name = Path(model_path).stem
    output_dir = os.path.join("predictions", model_name)
    os.makedirs(output_dir, exist_ok=True)

    # Load validation dataset
    val_dataset = Project(image_dir=IMAGE_PATH, label_dir=LABEL_PATH, list_dir=VAL_PATH, transform=transforms.Compose(
                                   [RandomGenerator(output_size=[512, 512])]))
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE)

    # Initialize and load model
    model = UNet(in_channels=1, num_classes=NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    dice_scores = []
    hausdorff_distances = []
    empty_masks_count = 0
    total_masks = 0
    
    with torch.no_grad():
        for idx, img_mask in enumerate(tqdm(val_dataloader)):
            # Get image/label batch
            img = img_mask['image'].float().to(device)
            mask = img_mask['label'].to(device)

            # print(f"img: {img.shape}, {img.min()}, {img.max()}")
            # print(f"mask: {mask.shape}, {mask.min()}, {mask.max()}")
            
            # Model prediction
            outputs = model(img)

            # Normalize image to [0, 1] range
            img = (img - img.min()) / (img.max() - img.min() + 1e-7)
            
            # Convert logits to predictions
            predictions = torch.argmax(torch.softmax(outputs, dim=1), dim=1)
            
            # Calculate Dice coefficient
            intersection = torch.sum(predictions * mask)
            union = torch.sum(predictions) + torch.sum(mask)
            dice = (2.0 * intersection) / (union + 1e-7)  # Add small epsilon to avoid division by zero
            dice_scores.append(dice.item())
            
            # Calculate Hausdorff Distance
            pred_np = predictions.cpu().numpy()
            mask_np = mask.cpu().numpy()
            img_np = img.cpu().numpy()
            
            # Check for empty masks
            total_masks += 1
            if torch.sum(predictions) == 0 or torch.sum(mask) == 0:
                empty_masks_count += 1
                print(f"Empty mask detected in image {idx}:")
                print(f"Prediction sum: {torch.sum(predictions).item()}")
                print(f"Ground truth sum: {torch.sum(mask).item()}")
            
            hausdorff = calculate_hausdorff_distance(pred_np[0], mask_np[0])
            hausdorff_distances.append(hausdorff)

            # Save visualization
            slice_name = img_mask['case_name'][0]  # Get original slice name
            output_path = os.path.join(output_dir, f"{slice_name}.png")
            save_visualization(img_np[0, 0], mask_np[0], pred_np[0], output_path)

    avg_dice = np.mean(dice_scores)
    avg_hausdorff = np.mean(hausdorff_distances)
    print(f"\nEvaluation Summary:")
    print(f"Average Dice Coefficient: {avg_dice:.4f}")
    print(f"Average Hausdorff Distance: {avg_hausdorff:.4f}")
    print(f"Total images processed: {total_masks}")
    print(f"Images with empty masks: {empty_masks_count}")
    print(f"Visualizations saved in: {output_dir}")
    
    return avg_dice, avg_hausdorff

if __name__ == "__main__":

    args = argparse.ArgumentParser()
    args.add_argument("--model_path", type=str, default="models/unet_final_project_vanilla_logits/unet_final_project_vanilla_logits_0_99.pth")
    args.add_argument("--kfold", type=str, default="0")
    args.add_argument("--device", type=str, default="cuda")
    args = args.parse_args()

    dice_score, hausdorff_distance = evaluate_model(args.model_path, args.kfold, args.device)