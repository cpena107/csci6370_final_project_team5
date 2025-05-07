import torch
from torch.utils.data import DataLoader
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from dataset_final_project import Project_dataset as Project
from tqdm import tqdm
import numpy as np
from torchvision import transforms
import argparse
from scipy.spatial.distance import directed_hausdorff
import os
from pathlib import Path
import cv2

def save_visualization(original_img, true_mask, pred_mask, output_path):
    """
    Save prediction mask
    
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

def save_kfold_data(original_img, true_mask, pred_mask, slice_name, output_dir):
    """
    Save original image, ground truth mask, and prediction mask
    
    Args:
        original_img (numpy.ndarray): Original input image
        true_mask (numpy.ndarray): Ground truth mask
        pred_mask (numpy.ndarray): Predicted mask
        slice_name (str): Name of the slice
        output_dir (str): Directory to save the files
    """
    # Create subdirectories for each type
    img_dir = os.path.join(output_dir, 'images')
    gt_dir = os.path.join(output_dir, 'ground_truth')
    pred_dir = os.path.join(output_dir, 'predictions')
    
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)
    os.makedirs(pred_dir, exist_ok=True)
    
    # Convert to uint8 and scale to 0-255
    img = ((original_img - original_img.min()) / (original_img.max() - original_img.min() + 1e-7) * 255).astype(np.uint8)
    gt_mask = (true_mask * 255).astype(np.uint8)
    pred_mask = (pred_mask * 255).astype(np.uint8)
    
    # Save files
    cv2.imwrite(os.path.join(img_dir, f"{slice_name}.png"), img)
    cv2.imwrite(os.path.join(gt_dir, f"{slice_name}.png"), gt_mask)
    cv2.imwrite(os.path.join(pred_dir, f"{slice_name}.png"), pred_mask)

def calculate_average_hausdorff_distance(pred_mask, true_mask):
    """
    Calculate the Average Hausdorff Distance between predicted and ground truth masks
    
    Args:
        pred_mask (numpy.ndarray): Binary predicted mask
        true_mask (numpy.ndarray): Binary ground truth mask
    
    Returns:
        float: Average Hausdorff Distance
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
    
    # Calculate distances from each point in pred_boundary to true_boundary
    pred_to_true = np.array([min(np.linalg.norm(p - true_boundary, axis=1)) for p in pred_boundary])
    true_to_pred = np.array([min(np.linalg.norm(t - pred_boundary, axis=1)) for t in true_boundary])
    
    # Calculate average distances
    avg_pred_to_true = np.mean(pred_to_true)
    avg_true_to_pred = np.mean(true_to_pred)
    
    # Return the average of both directed distances
    return (avg_pred_to_true + avg_true_to_pred) / 2

def evaluate_model(model_path, kfold, device="cuda"):
    """
    Evaluate a trained TransUNet model using Dice coefficient and Average Hausdorff Distance
    
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
    kfold_data_dir = os.path.join("kfold_data", f"fold_{kfold}")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(kfold_data_dir, exist_ok=True)

    # Load validation dataset
    val_dataset = Project(image_dir=IMAGE_PATH, label_dir=LABEL_PATH, list_dir=VAL_PATH, transform=None)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE)

    # Initialize and load model
    config = CONFIGS_ViT_seg["R50-ViT-B_16"]
    config.n_classes = NUM_CLASSES
    config.n_skip = 3
    config.vit_name = "R50-ViT-B_16"
    config.patches.grid = (int(512 / 16), int(512 / 16))  # Add grid configuration
    model = ViT_seg(config, img_size=512, num_classes=NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    dice_scores = []
    avg_hausdorff_distances = []
    empty_masks_count = 0
    total_masks = 0
    
    with torch.no_grad():
        for idx, img_mask in enumerate(tqdm(val_dataloader)):
            # Get image/label batch
            img = img_mask['image'].float().to(device)
            mask = img_mask['label'].to(device)

            # Normalize image to [0, 1] range
            img = (img - img.min()) / (img.max() - img.min() + 1e-7)

            # print(f"img: {img.shape}, {img.min()}, {img.max()}")
            # print(f"mask: {mask.shape}, {mask.min()}, {mask.max()}")
            
            # Model prediction
            outputs = model(img)
            
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
            
            avg_hausdorff = calculate_average_hausdorff_distance(pred_np[0], mask_np[0])
            avg_hausdorff_distances.append(avg_hausdorff)

            # Save visualization
            slice_name = img_mask['case_name'][0]  # Get original slice name
            output_path = os.path.join(output_dir, f"{slice_name}.png")
            save_visualization(img_np[0, 0], mask_np[0], pred_np[0], output_path)
            
            # Save kfold data
            save_kfold_data(img_np[0, 0], mask_np[0], pred_np[0], slice_name, kfold_data_dir)

    avg_dice = np.mean(dice_scores)
    avg_hausdorff = np.mean(avg_hausdorff_distances)
    print(f"\nEvaluation Summary:")
    print(f"Average Dice Coefficient: {avg_dice:.4f}")
    print(f"Average Hausdorff Distance: {avg_hausdorff:.4f}")
    print(f"Total images processed: {total_masks}")
    print(f"Images with empty masks: {empty_masks_count}")
    print(f"Visualizations saved in: {output_dir}")
    
    return avg_dice, avg_hausdorff

if __name__ == "__main__":

    args = argparse.ArgumentParser()
    args.add_argument("--model_path", type=str, default="models/transunet_final_project/transunet_final_project_0_final.pth")
    args.add_argument("--kfold", type=str, default="0")
    args.add_argument("--device", type=str, default="cuda")
    args = args.parse_args()

    dice_score, avg_hausdorff = evaluate_model(args.model_path, args.kfold, args.device)