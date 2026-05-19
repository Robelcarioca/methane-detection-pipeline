"""Spatial evaluation metrics for methane plume segmentation."""

import numpy as np
from scipy.ndimage import label

def calculate_iou(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculates Intersection over Union (IoU), also known as the Jaccard Index.
    Measures the exact overlap between the predicted and actual plume.
    """
    # Ensure inputs are binary (0 or 1)
    y_true = np.asarray(y_true, dtype=bool)
    y_pred = np.asarray(y_pred, dtype=bool)
    
    intersection = np.logical_and(y_true, y_pred).sum()
    union = np.logical_or(y_true, y_pred).sum()
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
        
    return float(intersection / union)

def calculate_dice(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculates the Dice Coefficient (F1 Score for spatial data).
    Similar to IoU but double-counts the intersection, making it slightly 
    more forgiving and heavily used in segmentation tasks.
    """
    y_true = np.asarray(y_true, dtype=bool)
    y_pred = np.asarray(y_pred, dtype=bool)
    
    intersection = np.logical_and(y_true, y_pred).sum()
    
    # Sum of all positive pixels in both arrays
    total_pixels = y_true.sum() + y_pred.sum()
    
    if total_pixels == 0:
        return 1.0
        
    return float((2. * intersection) / total_pixels)

def calculate_spatial_fdr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculates the False Discovery Rate (FDR).
    Answers: "Out of all the pixels the model claimed were methane, 
    what percentage were actually false alarms?"
    """
    y_true = np.asarray(y_true, dtype=bool)
    y_pred = np.asarray(y_pred, dtype=bool)
    
    false_positives = np.logical_and(np.logical_not(y_true), y_pred).sum()
    true_positives = np.logical_and(y_true, y_pred).sum()
    
    total_predicted_positive = false_positives + true_positives
    
    if total_predicted_positive == 0:
        return 0.0
        
    return float(false_positives / total_predicted_positive)

def calculate_object_metrics(y_true: np.ndarray, y_pred: np.ndarray, iou_threshold: float = 0.1) -> dict:
    """
    Calculates object-level Precision and Recall using Connected Components.
    A predicted plume is considered a "True Positive" if its IoU with a 
    ground truth plume is greater than the iou_threshold.
    """
    y_true = np.asarray(y_true, dtype=bool)
    y_pred = np.asarray(y_pred, dtype=bool)
    
    # label() groups touching True pixels into distinct numbered objects
    true_labels, num_true_objects = label(y_true)
    pred_labels, num_pred_objects = label(y_pred)
    
    if num_true_objects == 0 and num_pred_objects == 0:
        return {"object_recall": 1.0, "object_precision": 1.0, "f1_score": 1.0}
    if num_true_objects == 0:
        return {"object_recall": 1.0, "object_precision": 0.0, "f1_score": 0.0}
    if num_pred_objects == 0:
        return {"object_recall": 0.0, "object_precision": 1.0, "f1_score": 0.0}

    true_positives = 0
    
    # Check each predicted object against ground truth objects
    for pred_idx in range(1, num_pred_objects + 1):
        pred_mask = (pred_labels == pred_idx)
        
        # Find which true objects overlap with this prediction
        overlapping_true_indices = np.unique(true_labels[pred_mask])
        overlapping_true_indices = overlapping_true_indices[overlapping_true_indices != 0]
        
        for true_idx in overlapping_true_indices:
            true_mask = (true_labels == true_idx)
            
            # Calculate IoU for this specific object pair
            intersection = np.logical_and(pred_mask, true_mask).sum()
            union = np.logical_or(pred_mask, true_mask).sum()
            obj_iou = intersection / union if union > 0 else 0
            
            if obj_iou >= iou_threshold:
                true_positives += 1
                break # Move to next prediction once a match is found
                
    recall = true_positives / num_true_objects
    precision = true_positives / num_pred_objects if num_pred_objects > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "object_recall": float(recall),
        "object_precision": float(precision),
        "f1_score": float(f1)
    }

def evaluate_batch(y_true_batch: np.ndarray, p_pred_batch: np.ndarray, threshold: float = 0.5) -> dict:
    """
    Evaluates a batch of satellite tiles. Separates frames containing actual 
    methane plumes from empty background frames to prevent metric inflation.
    
    Args:
        y_true_batch: Ground truth binary masks, shape (B, H, W)
        p_pred_batch: Model probability outputs (0 to 1), shape (B, H, W)
        threshold: Cutoff to convert probabilities to binary predictions
    """
    # Apply threshold to Manish's probability maps
    y_pred_batch = (p_pred_batch >= threshold).astype(bool)
    y_true_batch = np.asarray(y_true_batch, dtype=bool)
    
    batch_size = y_true_batch.shape[0]
    
    # Tracking for Micro-averaging
    total_intersection = 0
    total_union = 0
    total_true_pixels = 0
    total_pred_pixels = 0
    
    # Tracking for empty/active separation
    active_frame_ious = []
    empty_frame_false_positive_rates = []
    
    for i in range(batch_size):
        y_true = y_true_batch[i]
        y_pred = y_pred_batch[i]
        
        has_plume = y_true.any()
        
        # Frame-level metrics
        if has_plume:
            # Reusing your existing calculate_iou function here
            iou = calculate_iou(y_true, y_pred)
            active_frame_ious.append(iou)
        else:
            # For empty frames, measure if the model hallucinated a plume
            false_pixels = y_pred.sum()
            total_pixels = y_pred.size
            empty_frame_false_positive_rates.append(false_pixels / total_pixels)
            
        # Accumulate for Micro-metrics across the whole batch
        total_intersection += np.logical_and(y_true, y_pred).sum()
        total_union += np.logical_or(y_true, y_pred).sum()
        total_true_pixels += y_true.sum()
        total_pred_pixels += y_pred.sum()
        
    # Final Calculations
    micro_iou = total_intersection / total_union if total_union > 0 else 1.0
    macro_active_iou = np.mean(active_frame_ious) if active_frame_ious else 0.0
    avg_background_noise = np.mean(empty_frame_false_positive_rates) if empty_frame_false_positive_rates else 0.0
    
    return {
        "micro_iou": float(micro_iou),
        "macro_active_iou": float(macro_active_iou),
        "background_noise_rate": float(avg_background_noise),
        "active_frames_count": len(active_frame_ious),
        "empty_frames_count": len(empty_frame_false_positive_rates)
    }

if __name__ == "__main__":
    import numpy as np
    from scipy.ndimage import label
    
    print("Generating synthetic MethaneSAT batch...")
    
    # 4 images, 64x64 pixels each
    BATCH_SIZE, H, W = 4, 64, 64
    
    y_true_batch = np.zeros((BATCH_SIZE, H, W), dtype=bool)
    p_pred_batch = np.zeros((BATCH_SIZE, H, W), dtype=float)
    
    # ---------------------------------------------------------
    # Frame 0: True Negative (Empty tile)
    # Model behaves well, outputting low background noise (0.1 to 0.3)
    # ---------------------------------------------------------
    p_pred_batch[0] = np.random.uniform(0.1, 0.3, (H, W))
    
    # ---------------------------------------------------------
    # Frame 1: Standard Plume (Good detection, slight offset)
    # Ground truth is a 20x20 square.
    # ---------------------------------------------------------
    y_true_batch[1, 20:40, 20:40] = 1
    # Model predicts it, but shifted by 5 pixels and with high confidence (0.6 - 0.9)
    p_pred_batch[1, 25:45, 25:45] = np.random.uniform(0.6, 0.9, (20, 20))
    
    # ---------------------------------------------------------
    # Frame 2: Multiple Plumes (Partial success + hallucination)
    # Ground truth has two separate 10x10 plumes.
    # ---------------------------------------------------------
    y_true_batch[2, 10:20, 10:20] = 1  # Plume A
    y_true_batch[2, 40:50, 40:50] = 1  # Plume B
    
    # Model detects Plume A perfectly
    p_pred_batch[2, 10:20, 10:20] = np.random.uniform(0.8, 0.99, (10, 10))
    # Model misses Plume B entirely (stays 0)
    # Model hallucinates a false positive elsewhere
    p_pred_batch[2, 5:15, 50:60] = np.random.uniform(0.7, 0.9, (10, 10))
    
    # ---------------------------------------------------------
    # Frame 3: False Positive (Empty tile, but model hallucinates)
    # ---------------------------------------------------------
    p_pred_batch[3, 30:40, 30:40] = np.random.uniform(0.6, 0.95, (10, 10))

    # --- RUN THE EVALUATION ---
    print("\n--- Running Batch Evaluator (Threshold = 0.5) ---")
    batch_results = evaluate_batch(y_true_batch, p_pred_batch, threshold=0.5)
    
    for key, value in batch_results.items():
        if isinstance(value, float):
            print(f"{key:25s}: {value:.4f}")
        else:
            print(f"{key:25s}: {value}")

    print("\n--- Running Object-Level Detection (Frame 2) ---")
    # Binarize Frame 2 at 0.5 threshold to test object metrics
    frame_2_pred_binary = (p_pred_batch[2] >= 0.5)
    object_results = calculate_object_metrics(y_true_batch[2], frame_2_pred_binary, iou_threshold=0.1)
    
    for key, value in object_results.items():
        print(f"{key:25s}: {value:.4f}")