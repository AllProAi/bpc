"""
Evaluation metrics for 6DoF pose estimation.

This module implements metrics used in the BOP Challenge, specifically:
- MSSD: Maximum Symmetry-Aware Surface Distance
- mAP: mean Average Precision based on MSSD thresholds

References:
- BOP Challenge metrics: https://bop.felk.cvut.cz/challenges/bop-challenge-2022/#evaluation
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from scipy.spatial.transform import Rotation as R


def calculate_mssd(
    pred_pose: np.ndarray,
    gt_pose: np.ndarray,
    model_points: np.ndarray,
    diameter: Optional[float] = None,
    symmetric: bool = False,
    symmetry_transforms: Optional[List[np.ndarray]] = None,
    normalized: bool = False
) -> float:
    """
    Calculate Maximum Symmetry-Aware Surface Distance (MSSD) between predicted and ground truth poses.

    Args:
        pred_pose: Predicted pose as a 4x4 homogeneous transformation matrix
        gt_pose: Ground truth pose as a 4x4 homogeneous transformation matrix
        model_points: 3D points sampled from the object model surface (Nx3)
        diameter: Object diameter (used for normalization), defaults to None
        symmetric: Whether the object has symmetries
        symmetry_transforms: List of symmetry transformation matrices (each 4x4)
        normalized: Whether to normalize the distance by object diameter

    Returns:
        MSSD value (normalized if requested)
    """
    # Extract rotation and translation
    pred_R = pred_pose[:3, :3]
    pred_t = pred_pose[:3, 3]
    gt_R = gt_pose[:3, :3]
    gt_t = gt_pose[:3, 3]

    # Transform model points by predicted and ground truth poses
    pred_points = np.dot(model_points, pred_R.T) + pred_t
    gt_points = np.dot(model_points, gt_R.T) + gt_t

    # Calculate point-wise distances
    distances = np.linalg.norm(pred_points - gt_points, axis=1)
    
    # For symmetric objects, consider all valid symmetric transformations
    if symmetric and symmetry_transforms is not None:
        min_distances = distances.copy()
        
        for transform in symmetry_transforms:
            # Apply symmetry transformation to ground truth
            sym_R = transform[:3, :3]
            sym_t = transform[:3, 3]
            sym_gt_R = np.dot(sym_R, gt_R)
            sym_gt_t = np.dot(sym_R, gt_t) + sym_t
            
            # Transform model points by symmetry-adjusted ground truth pose
            sym_gt_points = np.dot(model_points, sym_gt_R.T) + sym_gt_t
            
            # Calculate distances and update minimum
            sym_distances = np.linalg.norm(pred_points - sym_gt_points, axis=1)
            min_distances = np.minimum(min_distances, sym_distances)
        
        distances = min_distances

    # Get maximum distance (MSSD)
    mssd = np.max(distances)
    
    # Normalize if requested and diameter is provided
    if normalized and diameter is not None and diameter > 0:
        mssd /= diameter
    
    return mssd


def calculate_mAP(
    predictions: List[Dict[str, Any]],
    ground_truth: List[Dict[str, Any]],
    model_points_dict: Dict[str, np.ndarray],
    diameters_dict: Dict[str, float],
    symmetry_info: Dict[str, Dict[str, Any]],
    mssd_thresholds: Optional[List[float]] = None,
    normalized: bool = True
) -> Dict[str, float]:
    """
    Calculate mean Average Precision (mAP) based on MSSD thresholds.

    Args:
        predictions: List of predicted poses (each with 'pose', 'object_id', 'scene_id', 'confidence')
        ground_truth: List of ground truth poses (each with 'pose', 'object_id', 'scene_id')
        model_points_dict: Dictionary mapping object IDs to sampled model points
        diameters_dict: Dictionary mapping object IDs to object diameters
        symmetry_info: Dictionary mapping object IDs to symmetry information
        mssd_thresholds: MSSD thresholds for AP calculation, defaults to [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
        normalized: Whether to use normalized MSSD

    Returns:
        Dictionary with 'mAP' and individual APs at different thresholds
    """
    if mssd_thresholds is None:
        mssd_thresholds = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
    
    # Group ground truth by scene
    gt_by_scene = {}
    for gt in ground_truth:
        scene_id = gt['scene_id']
        if scene_id not in gt_by_scene:
            gt_by_scene[scene_id] = []
        gt_by_scene[scene_id].append(gt)
    
    # Sort predictions by confidence (descending)
    predictions = sorted(predictions, key=lambda x: x.get('confidence', 0.0), reverse=True)
    
    # Initialize counters for precision calculation
    tp = np.zeros(len(predictions))
    fp = np.zeros(len(predictions))
    
    # For each threshold, compute precision and recall
    results = {}
    
    # Track already matched ground truths for each scene and threshold
    matched_gt_indices = {threshold: {scene_id: set() for scene_id in gt_by_scene} 
                          for threshold in mssd_thresholds}
    
    # Check each prediction
    for i, pred in enumerate(predictions):
        scene_id = pred['scene_id']
        object_id = pred['object_id']
        pred_pose = pred['pose']
        
        # Skip if no ground truth for this scene
        if scene_id not in gt_by_scene:
            fp[i] = 1
            continue
        
        # Get model points and diameter for this object
        if object_id not in model_points_dict:
            fp[i] = 1
            continue
            
        model_points = model_points_dict[object_id]
        diameter = diameters_dict.get(object_id, 1.0)
        
        # Get symmetry information
        symmetric = object_id in symmetry_info
        symmetry_transforms = None
        if symmetric:
            symmetry_transforms = _get_symmetry_transforms(symmetry_info[object_id])
        
        # Find matching ground truth for this prediction
        gt_match_found = False
        gt_match_idx = -1
        min_mssd = float('inf')
        
        for j, gt in enumerate(gt_by_scene[scene_id]):
            # Skip if object IDs don't match
            if gt['object_id'] != object_id:
                continue
                
            # Calculate MSSD
            mssd = calculate_mssd(
                pred_pose, gt['pose'], model_points, diameter,
                symmetric, symmetry_transforms, normalized
            )
            
            # Keep track of minimum MSSD
            if mssd < min_mssd:
                min_mssd = mssd
                gt_match_idx = j
        
        # Check if match is valid for each threshold
        for threshold in mssd_thresholds:
            if min_mssd <= threshold and gt_match_idx >= 0:
                # Check if this ground truth was already matched for this threshold
                if gt_match_idx not in matched_gt_indices[threshold][scene_id]:
                    matched_gt_indices[threshold][scene_id].add(gt_match_idx)
                    # This is a true positive for this threshold
                    if f'tp_{threshold}' not in results:
                        results[f'tp_{threshold}'] = np.zeros(len(predictions))
                    results[f'tp_{threshold}'][i] = 1
                else:
                    # Ground truth already matched, this is a false positive
                    if f'fp_{threshold}' not in results:
                        results[f'fp_{threshold}'] = np.zeros(len(predictions))
                    results[f'fp_{threshold}'][i] = 1
            else:
                # No match or match above threshold, this is a false positive
                if f'fp_{threshold}' not in results:
                    results[f'fp_{threshold}'] = np.zeros(len(predictions))
                results[f'fp_{threshold}'][i] = 1
    
    # Calculate precision and recall for each threshold
    total_gt = sum(len(gts) for gts in gt_by_scene.values())
    
    for threshold in mssd_thresholds:
        tp_key = f'tp_{threshold}'
        fp_key = f'fp_{threshold}'
        
        if tp_key not in results or fp_key not in results:
            results[f'AP_{threshold}'] = 0.0
            continue
            
        # Accumulate TP and FP
        acc_tp = np.cumsum(results[tp_key])
        acc_fp = np.cumsum(results[fp_key])
        
        # Calculate precision and recall
        precision = acc_tp / (acc_tp + acc_fp)
        recall = acc_tp / total_gt if total_gt > 0 else np.zeros_like(acc_tp)
        
        # Calculate AP using VOC method (all-point interpolation)
        ap = _calculate_ap_voc(precision, recall)
        results[f'AP_{threshold}'] = ap
    
    # Calculate mAP
    ap_values = [results[f'AP_{threshold}'] for threshold in mssd_thresholds]
    results['mAP'] = np.mean(ap_values) if ap_values else 0.0
    
    return results


def _calculate_ap_voc(precision: np.ndarray, recall: np.ndarray) -> float:
    """
    Calculate Average Precision using VOC protocol (all-point interpolation).

    Args:
        precision: Precision values at each detection
        recall: Recall values at each detection

    Returns:
        Average Precision value
    """
    # Add sentinel values for beginning and end of list
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    
    # Compute maximum precision for recall values
    for i in range(mpre.size - 1, 0, -1):
        mpre[i-1] = max(mpre[i-1], mpre[i])
    
    # Find recall value changes
    i = np.where(mrec[1:] != mrec[:-1])[0]
    
    # Sum area under curve
    ap = np.sum((mrec[i+1] - mrec[i]) * mpre[i+1])
    
    return ap


def _get_symmetry_transforms(symmetry_info: Dict[str, Any]) -> List[np.ndarray]:
    """
    Generate transformation matrices for object symmetries.

    Args:
        symmetry_info: Dictionary with symmetry information

    Returns:
        List of 4x4 transformation matrices
    """
    transforms = [np.eye(4)]  # Identity transformation (no change)
    sym_type = symmetry_info.get('type', 'none')
    
    if sym_type == 'rotational':
        # Rotational symmetry around an axis
        axis = np.array(symmetry_info.get('axis', [0, 0, 1]))
        num_steps = symmetry_info.get('num_steps', 1)
        
        # Generate rotation matrices
        for i in range(1, num_steps):  # Skip identity (i=0)
            angle = 2 * np.pi * i / num_steps
            r = R.from_rotvec(angle * axis)
            rot_matrix = r.as_matrix()
            
            transform = np.eye(4)
            transform[:3, :3] = rot_matrix
            transforms.append(transform)
            
    elif sym_type == 'reflectional':
        # Reflectional symmetry
        normal = np.array(symmetry_info.get('normal', [0, 0, 1]))
        normal = normal / np.linalg.norm(normal)  # Normalize
        
        # Reflection matrix
        reflection = np.eye(3) - 2 * np.outer(normal, normal)
        
        transform = np.eye(4)
        transform[:3, :3] = reflection
        transforms.append(transform)
    
    return transforms


def calculate_batch_mssd(
    pred_poses: Dict[str, torch.Tensor],
    gt_poses: Dict[str, torch.Tensor],
    object_ids: List[str],
    model_points_dict: Dict[str, torch.Tensor],
    diameters_dict: Dict[str, float],
    symmetry_info: Dict[str, Dict[str, Any]],
    device: torch.device,
    normalized: bool = True
) -> torch.Tensor:
    """
    Calculate MSSD for a batch of predictions (PyTorch implementation).

    Args:
        pred_poses: Dictionary with predicted 'rotation' (Bx3x3 or Bx4) and 'translation' (Bx3)
        gt_poses: Dictionary with ground truth 'rotation' (Bx3x3 or Bx4) and 'translation' (Bx3)
        object_ids: List of object IDs for each sample in the batch
        model_points_dict: Dictionary mapping object IDs to sampled model points
        diameters_dict: Dictionary mapping object IDs to object diameters
        symmetry_info: Dictionary mapping object IDs to symmetry information
        device: Torch device to use
        normalized: Whether to normalize by object diameter

    Returns:
        Tensor of MSSD values for each sample in the batch
    """
    batch_size = pred_poses['translation'].shape[0]
    mssd_values = torch.zeros(batch_size, device=device)
    
    pred_rot = pred_poses['rotation']
    pred_trans = pred_poses['translation']
    gt_rot = gt_poses['rotation']
    gt_trans = gt_poses['translation']
    
    # Convert quaternions to rotation matrices if needed
    if pred_rot.shape[1] == 4:  # Quaternion
        pred_rot = _quaternion_to_matrix_batch(pred_rot)
    if gt_rot.shape[1] == 4:  # Quaternion
        gt_rot = _quaternion_to_matrix_batch(gt_rot)
    
    # Process each sample in the batch
    for i in range(batch_size):
        obj_id = object_ids[i]
        
        # Get model points for this object
        if obj_id in model_points_dict:
            points = model_points_dict[obj_id].to(device)
            diameter = diameters_dict.get(obj_id, 1.0)
        else:
            # Use default sphere if object not in dictionary
            points = torch.randn(100, 3, device=device)
            points = torch.nn.functional.normalize(points, p=2, dim=1)  # Unit sphere
            diameter = 1.0
        
        # Transform points by predicted and ground truth poses
        pred_points = torch.matmul(points, pred_rot[i].transpose(0, 1)) + pred_trans[i]
        gt_points = torch.matmul(points, gt_rot[i].transpose(0, 1)) + gt_trans[i]
        
        # Calculate point-wise distances
        distances = torch.norm(pred_points - gt_points, dim=1)
        
        # For symmetric objects, consider all valid symmetric transformations
        if obj_id in symmetry_info:
            min_distances = distances.clone()
            sym_transforms = _get_symmetry_transforms_torch(symmetry_info[obj_id], device)
            
            for transform in sym_transforms:
                # Apply symmetry transformation to ground truth
                sym_rot = transform[:3, :3]
                sym_trans = transform[:3, 3]
                sym_gt_rot = torch.matmul(sym_rot, gt_rot[i])
                sym_gt_trans = torch.matmul(sym_rot, gt_trans[i]) + sym_trans
                
                # Transform points by symmetry-adjusted ground truth pose
                sym_gt_points = torch.matmul(points, sym_gt_rot.transpose(0, 1)) + sym_gt_trans
                
                # Calculate distances and update minimum
                sym_distances = torch.norm(pred_points - sym_gt_points, dim=1)
                min_distances = torch.minimum(min_distances, sym_distances)
            
            distances = min_distances
        
        # Get maximum distance (MSSD)
        mssd = torch.max(distances)
        
        # Normalize if requested
        if normalized and diameter > 0:
            mssd /= diameter
        
        mssd_values[i] = mssd
    
    return mssd_values


def _quaternion_to_matrix_batch(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert batch of quaternions to rotation matrices.

    Args:
        quaternions: Batch of quaternions (Bx4) [w, x, y, z]

    Returns:
        Batch of rotation matrices (Bx3x3)
    """
    batch_size = quaternions.shape[0]
    
    # Ensure quaternions are normalized
    quaternions = torch.nn.functional.normalize(quaternions, p=2, dim=1)
    
    # Extract components
    w, x, y, z = quaternions[:, 0], quaternions[:, 1], quaternions[:, 2], quaternions[:, 3]
    
    # Pre-compute common terms
    xx, yy, zz = x * x, y * y, z * z
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z
    
    # Compute rotation matrix elements
    m00 = 1 - 2 * (yy + zz)
    m01 = 2 * (xy - wz)
    m02 = 2 * (xz + wy)
    
    m10 = 2 * (xy + wz)
    m11 = 1 - 2 * (xx + zz)
    m12 = 2 * (yz - wx)
    
    m20 = 2 * (xz - wy)
    m21 = 2 * (yz + wx)
    m22 = 1 - 2 * (xx + yy)
    
    # Stack into matrices
    matrices = torch.stack([
        torch.stack([m00, m01, m02], dim=1),
        torch.stack([m10, m11, m12], dim=1),
        torch.stack([m20, m21, m22], dim=1)
    ], dim=1)
    
    return matrices


def _get_symmetry_transforms_torch(symmetry_info: Dict[str, Any], device: torch.device) -> List[torch.Tensor]:
    """
    Generate transformation matrices for object symmetries (PyTorch version).

    Args:
        symmetry_info: Dictionary with symmetry information
        device: Torch device to use

    Returns:
        List of 4x4 transformation matrices (torch.Tensor)
    """
    transforms = [torch.eye(4, device=device)]  # Identity transformation
    sym_type = symmetry_info.get('type', 'none')
    
    if sym_type == 'rotational':
        # Rotational symmetry around an axis
        axis = torch.tensor(symmetry_info.get('axis', [0, 0, 1]), device=device)
        axis = torch.nn.functional.normalize(axis, p=2, dim=0)
        num_steps = symmetry_info.get('num_steps', 1)
        
        # Generate rotation matrices
        for i in range(1, num_steps):  # Skip identity (i=0)
            angle = 2 * np.pi * i / num_steps
            transform = torch.eye(4, device=device)
            transform[:3, :3] = _axis_angle_to_matrix_torch(axis, angle, device)
            transforms.append(transform)
            
    elif sym_type == 'reflectional':
        # Reflectional symmetry
        normal = torch.tensor(symmetry_info.get('normal', [0, 0, 1]), device=device)
        normal = torch.nn.functional.normalize(normal, p=2, dim=0)
        
        # Reflection matrix
        reflection = torch.eye(3, device=device) - 2 * torch.outer(normal, normal)
        
        transform = torch.eye(4, device=device)
        transform[:3, :3] = reflection
        transforms.append(transform)
    
    return transforms


def _axis_angle_to_matrix_torch(axis: torch.Tensor, angle: float, device: torch.device) -> torch.Tensor:
    """
    Convert axis-angle to rotation matrix in PyTorch.

    Args:
        axis: Unit axis of rotation
        angle: Angle of rotation in radians
        device: Torch device to use

    Returns:
        3x3 rotation matrix
    """
    # Create skew-symmetric matrix
    K = torch.zeros((3, 3), device=device)
    K[0, 1] = -axis[2]
    K[0, 2] = axis[1]
    K[1, 0] = axis[2]
    K[1, 2] = -axis[0]
    K[2, 0] = -axis[1]
    K[2, 1] = axis[0]
    
    # Rodrigues formula: R = I + sin(θ) * K + (1 - cos(θ)) * K^2
    rotation_matrix = (torch.eye(3, device=device) + 
                      torch.sin(angle) * K + 
                      (1 - torch.cos(angle)) * torch.matmul(K, K))
    
    return rotation_matrix 