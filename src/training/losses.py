"""
Loss functions for 6DoF pose estimation.

This module contains various loss functions for training pose estimation models,
including specialized losses for rotation, translation, and combined pose losses.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any


class PoseLoss(nn.Module):
    """
    Combined loss function for 6DoF pose estimation.
    
    This loss combines position and rotation components with optional
    weighting to train pose estimation models.
    
    Attributes:
        position_weight: Weight for position loss component
        rotation_weight: Weight for rotation loss component
        position_loss_fn: Loss function for position
        rotation_loss_fn: Loss function for rotation
        use_symmetry: Whether to use symmetry-aware rotation loss
        confidence_weight: Weight for confidence loss component
    """
    
    def __init__(
        self,
        position_weight: float = 1.0,
        rotation_weight: float = 1.0,
        position_loss_fn: str = 'l1',
        rotation_loss_fn: str = 'quaternion',
        use_symmetry: bool = True,
        confidence_weight: float = 0.1
    ):
        """
        Initialize the pose loss function.
        
        Args:
            position_weight: Weight for position loss component
            rotation_weight: Weight for rotation loss component
            position_loss_fn: Type of position loss ('l1', 'l2', 'smooth_l1')
            rotation_loss_fn: Type of rotation loss ('quaternion', 'matrix', 'geodesic')
            use_symmetry: Whether to use symmetry-aware rotation loss
            confidence_weight: Weight for confidence loss component
        """
        super(PoseLoss, self).__init__()
        
        self.position_weight = position_weight
        self.rotation_weight = rotation_weight
        self.position_loss_fn = position_loss_fn
        self.rotation_loss_fn = rotation_loss_fn
        self.use_symmetry = use_symmetry
        self.confidence_weight = confidence_weight
        
        # Initialize position loss function
        if position_loss_fn == 'l1':
            self.pos_criterion = nn.L1Loss(reduction='none')
        elif position_loss_fn == 'l2':
            self.pos_criterion = nn.MSELoss(reduction='none')
        elif position_loss_fn == 'smooth_l1':
            self.pos_criterion = nn.SmoothL1Loss(reduction='none')
        else:
            raise ValueError(f"Unsupported position loss: {position_loss_fn}")
    
    def forward(
        self,
        pred: Dict[str, torch.Tensor],
        target: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Calculate combined pose loss.
        
        Args:
            pred: Dictionary containing predicted position, rotation, and confidence
            target: Dictionary containing target position, rotation, and optional object IDs
            
        Returns:
            Tuple of (combined loss tensor, loss dictionary with components)
        """
        # Extract predictions and targets
        pred_position = pred['position']
        pred_rotation = pred['rotation']
        pred_confidence = pred.get('confidence', torch.ones_like(pred_position[:, 0:1]))
        
        target_position = target['position']
        target_rotation = target['rotation']
        object_ids = target.get('object_ids')
        
        # Calculate position loss
        position_loss = self._calculate_position_loss(pred_position, target_position)
        
        # Calculate rotation loss
        rotation_loss = self._calculate_rotation_loss(
            pred_rotation, target_rotation, object_ids
        )
        
        # Calculate confidence loss (optional)
        confidence_loss = torch.tensor(0.0, device=pred_position.device)
        if 'confidence' in pred:
            # Here we could implement a confidence-based weighting or regularization
            # For now, we just use a simple regularization to prevent overconfidence
            confidence_loss = -torch.log(pred_confidence + 1e-6).mean()
        
        # Combine losses with weights
        total_loss = (
            self.position_weight * position_loss +
            self.rotation_weight * rotation_loss +
            self.confidence_weight * confidence_loss
        )
        
        # Return both the total loss and individual components
        loss_dict = {
            'position_loss': position_loss.item(),
            'rotation_loss': rotation_loss.item(),
            'confidence_loss': confidence_loss.item()
        }
        
        return total_loss, loss_dict
    
    def _calculate_position_loss(
        self,
        pred_position: torch.Tensor,
        target_position: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate position loss.
        
        Args:
            pred_position: Predicted position tensor
            target_position: Target position tensor
            
        Returns:
            Position loss tensor
        """
        # Calculate position error
        position_error = self.pos_criterion(pred_position, target_position)
        
        # Return mean loss
        return position_error.mean()
    
    def _calculate_rotation_loss(
        self,
        pred_rotation: torch.Tensor,
        target_rotation: torch.Tensor,
        object_ids: Optional[List[str]] = None
    ) -> torch.Tensor:
        """
        Calculate rotation loss based on the specified loss function.
        
        Args:
            pred_rotation: Predicted rotation tensor
            target_rotation: Target rotation tensor
            object_ids: Optional list of object IDs for symmetry handling
            
        Returns:
            Rotation loss tensor
        """
        if self.rotation_loss_fn == 'quaternion':
            return self._quaternion_loss(pred_rotation, target_rotation, object_ids)
        elif self.rotation_loss_fn == 'matrix':
            return self._matrix_loss(pred_rotation, target_rotation, object_ids)
        elif self.rotation_loss_fn == 'geodesic':
            return self._geodesic_loss(pred_rotation, target_rotation, object_ids)
        else:
            raise ValueError(f"Unsupported rotation loss: {self.rotation_loss_fn}")
    
    def _quaternion_loss(
        self,
        pred_quat: torch.Tensor,
        target_quat: torch.Tensor,
        object_ids: Optional[List[str]] = None
    ) -> torch.Tensor:
        """
        Calculate quaternion-based rotation loss.
        
        This implements a negative dot product loss between quaternions,
        which is proportional to the angle between rotations.
        
        Args:
            pred_quat: Predicted quaternion tensor
            target_quat: Target quaternion tensor
            object_ids: Optional list of object IDs for symmetry handling
            
        Returns:
            Quaternion loss tensor
        """
        # Ensure quaternions are normalized
        pred_quat = F.normalize(pred_quat, p=2, dim=1)
        target_quat = F.normalize(target_quat, p=2, dim=1)
        
        # Calculate negative dot product (cosine distance)
        # Need to handle both q and -q representing the same rotation
        dot_product = torch.sum(pred_quat * target_quat, dim=1)
        loss = 1 - torch.abs(dot_product)
        
        # Handle symmetries if requested and object IDs are provided
        if self.use_symmetry and object_ids is not None:
            # This is a placeholder for symmetry handling
            # In a real implementation, we would adjust the loss based on
            # the known symmetries of each object type
            pass
        
        return loss.mean()
    
    def _matrix_loss(
        self,
        pred_matrix: torch.Tensor,
        target_matrix: torch.Tensor,
        object_ids: Optional[List[str]] = None
    ) -> torch.Tensor:
        """
        Calculate matrix-based rotation loss.
        
        This implements Frobenius norm between rotation matrices.
        
        Args:
            pred_matrix: Predicted rotation matrix tensor
            target_matrix: Target rotation matrix tensor
            object_ids: Optional list of object IDs for symmetry handling
            
        Returns:
            Matrix loss tensor
        """
        # Reshape to 3x3 if needed
        if pred_matrix.shape[-1] == 9:
            batch_size = pred_matrix.shape[0]
            pred_matrix = pred_matrix.view(batch_size, 3, 3)
            target_matrix = target_matrix.view(batch_size, 3, 3)
        
        # Calculate Frobenius norm
        matrix_diff = pred_matrix - target_matrix
        loss = torch.norm(matrix_diff, p='fro', dim=(1, 2))
        
        # Handle symmetries if requested and object IDs are provided
        if self.use_symmetry and object_ids is not None:
            # This is a placeholder for symmetry handling
            # In a real implementation, we would adjust the loss based on
            # the known symmetries of each object type
            pass
        
        return loss.mean()
    
    def _geodesic_loss(
        self,
        pred_rotation: torch.Tensor,
        target_rotation: torch.Tensor,
        object_ids: Optional[List[str]] = None
    ) -> torch.Tensor:
        """
        Calculate geodesic rotation loss.
        
        This computes the geodesic distance on the SO(3) manifold,
        which is the angle of the rotation that transforms one
        rotation to the other.
        
        Args:
            pred_rotation: Predicted rotation tensor (matrix or quaternion)
            target_rotation: Target rotation tensor (matrix or quaternion)
            object_ids: Optional list of object IDs for symmetry handling
            
        Returns:
            Geodesic loss tensor
        """
        # Assume inputs are 3x3 rotation matrices
        # Calculate R1 * R2^T, which gives the rotation between the two rotations
        batch_size = pred_rotation.shape[0]
        rel_rotation = torch.bmm(pred_rotation, target_rotation.transpose(1, 2))
        
        # Calculate the trace of the relative rotation matrix
        trace = torch.diagonal(rel_rotation, dim1=1, dim2=2).sum(1)
        
        # Calculate the rotation angle (geodesic distance)
        # theta = arccos((trace - 1) / 2)
        # Need to clamp the input to arccos to handle numerical issues
        cos_theta = (trace - 1) / 2
        cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
        theta = torch.acos(cos_theta)
        
        # Handle symmetries if requested and object IDs are provided
        if self.use_symmetry and object_ids is not None:
            # This is a placeholder for symmetry handling
            # In a real implementation, we would adjust the loss based on
            # the known symmetries of each object type
            pass
        
        return theta.mean()


class SymmetryAwarePoseLoss(PoseLoss):
    """
    Extended pose loss that explicitly handles object symmetries.
    
    This loss adapts the rotation component based on the symmetry
    properties of the object being detected.
    
    Attributes:
        symmetry_info: Dictionary mapping object IDs to symmetry information
    """
    
    def __init__(
        self,
        position_weight: float = 1.0,
        rotation_weight: float = 1.0,
        position_loss_fn: str = 'l1',
        rotation_loss_fn: str = 'quaternion',
        confidence_weight: float = 0.1,
        symmetry_info: Optional[Dict[str, Dict[str, Any]]] = None
    ):
        """
        Initialize the symmetry-aware pose loss.
        
        Args:
            position_weight: Weight for position loss component
            rotation_weight: Weight for rotation loss component
            position_loss_fn: Type of position loss ('l1', 'l2', 'smooth_l1')
            rotation_loss_fn: Type of rotation loss ('quaternion', 'matrix', 'geodesic')
            confidence_weight: Weight for confidence loss component
            symmetry_info: Dictionary mapping object IDs to symmetry information
        """
        super(SymmetryAwarePoseLoss, self).__init__(
            position_weight, rotation_weight, position_loss_fn,
            rotation_loss_fn, True, confidence_weight
        )
        
        self.symmetry_info = symmetry_info or {}
    
    def _get_symmetry_rotations(self, object_id: str) -> List[torch.Tensor]:
        """
        Get a list of valid rotations for the given object based on its symmetry.
        
        Args:
            object_id: ID of the object
            
        Returns:
            List of valid rotation matrices for the object
        """
        # Default to no symmetry (just identity)
        if object_id not in self.symmetry_info:
            return [torch.eye(3, device=self.device)]
        
        sym_type = self.symmetry_info[object_id].get('type', 'none')
        
        if sym_type == 'rotational':
            # Rotational symmetry around an axis
            axis = torch.tensor(self.symmetry_info[object_id].get('axis', [0, 0, 1]), 
                               device=self.device)
            num_steps = self.symmetry_info[object_id].get('num_steps', 1)
            
            # Generate rotation matrices for each step
            rotations = []
            for i in range(num_steps):
                angle = 2 * np.pi * i / num_steps
                rotations.append(self._axis_angle_to_matrix(axis, angle))
            
            return rotations
            
        elif sym_type == 'reflectional':
            # Reflectional symmetry across a plane
            normal = torch.tensor(self.symmetry_info[object_id].get('normal', [0, 0, 1]), 
                                 device=self.device)
            
            # Generate identity and reflection matrices
            reflection = torch.eye(3, device=self.device) - 2 * torch.outer(normal, normal)
            return [torch.eye(3, device=self.device), reflection]
            
        else:
            # No symmetry or unknown symmetry type
            return [torch.eye(3, device=self.device)]
    
    def _axis_angle_to_matrix(self, axis: torch.Tensor, angle: float) -> torch.Tensor:
        """
        Convert axis-angle representation to rotation matrix.
        
        Args:
            axis: Unit axis of rotation
            angle: Angle of rotation in radians
            
        Returns:
            Rotation matrix
        """
        # Normalize axis
        axis = F.normalize(axis, p=2, dim=0)
        
        # Create skew-symmetric matrix
        K = torch.zeros((3, 3), device=axis.device)
        K[0, 1] = -axis[2]
        K[0, 2] = axis[1]
        K[1, 0] = axis[2]
        K[1, 2] = -axis[0]
        K[2, 0] = -axis[1]
        K[2, 1] = axis[0]
        
        # Rodrigues formula: R = I + sin(θ) * K + (1 - cos(θ)) * K^2
        R = (torch.eye(3, device=axis.device) + 
             torch.sin(angle) * K + 
             (1 - torch.cos(angle)) * torch.mm(K, K))
        
        return R
    
    def _quaternion_loss_with_symmetry(
        self,
        pred_quat: torch.Tensor,
        target_quat: torch.Tensor,
        object_id: str
    ) -> torch.Tensor:
        """
        Calculate quaternion-based rotation loss with symmetry handling.
        
        Args:
            pred_quat: Predicted quaternion
            target_quat: Target quaternion
            object_id: Object ID for symmetry information
            
        Returns:
            Minimum loss considering all valid symmetric rotations
        """
        # Ensure quaternions are normalized
        pred_quat = F.normalize(pred_quat, p=2, dim=0)
        target_quat = F.normalize(target_quat, p=2, dim=0)
        
        # Get valid rotations based on symmetry
        valid_rots = self._get_symmetry_rotations(object_id)
        
        # Calculate loss for each valid rotation
        losses = []
        for rot_mat in valid_rots:
            # Convert rotation matrix to quaternion
            rot_quat = self._matrix_to_quaternion(rot_mat)
            
            # Apply symmetry rotation to target
            rotated_target = self._quaternion_multiply(target_quat, rot_quat)
            
            # Calculate negative dot product
            dot_product = torch.sum(pred_quat * rotated_target)
            loss = 1 - torch.abs(dot_product)
            losses.append(loss)
        
        # Return minimum loss among all valid rotations
        return torch.min(torch.stack(losses))
    
    def _matrix_to_quaternion(self, matrix: torch.Tensor) -> torch.Tensor:
        """
        Convert rotation matrix to quaternion.
        
        Args:
            matrix: 3x3 rotation matrix
            
        Returns:
            Quaternion [w, x, y, z]
        """
        trace = torch.trace(matrix)
        
        if trace > 0:
            S = torch.sqrt(trace + 1.0) * 2
            w = 0.25 * S
            x = (matrix[2, 1] - matrix[1, 2]) / S
            y = (matrix[0, 2] - matrix[2, 0]) / S
            z = (matrix[1, 0] - matrix[0, 1]) / S
        elif matrix[0, 0] > matrix[1, 1] and matrix[0, 0] > matrix[2, 2]:
            S = torch.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2
            w = (matrix[2, 1] - matrix[1, 2]) / S
            x = 0.25 * S
            y = (matrix[0, 1] + matrix[1, 0]) / S
            z = (matrix[0, 2] + matrix[2, 0]) / S
        elif matrix[1, 1] > matrix[2, 2]:
            S = torch.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2
            w = (matrix[0, 2] - matrix[2, 0]) / S
            x = (matrix[0, 1] + matrix[1, 0]) / S
            y = 0.25 * S
            z = (matrix[1, 2] + matrix[2, 1]) / S
        else:
            S = torch.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2
            w = (matrix[1, 0] - matrix[0, 1]) / S
            x = (matrix[0, 2] + matrix[2, 0]) / S
            y = (matrix[1, 2] + matrix[2, 1]) / S
            z = 0.25 * S
        
        return torch.tensor([w, x, y, z], device=matrix.device)
    
    def _quaternion_multiply(self, q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        """
        Multiply two quaternions.
        
        Args:
            q1: First quaternion [w, x, y, z]
            q2: Second quaternion [w, x, y, z]
            
        Returns:
            Result quaternion [w, x, y, z]
        """
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        
        return torch.tensor([w, x, y, z], device=q1.device)


class AdditiveMSSDLoss(nn.Module):
    """
    Additive Mean Squared Surface Distance (MSSD) loss for 6DoF pose estimation.
    
    This loss approximates the MSSD metric described in the BOP challenge,
    which measures the maximum distance between corresponding points on the
    object surface when transformed by the predicted and target poses.
    
    Attributes:
        point_clouds: Dictionary mapping object IDs to sampled point clouds
        diameters: Dictionary mapping object IDs to object diameters
        lambda_pos: Weight for position component
        lambda_rot: Weight for rotation component
        normalize: Whether to normalize by object diameter
    """
    
    def __init__(
        self,
        point_clouds: Dict[str, torch.Tensor],
        diameters: Dict[str, float],
        lambda_pos: float = 1.0,
        lambda_rot: float = 1.0,
        normalize: bool = True
    ):
        """
        Initialize the MSSD loss.
        
        Args:
            point_clouds: Dictionary mapping object IDs to sampled point clouds
            diameters: Dictionary mapping object IDs to object diameters
            lambda_pos: Weight for position component
            lambda_rot: Weight for rotation component
            normalize: Whether to normalize by object diameter
        """
        super(AdditiveMSSDLoss, self).__init__()
        
        self.point_clouds = point_clouds
        self.diameters = diameters
        self.lambda_pos = lambda_pos
        self.lambda_rot = lambda_rot
        self.normalize = normalize
        
        # Pre-compute max radius for each object
        self.max_radius = {
            obj_id: torch.max(torch.norm(points, dim=1))
            for obj_id, points in point_clouds.items()
        }
    
    def forward(
        self,
        pred: Dict[str, torch.Tensor],
        target: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Calculate MSSD loss.
        
        Args:
            pred: Dictionary containing predicted position, rotation
            target: Dictionary containing target position, rotation, object_ids
            
        Returns:
            Tuple of (loss tensor, loss dictionary with components)
        """
        # Extract predictions and targets
        pred_position = pred['position']
        pred_rotation = pred['rotation']
        
        target_position = target['position']
        target_rotation = target['rotation']
        object_ids = target['object_ids']
        
        batch_size = pred_position.shape[0]
        device = pred_position.device
        
        # Initialize loss components
        position_loss = torch.zeros(batch_size, device=device)
        rotation_loss = torch.zeros(batch_size, device=device)
        
        # Calculate loss for each sample in the batch
        for i in range(batch_size):
            obj_id = object_ids[i]
            
            # Get object point cloud and diameter
            if obj_id in self.point_clouds:
                points = self.point_clouds[obj_id].to(device)
                diameter = self.diameters[obj_id]
                max_radius = self.max_radius[obj_id].to(device)
            else:
                # Use default sphere if object not in dictionary
                # This is a fallback and should ideally never happen
                points = torch.randn(100, 3, device=device)
                points = F.normalize(points, p=2, dim=1)  # Unit sphere
                diameter = 1.0
                max_radius = torch.tensor(1.0, device=device)
            
            # Calculate position loss (direct L2 distance)
            pos_error = torch.norm(pred_position[i] - target_position[i])
            
            # Calculate rotation loss (approximate angular error effect on surface)
            if pred_rotation.shape[1] == 4:  # Quaternion
                # Convert quaternion to rotation matrix for simplicity
                pred_rot_matrix = self._quaternion_to_matrix(pred_rotation[i])
                target_rot_matrix = self._quaternion_to_matrix(target_rotation[i])
            else:  # Already rotation matrix
                pred_rot_matrix = pred_rotation[i].view(3, 3)
                target_rot_matrix = target_rotation[i].view(3, 3)
            
            # Approximate rotational error by max distance on sphere of radius = max_radius
            rel_rot = torch.mm(pred_rot_matrix, target_rot_matrix.t())
            trace = torch.trace(rel_rot)
            cos_theta = (trace - 1) / 2
            cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
            theta = torch.acos(cos_theta)
            rot_error = 2 * max_radius * torch.sin(theta / 2)
            
            # Normalize if requested
            if self.normalize:
                pos_error = pos_error / diameter
                rot_error = rot_error / diameter
            
            position_loss[i] = pos_error
            rotation_loss[i] = rot_error
        
        # Combine position and rotation components
        total_loss = self.lambda_pos * position_loss + self.lambda_rot * rotation_loss
        
        # Return both the mean loss and individual components
        return total_loss.mean(), {
            'position_loss': position_loss.mean().item(),
            'rotation_loss': rotation_loss.mean().item()
        }
    
    def _quaternion_to_matrix(self, quaternion: torch.Tensor) -> torch.Tensor:
        """
        Convert quaternion to rotation matrix.
        
        Args:
            quaternion: Quaternion tensor [w, x, y, z]
            
        Returns:
            3x3 rotation matrix
        """
        # Ensure quaternion is normalized
        quaternion = F.normalize(quaternion, p=2, dim=0)
        
        # Extract components
        w, x, y, z = quaternion.unbind()
        
        # Compute rotation matrix
        xx, yy, zz = x * x, y * y, z * z
        wx, wy, wz = w * x, w * y, w * z
        xy, xz, yz = x * y, x * z, y * z
        
        matrix = torch.stack([
            1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy),
            2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx),
            2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)
        ]).view(3, 3)
        
        return matrix


# Factory function to create losses
def create_loss_function(
    loss_type: str,
    config: Dict[str, Any]
) -> nn.Module:
    """
    Create a loss function based on the specified type.
    
    Args:
        loss_type: Type of loss function ('pose', 'symmetry_aware', 'mssd')
        config: Configuration parameters for the loss
        
    Returns:
        Initialized loss function
    """
    if loss_type == 'pose':
        return PoseLoss(
            position_weight=config.get('position_weight', 1.0),
            rotation_weight=config.get('rotation_weight', 1.0),
            position_loss_fn=config.get('position_loss_fn', 'l1'),
            rotation_loss_fn=config.get('rotation_loss_fn', 'quaternion'),
            use_symmetry=config.get('use_symmetry', False),
            confidence_weight=config.get('confidence_weight', 0.1)
        )
    elif loss_type == 'symmetry_aware':
        return SymmetryAwarePoseLoss(
            position_weight=config.get('position_weight', 1.0),
            rotation_weight=config.get('rotation_weight', 1.0),
            position_loss_fn=config.get('position_loss_fn', 'l1'),
            rotation_loss_fn=config.get('rotation_loss_fn', 'quaternion'),
            confidence_weight=config.get('confidence_weight', 0.1),
            symmetry_info=config.get('symmetry_info', {})
        )
    elif loss_type == 'mssd':
        return AdditiveMSSDLoss(
            point_clouds=config.get('point_clouds', {}),
            diameters=config.get('diameters', {}),
            lambda_pos=config.get('lambda_pos', 1.0),
            lambda_rot=config.get('lambda_rot', 1.0),
            normalize=config.get('normalize', True)
        )
    else:
        raise ValueError(f"Unsupported loss type: {loss_type}") 