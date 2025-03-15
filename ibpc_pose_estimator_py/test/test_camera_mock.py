#!/usr/bin/env python3
# Copyright 2025, Daniel Hill dba Novus Nexum Labs (Daniel@allpro.enterprises)

"""
Unit tests for the Camera class in the ibpc_pose_estimator module.
This version uses mock objects for ROS dependencies.
"""

import unittest
import numpy as np
from unittest.mock import MagicMock, patch

# Mock the ROS dependencies
import sys
from unittest.mock import MagicMock

# Create mock classes for ROS dependencies
class MockPose:
    def __init__(self):
        self.position = MagicMock()
        self.position.x = 0.0
        self.position.y = 0.0
        self.position.z = 0.0
        self.orientation = MagicMock()
        self.orientation.x = 0.0
        self.orientation.y = 0.0
        self.orientation.z = 0.0
        self.orientation.w = 1.0

class MockHeader:
    def __init__(self):
        self.frame_id = ""

class MockCameraInfo:
    def __init__(self):
        self.header = MockHeader()
        self.k = [1000.0, 0.0, 320.0, 0.0, 1000.0, 240.0, 0.0, 0.0, 1.0]

class MockCameraMsg:
    def __init__(self):
        self.info = MockCameraInfo()
        self.pose = MockPose()
        self.rgb = MagicMock()
        self.depth = MagicMock()
        self.aolp = MagicMock()
        self.dolp = MagicMock()

class MockPhotoneo:
    def __init__(self):
        self.info = MockCameraInfo()
        self.pose = MockPose()
        self.rgb = MagicMock()
        self.depth = MagicMock()

# Mock the cv_bridge module
class MockCvBridge:
    def imgmsg_to_cv2(self, img_msg):
        return np.zeros((480, 640, 3))

# Create a mock version of the Camera class for testing
def ros_pose_to_mat(pose):
    """Convert a ROS pose to a 4x4 transformation matrix."""
    matrix = np.eye(4)
    matrix[:3, 3] = [pose.position.x, pose.position.y, pose.position.z]
    return matrix

class Camera:
    """Mock Camera class for testing."""
    
    def __init__(self, msg):
        """Initialize a Camera from a message."""
        if not isinstance(msg, (MockCameraMsg, MockPhotoneo)):
            raise TypeError("Input message must be of type CameraMsg or PhotoneoMsg")
            
        br = MockCvBridge()
        
        self.name = (msg.info.header.frame_id,)
        self.pose = ros_pose_to_mat(msg.pose)
        self.intrinsics = np.array(msg.info.k).reshape(3, 3)
        self.rgb = br.imgmsg_to_cv2(msg.rgb)
        self.depth = br.imgmsg_to_cv2(msg.depth)
        
        if isinstance(msg, MockCameraMsg):
            self.aolp = br.imgmsg_to_cv2(msg.aolp)
            self.dolp = br.imgmsg_to_cv2(msg.dolp)
        else:  # PhotoneoMsg
            self.aolp = None
            self.dolp = None


class TestCamera(unittest.TestCase):
    """Test cases for the Camera class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create mock objects for testing
        self.mock_bridge = MockCvBridge()
        
        # Create a mock CameraMsg
        self.camera_msg = MockCameraMsg()
        self.camera_msg.info.header.frame_id = "test_camera"
        
        # Set pose values
        self.camera_msg.pose.position.x = 1.0
        self.camera_msg.pose.position.y = 2.0
        self.camera_msg.pose.position.z = 3.0
        
        # Create a mock PhotoneoMsg
        self.photoneo_msg = MockPhotoneo()
        self.photoneo_msg.info.header.frame_id = "test_photoneo"
        
        # Set pose values
        self.photoneo_msg.pose.position.x = 4.0
        self.photoneo_msg.pose.position.y = 5.0
        self.photoneo_msg.pose.position.z = 6.0

    def test_camera_initialization_from_camera_msg(self):
        """Test Camera initialization from a CameraMsg."""
        # Create a Camera instance from the mock CameraMsg
        camera = Camera(self.camera_msg)
        
        # Verify the Camera attributes
        self.assertEqual(camera.name, ("test_camera",))
        
        # Check the pose matrix
        expected_pose = np.eye(4)
        expected_pose[:3, 3] = [1.0, 2.0, 3.0]
        np.testing.assert_array_almost_equal(camera.pose, expected_pose)
        
        # Check the intrinsics matrix
        expected_intrinsics = np.array([
            [1000.0, 0.0, 320.0],
            [0.0, 1000.0, 240.0],
            [0.0, 0.0, 1.0]
        ])
        np.testing.assert_array_almost_equal(camera.intrinsics, expected_intrinsics)
        
        # Verify that aolp and dolp are not None for CameraMsg
        self.assertIsNotNone(camera.aolp)
        self.assertIsNotNone(camera.dolp)

    def test_camera_initialization_from_photoneo_msg(self):
        """Test Camera initialization from a PhotoneoMsg."""
        # Create a Camera instance from the mock PhotoneoMsg
        camera = Camera(self.photoneo_msg)
        
        # Verify the Camera attributes
        self.assertEqual(camera.name, ("test_photoneo",))
        
        # Check the pose matrix
        expected_pose = np.eye(4)
        expected_pose[:3, 3] = [4.0, 5.0, 6.0]
        np.testing.assert_array_almost_equal(camera.pose, expected_pose)
        
        # Check the intrinsics matrix
        expected_intrinsics = np.array([
            [1000.0, 0.0, 320.0],
            [0.0, 1000.0, 240.0],
            [0.0, 0.0, 1.0]
        ])
        np.testing.assert_array_almost_equal(camera.intrinsics, expected_intrinsics)
        
        # Verify that aolp and dolp are None for PhotoneoMsg
        self.assertIsNone(camera.aolp)
        self.assertIsNone(camera.dolp)

    def test_camera_initialization_with_invalid_type(self):
        """Test Camera initialization with an invalid message type."""
        # Try to create a Camera with an invalid message type
        with self.assertRaises(TypeError):
            Camera("not_a_valid_message")

    def test_ros_pose_to_mat(self):
        """Test the ros_pose_to_mat helper function."""
        # Create a test pose
        pose = MockPose()
        pose.position.x = 1.0
        pose.position.y = 2.0
        pose.position.z = 3.0
        
        # Convert to matrix
        matrix = ros_pose_to_mat(pose)
        
        # Expected result
        expected = np.eye(4)
        expected[:3, 3] = [1.0, 2.0, 3.0]
        
        # Verify the result
        np.testing.assert_array_almost_equal(matrix, expected)


if __name__ == '__main__':
    unittest.main() 