"""
Vision pipeline: camera capture, object detection via background subtraction.

Uses MuJoCo's offscreen renderer for image capture and OpenCV for
contour-based object localization.
"""

import numpy as np
import cv2
from typing import Optional, Tuple

from utils.constants import IMAGE_WIDTH, IMAGE_HEIGHT


class OverheadCamera:
    """
    Simulated overhead camera with background subtraction for object detection.
    """

    def __init__(self, robot):
        """
        Initialize camera with reference to robot (for rendering).

        Args:
            robot: KukaRobot instance for calling render_image().
        """
        self.robot = robot
        self.width = IMAGE_WIDTH
        self.height = IMAGE_HEIGHT
        self._background = None

        # Camera parameters for pixel->world mapping
        # Overhead camera at (0, 0, 2.0) looking down
        # Table is roughly centered at (0.3, 0, 0.8)
        # FOV is 45 degrees
        self._cam_height = 2.0
        self._table_z = 0.8
        self._fov_deg = 45.0

    def capture_rgb(self) -> np.ndarray:
        """
        Capture RGB image from overhead camera.

        Returns:
            RGB image array (H, W, 3), uint8.
        """
        return self.robot.render_image(width=self.width, height=self.height)

    def capture_depth(self) -> Optional[np.ndarray]:
        """
        Capture depth image (requires mujoco.Renderer with depth=True).
        For now, returns None as we use RGB-based detection.
        """
        # Can be implemented with mujoco.Renderer(..., depth=True)
        return None

    def compute_background(self, n_samples: int = 1) -> Optional[np.ndarray]:
        """
        Compute background model by averaging frames with no object.
        For simulation, we can just use a solid color or capture empty table.

        Args:
            n_samples: Number of samples to average (not used in sim).

        Returns:
            Background image (H, W, 3), uint8.
        """
        # In simulation, we can use a known background color or render
        # an empty scene. For simplicity, return None and use color thresholding.
        return None

    def detect_object(self, rgb: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        """
        Detect object in the scene using color thresholding + contours.

        Args:
            rgb: Pre-captured RGB image. If None, captures one.

        Returns:
            Object world position (3,) if found, else None.
        """
        if rgb is None:
            rgb = self.capture_rgb()

        # Convert to HSV for better color segmentation
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)

        # Object color range (red-orange: hue 0-20, sat>100, val>50)
        lower = np.array([0, 100, 50])
        upper = np.array([20, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)

        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None

        # Find largest contour (the object)
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)

        if area < 50:  # Minimum area threshold
            return None

        # Get centroid in pixel coordinates
        M = cv2.moments(largest)
        if M['m00'] == 0:
            return None

        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])

        # Convert to world coordinates
        world_pos = self.pixel_to_world(cx, cy)

        return world_pos

    def pixel_to_world(self, px: float, py: float, table_z: float = 0.8) -> np.ndarray:
        """
        Convert pixel coordinates to world coordinates on the table plane.

        Simple pinhole camera model for overhead camera.

        Args:
            px: Pixel x coordinate (0 to width).
            py: Pixel y coordinate (0 to height).
            table_z: Z height of the table plane.

        Returns:
            World position (3,) on the table plane.
        """
        # Camera is at (0, 0, 2.0), looking down (-Z)
        # Image plane: center is (0,0) in world, edges depend on FOV
        # FOV is vertical (y direction in image)

        # Normalize pixel coordinates to [-1, 1]
        nx = (px / self.width - 0.5) * 2.0
        ny = -(py / self.height - 0.5) * 2.0  # flip y (image y is down, world y is up)

        # Calculate world extents at table_z
        # Camera at z=2.0, table at z=0.8 -> distance = 1.2m
        # Vertical FOV = 45 deg -> half angle = 22.5 deg
        # tan(22.5) ≈ 0.414
        # World height at table = 2 * distance * tan(fov/2) = 2 * 1.2 * 0.414 ≈ 0.994
        # World width = height * aspect_ratio

        dist = self._cam_height - table_z
        fov_rad = np.deg2rad(self._fov_deg)
        half_h = dist * np.tan(fov_rad / 2.0)
        half_w = half_h * (self.width / self.height)

        # Map normalized coords to world
        wx = nx * half_w
        wy = ny * half_h
        wz = table_z

        return np.array([wx, wy, wz])

    def visualize_detection(self, rgb: np.ndarray, world_pos: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Draw detection results on the RGB image for debugging.

        Args:
            rgb: RGB image.
            world_pos: Detected world position (for info display).

        Returns:
            Annotated RGB image.
        """
        viz = rgb.copy()
        if world_pos is not None:
            # Draw crosshair at center (object should be near center if detected)
            h, w = viz.shape[:2]
            cv2.drawMarker(viz, (w // 2, h // 2), (0, 255, 0), cv2.MARKER_CROSS, 15, 2)
            cv2.putText(viz, f"Obj: {world_pos}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        return viz
