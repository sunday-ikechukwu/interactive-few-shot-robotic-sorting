"""
Module 1: Read-World Perception & HUD(Head-Up Display) Interface

Purpose:
---------
This module provides the functionality for real-time camera capture via OpenCV with an on-screen Heads-Up Display (HUD) overlay. It enforces the on-demand capture constraint (Target ROI Box) to keep CPU usage minimal on a dual-core machine (Intel i5-6200U) without running continuous heavy inference. 

Key Components:
---------------
1. TArget ROI Box: A fixed, centered bounding box (280x280 pixels).
2. Dyanmic HUD States & Color Codes:
    - "READY" -> Green  (0, 255, 0 in BGR): Default inspection state.
    - "AMBIGUOUS" -> Yellow (0, 255, 255 in BGR): Robot asking for human clarification.
    - "NOVEL" -> Red    (0, 0, 255 in BGR): Out-of-distribution item detected.
3. Croppping & Color Space Conversion: 
    - OpenCV frane are BGR (Blue-Green-Red).
    - DINOv2 / PyTroch expects RGB (Red-Green-Blue) input.
    - This module cleanly crops the ROI and performs BGR -> RGB conversion.
"""

from typing import Tuple, Optional
import cv2
import numpy as np


class PerceptionModule:
    """ Manages webcam capture, HUD rendering, and on-demand ROI extraction."""

    # BGR color constants for OpenCv drawing
    COLORS = {
        "READY": (0, 255, 0),       # Green
        "AMBIGUOUS": (0, 255, 255), # Yellow
        "NOVEL": (0, 0, 255),       # Red
        "INFO": (255, 255, 255),    # White
        "SHADOW": (0, 0, 0),        # Black shadow for text readability
    }

    def __init__(self, camera_index: int = 0, roi_size: int = 280):
        """
        Initialize the perception module.

        Args:
            camera_index: Device index for OpenCV VideoCapture (0 is typically the default webcam).
            roi_size: Side length of the square Target ROI box in pixels (default: 280).
        """
        self.camera_index = camera_index
        self.roi_size = roi_size
        self.cap: Optional[cv2.VideoCapture] = None
        self.state: str = "READY"  # Default HUD state
        self.status_message: str = "Place object in box & press SPACE to inspect"

    def open_camera(self) -> bool:
        """
        Opens the camera stream using DirectShow on Windows for  fast initialization.

        Returns:
            True if camera  opens successfully, False otherwise.
        """

        self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            # Fallback to default backend if DirectShow fails
            self.cap = cv2.VideoCapture(self.camera_index)

        if not self.cap.isOpened():
            print(f"[PerceptionModule] ERROR: Unable to open camera with index {self.camera_index}.")
            return False

        # Set standard resolution (640x480 is lightweight and optimal for Core i5-6200U)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        print(f"[PerceptionModule] Camera {self.camera_index} initialized successfully.")
        return True

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Reads a single raw frame from the webcam.

        Returns:
            A tuple (success, frame) where 'success' is a boolean indicating if the frame was read successfully,
            and 'frame' is the captured image in BGR format (or None if unsuccessful).
        """
        if self.cap is None or not self.cap.isOpened():
            return False, None
        ret, frame = self.cap.read()
        return ret, frame

    def get_roi_coordinates(self, frame_shape: Tuple[int, ...]) -> Tuple[int, int, int, int]:
        """
        calculates the top-left (x1,y1) and bottom-right (x2,y2) coordinates of a centered square ROI box.

        Args: 
            frame_shape: Shape of the frame (height, width, channels).
        
        Returns:
            A tuple (x1, y1, x2, y2): Pixel coordinates of the ROI box.
        """

        height, width = frame_shape[:2]
        center_x, center_y = width // 2, height // 2
        half_roi = self.roi_size // 2

        x1 = max(0, center_x - half_roi)
        y1 = max(0, center_y - half_roi)
        x2 = min(width, center_x + half_roi)
        y2 = min(height, center_y + half_roi)

        return x1, y1, x2, y2

    def crop_roi(self, frame: np.ndarray, to_rgb: bool = True) -> np.ndarray:
        """
        Crops the Target ROI from the given frame and converts it from BGR to RGB.

        Args:
            frame: The input frame in BGR format from OpenCv.
            to_rgb: If True, converts the cropped ROI to RGB format.
        """
        x1, y1, x2, y2 = self.get_roi_coordinates(frame.shape)
        roi = frame[y1:y2, x1:x2].copy()

        if to_rgb:
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

        return roi

    def set_state(self, state: str, message: Optional[str] = None):
        """
        Updates the HUD state and optional status message.

        Args:
            state: One of 'READY', 'AMBIGUOUS', or 'NOVEL'.
            message: Informational text to display on the HUD.
        """
        if state in self.COLORS:
            self.state = state
        if message is not None:
            self.status_message = message

    def draw_hud(self, frame: np.ndarray) -> np.ndarray:
        """
        Draws the Target ROI bounding box, corner accents, and status text on a copy
        of the frame for visualization.

        Args:
            frame: Raw BGR frame.

        Returns:
            Annotated BGR frame ready for display in cv2.imshow.
        """
        display = frame.copy()
        x1, y1, x2, y2 = self.get_roi_coordinates(frame.shape)
        box_color = self.COLORS.get(self.state, self.COLORS["READY"])

        # 1. Draw central Target ROI Box
        cv2.rectangle(display, (x1, y1), (x2, y2), box_color, 2)

        # 2. Draw stylish corner brackets for a crisp robotics HUD look
        corner_len = 20
        thick = 3
        # Top-left corner
        cv2.line(display, (x1, y1), (x1 + corner_len, y1), box_color, thick)
        cv2.line(display, (x1, y1), (x1, y1 + corner_len), box_color, thick)
        # Top-right corner
        cv2.line(display, (x2, y1), (x2 - corner_len, y1), box_color, thick)
        cv2.line(display, (x2, y1), (x2, y1 + corner_len), box_color, thick)
        # Bottom-left corner
        cv2.line(display, (x1, y2), (x1 + corner_len, y2), box_color, thick)
        cv2.line(display, (x1, y2), (x1, y2 - corner_len), box_color, thick)
        # Bottom-right corner
        cv2.line(display, (x2, y2), (x2 - corner_len, y2), box_color, thick)
        cv2.line(display, (x2, y2), (x2, y2 - corner_len), box_color, thick)

        # 3. Draw Center Reticle / Crosshair (+ symbol)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        reticle_len = 10
        cv2.line(display, (cx - reticle_len, cy), (cx + reticle_len, cy), box_color, 1)
        cv2.line(display, (cx, cy - reticle_len), (cx, cy + reticle_len), box_color, 1)

        # 4. Draw Header Banner with State
        font = cv2.FONT_HERSHEY_SIMPLEX
        state_label = f"STATE: {self.state}"
        # Shadow for readability
        cv2.putText(display, state_label, (22, 37), font, 0.65, self.COLORS["SHADOW"], 3, cv2.LINE_AA)
        cv2.putText(display, state_label, (20, 35), font, 0.65, box_color, 2, cv2.LINE_AA)

        # 5. Draw Instruction / Status Message at bottom
        cv2.putText(display, self.status_message, (22, display.shape[0] - 18), font, 0.55, self.COLORS["SHADOW"], 2, cv2.LINE_AA)
        cv2.putText(display, self.status_message, (20, display.shape[0] - 20), font, 0.55, self.COLORS["INFO"], 2, cv2.LINE_AA)

        # 6. Helper prompt for testing controls
        controls_hint = "[SPACE]: Capture ROI | [1]: Ready | [2]: Ambiguous | [3]: Novel | [Q]: Quit"
        cv2.putText(display, controls_hint, (21, 62), font, 0.45, self.COLORS["SHADOW"], 2, cv2.LINE_AA)
        cv2.putText(display, controls_hint, (20, 60), font, 0.45, (200, 200, 200), 2, cv2.LINE_AA)

        return display

    def release(self):
        """Releases the camera hardware and closes any OpenCV windows."""
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()
        print("[PerceptionModule] Camera and windows released cleanly.")



def run_standalone_test():
    """
    Self-contained interactive test for perception.py.
    Allows user to verify camera stream, HUD states, and ROI cropping.
    """
    perception = PerceptionModule(camera_index=0, roi_size=280)
    if not perception.open_camera():
        print("[Test] Could not initialize webcam. Check camera connection or index.")
        return

    print("\n" + "=" * 60)
    print(" PERCEPTION MODULE TEST MODE")
    print("=" * 60)
    print(" - Place an object inside the central box.")
    print(" - Press [SPACE] to capture the ROI and inspect the cropped patch.")
    print(" - Press [1] for GREEN (READY state)")
    print(" - Press [2] for YELLOW (AMBIGUOUS state)")
    print(" - Press [3] for RED (NOVEL state)")
    print(" - Press [Q] to quit.")
    print("=" * 60 + "\n")

    captured_patch_bgr = None

    try:
        while True:
            ret, frame = perception.read_frame()
            if not ret or frame is None:
                print("[Test] Failed to grab frame. Exiting loop.")
                break

            # Render the HUD overlay
            annotated_frame = perception.draw_hud(frame)
            cv2.imshow("Robot Perception HUD", annotated_frame)

            # If user has captured an ROI, show it in a side-by-side inspection window
            if captured_patch_bgr is not None:
                cv2.imshow("Captured Target ROI (What DINOv2 Sees)", captured_patch_bgr)

            # Wait 1ms for key press
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q') or key == 27:  # 'q' or ESC
                print("[Test] Exit key pressed.")
                break
            elif key == 32:  # SPACEBAR
                # Crop ROI and store for display (in BGR for cv2.imshow)
                captured_patch_rgb = perception.crop_roi(frame, to_rgb=True)
                captured_patch_bgr = perception.crop_roi(frame, to_rgb=False)
                print(f"[Test] Captured ROI! Shape: {captured_patch_rgb.shape} (H x W x Channels). Ready for DINOv2.")
                perception.set_state(perception.state, "ROI Captured! Check the inspection window.")
            elif key == ord('1'):
                perception.set_state("READY", "State: Normal / Ready for inspection")
            elif key == ord('2'):
                perception.set_state("AMBIGUOUS", "State: Ambiguity Resolution Mode (HRI Active)")
            elif key == ord('3'):
                perception.set_state("NOVEL", "State: Novel Object Detected! Needs new bin")

    finally:
        perception.release()


if __name__ == "__main__":
    run_standalone_test()

