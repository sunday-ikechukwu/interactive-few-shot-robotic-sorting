"""
Module 2: DINOv2 Feature Extractor

Purpose:
--------
Extracts dense, semantically rich 384-dimensional feature embeddings from cropped
RGB object images using Meta AI's pre-trained DINOv2-small Vision Transformer (ViT-S/14).

Hardware & Architectural Invariants:
-------------------------------------
1. CPU Inference: Runs strictly on Intel Core i5-6200U (pure PyTorch torch.no_grad()).
2. Frozen Backbone: Model weights are completely frozen (eval mode, requires_grad=False).
3. L2 Normalization: All feature vectors are normalized to unit length (||f||_2 = 1.0),
   turning cosine similarity into a fast, single dot-product operation:
       sim(f_1, f_2) = f_1 . f_2
"""

from typing import Union, Tuple
import time
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T


class DINOv2FeatureExtractor:
    """Extracts L2-normalized 384-dimensional embeddings using frozen DINOv2-small."""

    def __init__(self, model_name: str = "dinov2_vits14", device: str = "cpu"):
        """
        Initializes and loads the DINOv2-small backbone.

        Args:
            model_name: Torch Hub model identifier (default: 'dinov2_vits14', 21M params).
            device: 'cpu' for host machine execution.
        """
        self.device = torch.device(device)
        self.model_name = model_name
        self.feature_dim = 384  # DINOv2-small output embedding dimension

        print(f"[DINOv2FeatureExtractor] Loading {model_name} from PyTorch Hub (CPU mode)...")
        start_time = time.time()

        # Load pre-trained model from Meta AI's official GitHub repository
        # torch.hub caches the model locally after the first download (~85 MB)
        self.model = torch.hub.load("facebookresearch/dinov2", model_name)
        self.model.to(self.device)
        self.model.eval()  # Freeze BatchNorm / Dropout layers

        # Freeze all gradient computations to save memory and avoid autograd overhead
        for param in self.model.parameters():
            param.requires_grad = False

        load_latency = (time.time() - start_time) * 1000
        print(f"[DINOv2FeatureExtractor] Model loaded successfully in {load_latency:.1f} ms.")

        # Standard ImageNet pre-processing required by DINOv2:
        self.transform = T.Compose([
            T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

    def extract(self, image_input: Union[np.ndarray, Image.Image]) -> Tuple[np.ndarray, float]:
        """
        Extracts a normalized 384-dimensional feature vector from an input image.

        Args:
            image_input: Either an RGB NumPy array (from perception.crop_roi)
                         or a PIL Image instance.

        Returns:
            (feature_vector, latency_ms):
                - feature_vector: 1D NumPy array of shape (384,) with L2 norm = 1.0.
                - latency_ms: Forward pass inference time in milliseconds.
        """

        if isinstance(image_input, np.ndarray):
            pil_img = Image.fromarray(image_input)
        elif isinstance(image_input, Image.Image):
            pil_img = image_input
        else:
            raise TypeError(f"Expected np.ndarray or PIL.Image, got {type(image_input)}")

        if pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")

        # Shape becomes (3, 224, 224) -> Unsqueeze batch dim: (1, 3, 224, 224)
        input_tensor = self.transform(pil_img).unsqueeze(0).to(self.device)

        start_t = time.perf_counter()
        with torch.no_grad():
            raw_features = self.model(input_tensor)  # Tensor of shape (1, 384)

            # 3. L2 Normalization: f = v / ||v||_2
            # torch.nn.functional.normalize normalizes along dim=1 with p=2
            norm_features = torch.nn.functional.normalize(raw_features, p=2, dim=1)
        end_t = time.perf_counter()

        latency_ms = (end_t - start_t) * 1000

        feature_vector = norm_features.squeeze(0).cpu().numpy().astype(np.float32)

        return feature_vector, latency_ms

    @staticmethod
    def compute_cosine_similarity(feat_a: np.ndarray, feat_b: np.ndarray) -> float:
        """
        Computes cosine similarity between two unit-normalized vectors.
        Because ||feat_a||_2 = 1.0 and ||feat_b||_2 = 1.0:
            sim(a, b) = a . b

        Args:
            feat_a: 1D NumPy array of shape (384,).
            feat_b: 1D NumPy array of shape (384,).

        Returns:
            Cosine similarity scalar in range [-1.0, 1.0].
        """
        return float(np.dot(feat_a, feat_b))


def run_standalone_test():
    """
    Interactive test verifying:
    1. DINOv2 loading and CPU execution.
    2. Connection with perception.py (capturing from webcam).
    3. Unit vector norm verification (||f||_2 == 1.0).
    4. Real-time cosine similarity between sequential captures.
    """
    import cv2
    from perception import PerceptionModule

    print("\n" + "=" * 65)
    print(" DINOv2 FEATURE EXTRACTOR & PERCEPTION INTEGRATION TEST")
    print("=" * 65)

    # 1. Initialize Feature Extractor
    extractor = DINOv2FeatureExtractor()

    # 2. Initialize Camera & HUD
    perception = PerceptionModule(camera_index=0, roi_size=280)
    if not perception.open_camera():
        print("[Test] Could not access webcam. Aborting test.")
        return

    print("\n[Instructions]:")
    print(" 1. Hold Object A in the green box and press [SPACE] to capture Baseline.")
    print(" 2. Move or rotate Object A, or hold Object B, and press [SPACE] again.")
    print(" 3. Watch the terminal print the Cosine Similarity and CPU inference latency.")
    print(" 4. Press [Q] to quit.\n" + "=" * 65 + "\n")

    baseline_feature = None
    baseline_label = "None"
    capture_count = 0

    try:
        while True:
            ret, frame = perception.read_frame()
            if not ret or frame is None:
                break

            annotated_frame = perception.draw_hud(frame)
            cv2.imshow("DINOv2 Feature Extractor Test", annotated_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                print("[Test] Exiting.")
                break
            elif key == 32:  # SPACEBAR pressed -> On-demand extraction
                capture_count += 1
                # Crop ROI and convert BGR -> RGB
                crop_rgb = perception.crop_roi(frame, to_rgb=True)

                # Extract DINOv2 384-d feature vector
                feat, latency = extractor.extract(crop_rgb)
                l2_norm = np.linalg.norm(feat)

                print(f"\n--- [Capture #{capture_count}] ---")
                print(f" * Feature Vector Shape : {feat.shape} (DINOv2-small)")
                print(f" * Inference Latency    : {latency:.1f} ms on Intel Core i5-6200U")
                print(f" * Vector L2 Norm       : {l2_norm:.4f} (Must be exactly 1.0000)")

                if baseline_feature is None:
                    baseline_feature = feat
                    baseline_label = f"Capture #1"
                    print(f" * Stored as BASELINE comparison reference.")
                    perception.set_state("READY", "Baseline captured! Present same or new object & press SPACE.")
                else:
                    similarity = extractor.compute_cosine_similarity(baseline_feature, feat)
                    print(f" * Cosine Similarity with {baseline_label}: {similarity:+.4f}")

                    if similarity >= 0.80:
                        assessment = "HIGH MATCH (Same Class / In-Distribution)"
                        perception.set_state("READY", f"Similarity: {similarity:.2f} (Match)")
                    elif similarity >= 0.50:
                        assessment = "AMBIGUOUS / UNCERTAIN (HRI Disambiguation Zone)"
                        perception.set_state("AMBIGUOUS", f"Similarity: {similarity:.2f} (Ambiguous)")
                    else:
                        assessment = "DIVERGENT (Novel / Different Object)"
                        perception.set_state("NOVEL", f"Similarity: {similarity:.2f} (Novel Object)")

                    print(f" * Dual-Threshold Evaluation: {assessment}")

    finally:
        perception.release()


if __name__ == "__main__":
    run_standalone_test()
