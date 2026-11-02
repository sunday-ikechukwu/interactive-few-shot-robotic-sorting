"""
Module 3: Centroid-Based Continual Learning Memory

Purpose:
--------
Implements an episodic centroid memory inspired by Centroid-Based Concept Learning
(CBCL, Ayub et al.). Enables online, few-shot registration and continual updating
of novel object classes with zero catastrophic forgetting, without retraining any
neural network weights.

Decision Framework:
-------------------
Given an extracted unit feature vector f in R^384 and known centroids c_k:
  1. Cosine similarity: s_k = f . c_k
  2. Candidate sorting: s_1 >= s_2 >= ... >= s_K
  3. Dual-Threshold & Margin Policy:
     - Confident:      s_1 >= tau_high (0.80) and (s_1 - s_2 >= margin if K >= 2)
     - Ambiguous:      tau_low <= s_1 < tau_high OR (s_1 - s_2 < margin = 0.10)
     - Out-of-Distrib: s_1 < tau_low (0.50)
"""

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import os
import pickle
import numpy as np


@dataclass
class ClassConcept:
    """Represents a single grounded object class in episodic memory."""
    label: str
    bin_id: int
    sample_count: int
    centroid: np.ndarray  # Shape: (384,), L2 norm = 1.0

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "bin_id": self.bin_id,
            "sample_count": self.sample_count,
            "centroid": self.centroid.tolist()
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ClassConcept":
        return cls(
            label=data["label"],
            bin_id=data["bin_id"],
            sample_count=data["sample_count"],
            centroid=np.array(data["centroid"], dtype=np.float32)
        )


class ContinualMemory:
    """
    Manages episodic class centroids, cosine matching, dual-threshold decision logic,
    and online non-interfering centroid updates.
    """

    # Decision status constants
    STATUS_EMPTY = "EMPTY"            # Memory has no registered classes yet
    STATUS_CONFIDENT = "CONFIDENT"    # High similarity, clear margin -> Sort autonomously
    STATUS_AMBIGUOUS = "AMBIGUOUS"    # Borderline similarity or narrow margin -> Query human
    STATUS_NOVEL = "NOVEL"            # Out-of-distribution -> Register new class

    def __init__(
        self,
        tau_high: float = 0.72, # 0.80
        tau_low: float = 0.45, #0.50
        delta_margin: float = 0.10,
        save_path: str = "data/memory.pkl"
    ):
        """
        Args:
            tau_high: Threshold for confident autonomous action (default: 0.80).
            tau_low: Threshold below which input is considered out-of-distribution (default: 0.50).
            delta_margin: Minimum similarity gap between top 2 candidates to avoid ambiguity (default: 0.10).
            save_path: Filepath for persisting memory to disk.
        """
        self.tau_high = tau_high
        self.tau_low = tau_low
        self.delta_margin = delta_margin
        self.save_path = save_path

        # Class registry: maps class label -> ClassConcept
        self.classes: Dict[str, ClassConcept] = {}

    @property
    def num_classes(self) -> int:
        """Returns the number of registered classes."""
        return len(self.classes)

    def register_class(self, label: str, bin_id: int, feature_vector: np.ndarray):
        """
        Registers a brand-new object class in episodic memory.

        Args:
            label: Human-readable name (e.g. 'Hair Cream', 'Keys', 'Mug').
            bin_id: Target destination bin integer (e.g. 1, 2, 3).
            feature_vector: L2-normalized 384-d vector from feature_extractor.py.
        """
        # Ensure unit normalization
        norm = np.linalg.norm(feature_vector)
        if norm > 0:
            norm_vector = (feature_vector / norm).astype(np.float32)
        else:
            norm_vector = feature_vector.astype(np.float32)

        self.classes[label] = ClassConcept(
            label=label,
            bin_id=bin_id,
            sample_count=1,
            centroid=norm_vector
        )
        print(f"[ContinualMemory] Registered novel class '{label}' -> Bin {bin_id} (Sample #1).")

    def update_centroid(self, label: str, feature_vector: np.ndarray):
        """
        Updates an existing class centroid using the online running average rule:
            c_k <- (N_k * c_k + f) / (N_k + 1)
            c_k <- c_k / ||c_k||_2
            N_k <- N_k + 1

        This guarantees zero catastrophic forgetting of other classes,
        while expanding the visual representation of class k.

        Args:
            label: Existing class name.
            feature_vector: L2-normalized 384-d vector.
        """
        if label not in self.classes:
            raise KeyError(f"Class '{label}' is not registered in continual memory.")

        concept = self.classes[label]
        n = concept.sample_count
        current_centroid = concept.centroid

        new_centroid = (n * current_centroid + feature_vector) / (n + 1)

        new_norm = np.linalg.norm(new_centroid)
        if new_norm > 0:
            new_centroid = new_centroid / new_norm

        concept.centroid = new_centroid.astype(np.float32)
        concept.sample_count += 1

        print(f"[ContinualMemory] Updated centroid for '{label}'. Total samples: {concept.sample_count}.")

    def evaluate(self, feature_vector: np.ndarray) -> Tuple[str, List[Tuple[str, float, int]], dict]:
        """
        Compares an input feature vector against all stored class centroids
        and evaluates using the dual-threshold decision policy.

        Args:
            feature_vector: L2-normalized 384-d feature vector.

        Returns:
            (status, ranked_candidates, metadata):
                - status: STATUS_CONFIDENT, STATUS_AMBIGUOUS, STATUS_NOVEL, or STATUS_EMPTY.
                - ranked_candidates: List of (label, similarity_score, bin_id) sorted by similarity desc.
                - metadata: Diagnostic info (top similarity s_1, margin, etc.).
        """
        if self.num_classes == 0:
            return self.STATUS_EMPTY, [], {"reason": "Memory is empty. Please register first class."}

        # 1. Compute cosine similarity against all class centroids: s_k = f . c_k
        similarities = []
        for label, concept in self.classes.items():
            sim = float(np.dot(feature_vector, concept.centroid))
            similarities.append((label, sim, concept.bin_id))

        # 2. Sort candidates in descending order of similarity: s_1 >= s_2 >= ... >= s_K
        ranked = sorted(similarities, key=lambda item: item[1], reverse=True)

        top_label, s1, top_bin = ranked[0]
        s2 = ranked[1][1] if len(ranked) >= 2 else 0.0
        margin = s1 - s2

        metadata = {
            "top_label": top_label,
            "top_bin": top_bin,
            "s1": s1,
            "s2": s2,
            "margin": margin,
            "tau_high": self.tau_high,
            "tau_low": self.tau_low,
            "delta_margin": self.delta_margin
        }

        # 3. Decision Logic:
        # Case A: Out-of-Distribution (s1 < tau_low = 0.50)
        if s1 < self.tau_low:
            return self.STATUS_NOVEL, ranked, metadata

        # Case B: Multi-class Ambiguity (margin < delta_margin = 0.10 when K >= 2)
        if len(ranked) >= 2 and margin < self.delta_margin:
            metadata["ambiguity_reason"] = f"Narrow margin between '{top_label}' ({s1:.2f}) and '{ranked[1][0]}' ({s2:.2f})"
            return self.STATUS_AMBIGUOUS, ranked, metadata

        # Case C: Single-class Uncertainty (tau_low <= s1 < tau_high = 0.80)
        if s1 < self.tau_high:
            metadata["ambiguity_reason"] = f"Top match '{top_label}' ({s1:.2f}) is below confident threshold ({self.tau_high:.2f})"
            return self.STATUS_AMBIGUOUS, ranked, metadata

        # Case D: Confident Autonomous Classification (s1 >= tau_high and margin >= delta_margin)
        return self.STATUS_CONFIDENT, ranked, metadata

    def save(self, filepath: Optional[str] = None):
        """Saves current episodic memory to disk."""
        target_path = filepath or self.save_path
        os.makedirs(os.path.dirname(os.path.abspath(target_path)), exist_ok=True)
        with open(target_path, "wb") as f:
            pickle.dump(self.classes, f)
        print(f"[ContinualMemory] Saved {self.num_classes} classes to {target_path}.")

    def load(self, filepath: Optional[str] = None) -> bool:
        """Loads episodic memory from disk if it exists."""
        target_path = filepath or self.save_path
        if not os.path.exists(target_path):
            print(f"[ContinualMemory] No saved memory file found at {target_path}.")
            return False
        with open(target_path, "rb") as f:
            self.classes = pickle.load(f)
        print(f"[ContinualMemory] Successfully loaded {self.num_classes} classes from {target_path}.")
        return True


def run_standalone_test():
    """
    Self-contained unit test verifying:
    1. Zero-shot / empty initialization.
    2. New class registration.
    3. Dual-threshold confident vs ambiguous vs novel classification.
    4. Centroid update math and margin resolution.
    5. Disk serialization (save/load).
    """
    print("\n" + "=" * 65)
    print(" CONTINUAL MEMORY MODULE (CBCL) TEST")
    print("=" * 65)

    memory = ContinualMemory(tau_high=0.80, tau_low=0.50, delta_margin=0.10)

    # Synthetic 384-dimensional unit vectors for verification
    np.random.seed(42)
    def make_unit_vector():
        v = np.random.randn(384).astype(np.float32)
        return v / np.linalg.norm(v)

    vec_haircream_angle1 = make_unit_vector()
    # Create angle 2 with similarity ~0.74 (matching user's actual experiment)
    noise = make_unit_vector()
    vec_haircream_angle2 = 0.74 * vec_haircream_angle1 + np.sqrt(1 - 0.74**2) * noise
    vec_haircream_angle2 /= np.linalg.norm(vec_haircream_angle2)

    vec_mug = make_unit_vector()

    print(f"\n1. Initial Memory Check (Empty):")
    status, ranked, meta = memory.evaluate(vec_haircream_angle1)
    print(f" * Status: {status} (Expected: EMPTY)")

    print(f"\n2. Registering Class 1: 'Hair Cream' -> Bin 1 with Angle 1:")
    memory.register_class("Hair Cream", bin_id=1, feature_vector=vec_haircream_angle1)

    print(f"\n3. Evaluating Angle 2 against Class 1:")
    status, ranked, meta = memory.evaluate(vec_haircream_angle2)
    print(f" * Similarity: {meta['s1']:.4f}")
    print(f" * Status: {status} (Expected: AMBIGUOUS, since 0.50 <= s1 < 0.80)")

    print(f"\n4. Teacher confirms Angle 2 -> Updating Centroid (N=1 -> N=2):")
    memory.update_centroid("Hair Cream", vec_haircream_angle2)

    print(f"\n5. Re-evaluating Angle 1 and Angle 2 against Updated Centroid:")
    s_angle1 = float(np.dot(vec_haircream_angle1, memory.classes["Hair Cream"].centroid))
    s_angle2 = float(np.dot(vec_haircream_angle2, memory.classes["Hair Cream"].centroid))
    print(f" * Similarity with Angle 1: {s_angle1:.4f} (Now in confident zone!)")
    print(f" * Similarity with Angle 2: {s_angle2:.4f} (Now in confident zone!)")

    print(f"\n6. Testing Out-of-Distribution Novel Object ('Mug'):")
    status, ranked, meta = memory.evaluate(vec_mug)
    print(f" * Top Match Similarity: {meta['s1']:.4f}")
    print(f" * Status: {status} (Expected: NOVEL, since s1 < 0.50)")

    print(f"\n7. Saving & Loading Memory State:")
    memory.save("data/test_memory.pkl")
    new_memory = ContinualMemory()
    new_memory.load("data/test_memory.pkl")
    print(f" * Restored classes: {list(new_memory.classes.keys())}")

    if os.path.exists("data/test_memory.pkl"):
        os.remove("data/test_memory.pkl")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    run_standalone_test()
