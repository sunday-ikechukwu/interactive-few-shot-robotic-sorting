"""
Interactive Integration Test: Full Phase 1 Loop
Connects Modules 1, 2, 3, and 4:
  - Perception (OpenCV webcam + Target ROI + HUD)
  - Feature Extractor (Frozen DINOv2-small on CPU)
  - Continual Memory (Centroid-Based Concept Learning + Dual-Thresholds)
  - HRI Interface (Offline Windows SAPI5 Voice Synthesis)
"""

import sys
import os
import cv2
import numpy as np

# Ensure project root is in Python module path
sys.path.append(os.path.abspath("."))

from src.perception import PerceptionModule
from src.feature_extractor import DINOv2FeatureExtractor
from src.continual_memory import ContinualMemory
from src.hri_interface import HRIInterface


def main():
    print("\n" + "=" * 65)
    print(" PHASE 1 COMPLETE INTEGRATION: PERCEPTION, MEMORY & HRI")
    print("=" * 65)

    # 1. Initialize all Phase 1 modules
    memory = ContinualMemory(tau_high=0.72, tau_low=0.45, delta_margin=0.10, save_path="data/memory.pkl")
    memory.load()  # Loads existing data/memory.pkl if previously saved

    extractor = DINOv2FeatureExtractor(device="cpu")
    perception = PerceptionModule(camera_index=0, roi_size=280)
    hri = HRIInterface(voice_rate=175)

    if not perception.open_camera():
        print("[Error] Could not open webcam.")
        hri.stop()
        return

    # Startup Voice Greeting
    hri.prompt_ready()

    print("\n[Controls]:")
    print(" - Place an object in the box and press [SPACE] to inspect.")
    print(" - Follow audio & terminal instructions when novel or ambiguous.")
    print(" - Press [C] to clear memory.")
    print(" - Press [Q] to quit.")
    print("=" * 65 + "\n")

    try:
        while True:
            ret, frame = perception.read_frame()
            if not ret or frame is None:
                break

            annotated_frame = perception.draw_hud(frame)
            cv2.imshow("Robot Perception & Memory Pipeline", annotated_frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q') or key == 27:
                print("\n[Exiting] Saving memory and closing camera...")
                memory.save()
                break

            elif key == ord('c'):
                # Reset memory for a fresh experiment
                memory.classes.clear()
                if os.path.exists("data/memory.pkl"):
                    os.remove("data/memory.pkl")
                print("\n[Memory Cleared] Episodic memory reset to empty.")
                perception.set_state("READY", "Memory cleared! Present an object to begin.")
                hri.speak("Memory cleared. Ready for new objects.")

            elif key == 32:  # SPACEBAR -> Capture & Inspect
                print("\n" + "-" * 50)
                print("[Capture Triggered] Processing Target ROI...")

                # 1. Crop ROI and convert to RGB
                crop_rgb = perception.crop_roi(frame, to_rgb=True)

                # 2. Extract DINOv2 feature vector
                feature_vector, latency = extractor.extract(crop_rgb)
                print(f" * Feature extraction latency: {latency:.1f} ms")

                # 3. Evaluate against Continual Memory
                status, ranked, meta = memory.evaluate(feature_vector)
                print(f" * Decision Status: {status}")

                # -----------------------------------------------------------------
                # CASE 1: MEMORY IS EMPTY OR OBJECT IS NOVEL
                # -----------------------------------------------------------------
                if status in (ContinualMemory.STATUS_EMPTY, ContinualMemory.STATUS_NOVEL):
                    perception.set_state("NOVEL", "Novel Object Detected! Check terminal...")
                    cv2.imshow("Robot Perception & Memory Pipeline", perception.draw_hud(frame))
                    cv2.waitKey(1)

                    # Trigger Voice Query
                    hri.query_novel_object()

                    if status == ContinualMemory.STATUS_EMPTY:
                        print(" >>> MEMORY IS EMPTY. Let's teach the robot its first object! <<<")
                    else:
                        top_sim = meta.get("s1", 0.0)
                        print(f" >>> NOVEL OBJECT (Top similarity is only {top_sim:.2f} < 0.45). <<<")

                    print(" Please switch to this terminal window to enter details:")
                    name = input("   Enter Object Name (e.g., 'Hair Cream', 'Keys', 'Mug'): ").strip()
                    if name:
                        try:
                            bin_id = int(input("   Enter Target Destination Bin (1, 2, or 3): ").strip())
                        except ValueError:
                            bin_id = 1
                            print("   [Defaulting to Bin 1]")

                        # Register the first view
                        memory.register_class(label=name, bin_id=bin_id, feature_vector=feature_vector)
                        memory.save()
                        print(f"   [Sample 1/3 Registered]")

                        # 3-Shot Teaching Burst: Prompt for 2 additional rapid views
                        add_shots = input("   Collect 2 more quick angles (N=3) for wide tilt tolerance? [Y/n]: ").strip().lower()
                        if add_shots != 'n':
                            for shot_idx in (2, 3):
                                angle_prompt = "Rotate slightly" if shot_idx == 2 else "Show different side or top"
                                hri.speak(f"Shot {shot_idx}. {angle_prompt} and press Space.")
                                print(f"\n   >>> Shot {shot_idx}/3: {angle_prompt} in the green box and press [SPACE] on camera window...")
                                perception.set_state("READY", f"Shot {shot_idx}/3 for '{name}': {angle_prompt} & press SPACE")

                                # Mini-loop waiting for user to press SPACE in camera window
                                while True:
                                    r, f = perception.read_frame()
                                    if not r or f is None:
                                        break
                                    cv2.imshow("Robot Perception & Memory Pipeline", perception.draw_hud(f))
                                    k = cv2.waitKey(1) & 0xFF
                                    if k == 32:  # SPACE
                                        crop_burst = perception.crop_roi(f, to_rgb=True)
                                        feat_burst, _ = extractor.extract(crop_burst)
                                        memory.update_centroid(name, feat_burst)
                                        memory.save()
                                        print(f"   [Sample {shot_idx}/3 Enrolled!]")
                                        break

                            hri.speak(f"Three shots enrolled for {name}. Concept grounded.")
                            print(f"\n   >>> 3-Shot Centroid Enrolled for '{name}'! Wide tilt tolerance active. <<<")
                        else:
                            hri.speak(f"Registered {name} for Bin {bin_id}.")

                        perception.set_state("READY", f"Grounded: '{name}' (N={memory.classes[name].sample_count}) -> Bin {bin_id}")
                    else:
                        print("   [Skipped registration]")
                        perception.set_state("READY", "Registration skipped.")

                # -----------------------------------------------------------------
                # CASE 2: AMBIGUOUS (Uncertain / Between classes or borderline similarity)
                # -----------------------------------------------------------------
                elif status == ContinualMemory.STATUS_AMBIGUOUS:
                    top_label = meta["top_label"]
                    top_sim = meta["s1"]
                    top_bin = meta["top_bin"]

                    perception.set_state("AMBIGUOUS", f"Uncertain: Is this '{top_label}'? ({top_sim:.2f})")
                    cv2.imshow("Robot Perception & Memory Pipeline", perception.draw_hud(frame))
                    cv2.waitKey(1)

                    # If multiple classes are close in similarity, query disambiguation between top 2
                    if len(ranked) >= 2 and meta.get("margin", 1.0) < meta.get("delta_margin", 0.10):
                        class_1, class_2 = ranked[0][0], ranked[1][0]
                        hri.query_disambiguation(class_1, class_2)
                        print(f"\n >>> AMBIGUITY BETWEEN CANDIDATES! <<<")
                        print(f" 1. {class_1} ({ranked[0][1]:.2f})")
                        print(f" 2. {class_2} ({ranked[1][1]:.2f})")
                        choice = input(f" Which class is this? Press 1 for '{class_1}', 2 for '{class_2}', or 'n' for none: ").strip()

                        if choice == '1':
                            memory.update_centroid(class_1, feature_vector)
                            memory.save()
                            hri.confirm_action(class_1, ranked[0][2])
                            perception.set_state("READY", f"Clarified: '{class_1}' -> Bin {ranked[0][2]}")
                        elif choice == '2':
                            memory.update_centroid(class_2, feature_vector)
                            memory.save()
                            hri.confirm_action(class_2, ranked[1][2])
                            perception.set_state("READY", f"Clarified: '{class_2}' -> Bin {ranked[1][2]}")
                        else:
                            new_name = input(" Enter novel object name: ").strip()
                            if new_name:
                                try:
                                    b_id = int(input(" Enter Bin (1, 2, or 3): ").strip())
                                except ValueError:
                                    b_id = 1
                                memory.register_class(new_name, b_id, feature_vector)
                                memory.save()
                                hri.speak(f"Registered {new_name} to Bin {b_id}.")
                    else:
                        # Single-class borderline similarity
                        hri.speak(f"I am uncertain. Is this {top_label}?")
                        print(f"\n >>> AMBIGUOUS DETECTION! <<<")
                        print(f" * Robot thinks this is '{top_label}' with similarity {top_sim:.4f}.")
                        answer = input(f" Teacher, is this '{top_label}'? (y/n): ").strip().lower()

                        if answer == 'y':
                            memory.update_centroid(top_label, feature_vector)
                            memory.save()
                            hri.speak(f"Centroid updated for {top_label}.")
                            perception.set_state("READY", f"Updated '{top_label}' (Now {memory.classes[top_label].sample_count} views)")
                        else:
                            new_name = input(" Enter actual name of this novel object: ").strip()
                            if new_name:
                                try:
                                    b_id = int(input(" Enter Target Bin (1, 2, or 3): ").strip())
                                except ValueError:
                                    b_id = 2
                                memory.register_class(new_name, b_id, feature_vector)
                                memory.save()
                                hri.speak(f"Registered {new_name} to Bin {b_id}.")
                                perception.set_state("READY", f"Grounded: '{new_name}' -> Bin {b_id}")

                # -----------------------------------------------------------------
                # CASE 3: CONFIDENT (Similarity >= 0.72 and clear margin)
                # -----------------------------------------------------------------
                elif status == ContinualMemory.STATUS_CONFIDENT:
                    top_label = meta["top_label"]
                    top_sim = meta["s1"]
                    top_bin = meta["top_bin"]

                    msg = f"CONFIDENT: '{top_label}' -> Bin {top_bin} ({top_sim:.2f})"
                    print(f" >>> {msg} <<<")
                    perception.set_state("READY", msg)

                    # Trigger Confident Action Speech Protocol
                    hri.confirm_action(top_label, top_bin)

    finally:
        perception.release()
        hri.stop()


if __name__ == "__main__":
    main()
