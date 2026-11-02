"""
Master Integration Script: Real-World Perception, Continual Memory & PyBullet Simulation (main.py)
Part of: Interactive Few-Shot Concept Grounding for Continual Robotic Sorting

Master Workflow:
----------------
1. Spawns PyBullet 3D Tabletop Simulation Station (Franka Panda Arm + 3 Colored Bins).
2. Starts Laptop Webcam Perception with centered Target ROI and dynamic HUD.
3. Loads offline Windows SAPI5 voice dialogue (HRI Interface).
4. Restores episodic centroid memory from data/memory.pkl (Zero Catastrophic Forgetting).
5. User presents real-world object to camera and presses [SPACE]:
   - DINOv2 extracts 384-d unit feature vector on CPU (~300ms).
   - Continual Memory evaluates via Dual-Threshold Policy:
       * NOVEL: Voice query to register object & target bin (3-shot enrollment).
       * AMBIGUOUS: Active voice query to resolve uncertainty & update centroid online.
       * CONFIDENT: Robot speaks confirmation aloud, then triggers PyBullet Franka Panda arm
                    to autonomously pick up proxy payload, transit, and drop into destination bin!
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
from src.simulation import SimulationStation


def main():
    print("\n" + "=" * 70)
    print(" INTERACTIVE FEW-SHOT CONCEPT GROUNDING FOR CONTINUAL ROBOTIC SORTING")
    print(" Master System: Real-World Perception + DINOv2 + Continual Memory + PyBullet Sim")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # 1. INITIALIZE ALL SYSTEM MODULES
    # -------------------------------------------------------------------------
    print("\n[System Startup] Initializing modules...")

    # A. Continual Memory (CBCL)
    memory = ContinualMemory(
        tau_high=0.72,
        tau_low=0.45,
        delta_margin=0.10,
        save_path="data/memory.pkl"
    )
    memory.load()

    # B. DINOv2 Feature Extractor (CPU ViT-S/14)
    extractor = DINOv2FeatureExtractor(device="cpu")

    # C. Real-World Perception & HUD
    perception = PerceptionModule(camera_index=0, roi_size=280)
    if not perception.open_camera():
        print("[Fatal Error] Unable to open webcam. Exiting.")
        return

    # D. Offline SAPI5 Voice Dialogue
    hri = HRIInterface(voice_rate=175)

    # E. PyBullet 3D Simulation Station
    print("[System Startup] Launching PyBullet 3D Physics Simulation...")
    sim = SimulationStation(gui=True)

    # Position OpenCV window neatly on screen
    cv2.namedWindow("Robot Perception HUD", cv2.WINDOW_AUTOSIZE)
    cv2.moveWindow("Robot Perception HUD", 50, 100)

    # Robot Startup Voice Greeting
    hri.prompt_ready()

    print("\n" + "=" * 70)
    print(" SYSTEM READY - DUAL-WINDOW MODE ACTIVE")
    print(" - Left Window  : Real-World Camera & Target ROI Box")
    print(" - Right Window : PyBullet 3D Tabletop Robotic Station")
    print("\n Controls:")
    print("   [SPACE] : Inspect object in box & trigger sorting")
    print("   [C]     : Clear / Reset memory")
    print("   [Q]     : Save memory & safely shut down")
    print("=" * 70 + "\n")

    try:
        while True:
            # 1. Grab camera frame
            ret, frame = perception.read_frame()
            if not ret or frame is None:
                break

            # 2. Render HUD and display camera window
            annotated_frame = perception.draw_hud(frame)
            cv2.imshow("Robot Perception HUD", annotated_frame)

            # 3. Keep PyBullet physics engine alive and responsive to camera rotation
            sim.step_sim(1)

            key = cv2.waitKey(1) & 0xFF

            # -----------------------------------------------------------------
            # USER INPUT: EXIT SYSTEM
            # -----------------------------------------------------------------
            if key == ord('q') or key == 27:
                print("\n[Shutdown] Saving memory and closing all subsystems...")
                memory.save()
                hri.speak("Shutting down. Goodbye.")
                break

            # -----------------------------------------------------------------
            # USER INPUT: RESET MEMORY
            # -----------------------------------------------------------------
            elif key == ord('c'):
                memory.classes.clear()
                if os.path.exists("data/memory.pkl"):
                    os.remove("data/memory.pkl")
                print("\n[Memory Cleared] Episodic memory reset to empty.")
                perception.set_state("READY", "Memory cleared! Present an object to begin.")
                hri.speak("Memory cleared. Ready for new objects.")

            # -----------------------------------------------------------------
            # USER INPUT: INSPECT OBJECT [SPACEBAR]
            # -----------------------------------------------------------------
            elif key == 32:
                print("\n" + "-" * 55)
                print("[Inspection Triggered] Processing Target ROI...")

                # A. Crop ROI (BGR -> RGB)
                crop_rgb = perception.crop_roi(frame, to_rgb=True)

                # B. Extract DINOv2 Feature Vector
                feature_vector, latency = extractor.extract(crop_rgb)
                print(f" * DINOv2 Latency : {latency:.1f} ms on Intel Core i5-6200U")

                # C. Evaluate against Continual Centroid Memory
                status, ranked, meta = memory.evaluate(feature_vector)
                print(f" * Decision Status : {status}")

                # =============================================================
                # CASE 1: NOVEL OBJECT / EMPTY MEMORY
                # =============================================================
                if status in (ContinualMemory.STATUS_EMPTY, ContinualMemory.STATUS_NOVEL):
                    perception.set_state("NOVEL", "Novel Object Detected! Check terminal...")
                    cv2.imshow("Robot Perception HUD", perception.draw_hud(frame))
                    cv2.waitKey(1)

                    # Trigger Protocol 1 (Voice)
                    hri.query_novel_object()

                    if status == ContinualMemory.STATUS_EMPTY:
                        print(" >>> MEMORY IS EMPTY. Let's teach the robot its first object! <<<")
                    else:
                        top_sim = meta.get("s1", 0.0)
                        print(f" >>> NOVEL OBJECT DETECTED (Top similarity: {top_sim:.2f} < 0.45). <<<")

                    print(" Please switch to this terminal window to enter details:")
                    name = input("   Enter Object Name (e.g., 'Pod', 'Hair Cream', 'Keys'): ").strip()
                    if name:
                        try:
                            bin_id = int(input("   Enter Target Destination Bin (1, 2, or 3): ").strip())
                            if bin_id not in (1, 2, 3):
                                bin_id = 1
                        except ValueError:
                            bin_id = 1
                            print("   [Defaulting to Bin 1]")

                        # Register initial view
                        memory.register_class(label=name, bin_id=bin_id, feature_vector=feature_vector)
                        memory.save()
                        print(f"   [Sample 1/3 Registered]")

                        # 3-Shot Teaching Burst (Wide angular tolerance)
                        add_shots = input("   Collect 2 more quick angles (N=3) for wide tilt tolerance? [Y/n]: ").strip().lower()
                        if add_shots != 'n':
                            for shot_idx in (2, 3):
                                angle_prompt = "Rotate slightly" if shot_idx == 2 else "Show different side or top"
                                hri.speak(f"Shot {shot_idx}. {angle_prompt} and press Space.")
                                print(f"\n   >>> Shot {shot_idx}/3: {angle_prompt} in the green box and press [SPACE] on camera window...")
                                perception.set_state("READY", f"Shot {shot_idx}/3 for '{name}': {angle_prompt} & press SPACE")

                                while True:
                                    r, f = perception.read_frame()
                                    if not r or f is None:
                                        break
                                    cv2.imshow("Robot Perception HUD", perception.draw_hud(f))
                                    sim.step_sim(1)
                                    k = cv2.waitKey(1) & 0xFF
                                    if k == 32:  # SPACE
                                        crop_burst = perception.crop_roi(f, to_rgb=True)
                                        feat_burst, _ = extractor.extract(crop_burst)
                                        memory.update_centroid(name, feat_burst)
                                        memory.save()
                                        print(f"   [Sample {shot_idx}/3 Enrolled!]")
                                        break

                            hri.speak(f"Three shots enrolled for {name}. Concept grounded.")
                            print(f"\n   >>> 3-Shot Centroid Enrolled for '{name}'! <<<")
                        else:
                            hri.speak(f"Registered {name} for Bin {bin_id}.")

                        perception.set_state("READY", f"Grounded: '{name}' -> Bin {bin_id}")
                    else:
                        print("   [Skipped registration]")
                        perception.set_state("READY", "Registration skipped.")

                # =============================================================
                # CASE 2: AMBIGUOUS (Uncertain between candidates or borderline)
                # =============================================================
                elif status == ContinualMemory.STATUS_AMBIGUOUS:
                    top_label = meta["top_label"]
                    top_sim = meta["s1"]
                    top_bin = meta["top_bin"]

                    perception.set_state("AMBIGUOUS", f"Uncertain: Is this '{top_label}'? ({top_sim:.2f})")
                    cv2.imshow("Robot Perception HUD", perception.draw_hud(frame))
                    cv2.waitKey(1)

                    # Subcase 2A: Multi-Class Margin Ambiguity
                    if len(ranked) >= 2 and meta.get("margin", 1.0) < meta.get("delta_margin", 0.10):
                        class_1, class_2 = ranked[0][0], ranked[1][0]
                        # Trigger Protocol 2 (Voice)
                        hri.query_disambiguation(class_1, class_2)
                        print(f"\n >>> CANDIDATE AMBIGUITY (Narrow Margin < 0.10)! <<<")
                        print(f" 1. {class_1} (Similarity: {ranked[0][1]:.2f})")
                        print(f" 2. {class_2} (Similarity: {ranked[1][1]:.2f})")
                        choice = input(f" Which class is this? Press 1 for '{class_1}', 2 for '{class_2}', or 'n' for neither: ").strip()

                        if choice == '1':
                            memory.update_centroid(class_1, feature_vector)
                            memory.save()
                            hri.confirm_action(class_1, ranked[0][2])
                            perception.set_state("READY", f"Resolved: '{class_1}' -> Bin {ranked[0][2]}")
                            # Execute Simulation Sorting Action
                            sim.execute_pick_and_place(bin_id=ranked[0][2], object_name=class_1, confidence=ranked[0][1])
                        elif choice == '2':
                            memory.update_centroid(class_2, feature_vector)
                            memory.save()
                            hri.confirm_action(class_2, ranked[1][2])
                            perception.set_state("READY", f"Resolved: '{class_2}' -> Bin {ranked[1][2]}")
                            # Execute Simulation Sorting Action
                            sim.execute_pick_and_place(bin_id=ranked[1][2], object_name=class_2, confidence=ranked[1][1])
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

                    # Subcase 2B: Single-Class Borderline Similarity
                    else:
                        hri.speak(f"I am uncertain. Is this {top_label}?")
                        print(f"\n >>> BORDERLINE AMBIGUITY ({top_sim:.2f} in [0.45, 0.72])! <<<")
                        answer = input(f" Teacher, is this '{top_label}'? (y/n): ").strip().lower()

                        if answer == 'y':
                            memory.update_centroid(top_label, feature_vector)
                            memory.save()
                            hri.confirm_action(top_label, top_bin)
                            perception.set_state("READY", f"Centroid updated for '{top_label}'")
                            # Execute Simulation Sorting Action
                            sim.execute_pick_and_place(bin_id=top_bin, object_name=top_label, confidence=top_sim)
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

                # =============================================================
                # CASE 3: CONFIDENT AUTONOMOUS SORTING
                # =============================================================
                elif status == ContinualMemory.STATUS_CONFIDENT:
                    top_label = meta["top_label"]
                    top_sim = meta["s1"]
                    top_bin = meta["top_bin"]

                    msg = f"CONFIDENT: '{top_label}' -> Bin {top_bin} ({top_sim:.0%})"
                    print(f"\n >>> {msg} <<<")
                    perception.set_state("READY", msg)
                    cv2.imshow("Robot Perception HUD", perception.draw_hud(frame))
                    cv2.waitKey(1)

                    # Trigger Protocol 3 (Voice Confirmation)
                    hri.confirm_action(top_label, top_bin)

                    # TRIGGER PHYSICAL PYBULLET ROBOTIC ARM SORTING!
                    sim.execute_pick_and_place(
                        bin_id=top_bin,
                        object_name=top_label,
                        confidence=top_sim
                    )

                    perception.set_state("READY", "Sorting complete. Place next object in box.")

    finally:
        print("\n[Cleanup] Releasing camera, speech worker, and physics engine...")
        perception.release()
        hri.stop()
        sim.close()
        print("[Shutdown] System closed cleanly.")


if __name__ == "__main__":
    main()
