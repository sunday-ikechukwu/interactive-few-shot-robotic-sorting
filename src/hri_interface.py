"""
Module 4: Human-Robot Interaction (HRI) Dialogue Interface

Purpose:
--------
Manages real-time human-in-the-loop active dialogue using offline voice synthesis
(Windows SAPI5 via pyttsx3) and keyboard event handling.

Key Architectural Invariants:
-----------------------------
1. Offline Execution: Pure Windows SAPI5 text-to-speech engine (zero internet, zero latency).
2. Non-Blocking / Asynchronous Speech: Audio is processed on a dedicated worker thread
   so the camera HUD never freezes or drops frames while the robot is speaking.
3. Resilient SAPI5 Lifecycle: Initializes a fresh SAPI5 COM context per utterance to
   prevent Windows event pump closure from silently dropping subsequent sentences.
4. Dialogue Protocols (Section 4 of PROJECT_SPEC.md):
   - Disambiguation Query: "I am uncertain. Is this [Class 1] or [Class 2]? Press 1 or 2."
   - Novel Object Query:   "I have not seen this item before. Which bin should it go to?"
   - Confirmation:         "Confirmed. Sorting [Class] to Bin [X]."
"""

import threading
import queue
import time
from typing import Optional
import pyttsx3


class HRIInterface:
    """Manages offline voice synthesis and interactive dialogue protocols."""

    def __init__(self, voice_rate: int = 175, volume: float = 1.0):
        """
        Initializes the offline SAPI5 speech interface and threaded worker queue.

        Args:
            voice_rate: Words per minute (default: 175, natural cadence).
            volume: Audio volume scalar from 0.0 to 1.0 (default: 1.0).
        """
        self.voice_rate = voice_rate
        self.volume = volume

        # Thread-safe speech queue to prevent blocking the OpenCV video feed
        self._speech_queue: queue.Queue = queue.Queue()
        self._stop_event = threading.Event()

        # Start background TTS worker thread
        self._worker_thread = threading.Thread(target=self._speech_worker, daemon=True)
        self._worker_thread.start()

        print("[HRIInterface] Offline Windows SAPI5 speech engine initialized (Asynchronous).")

    def _speech_worker(self):
        """
        Dedicated background worker that processes queued speech utterances.
        Initializes a fresh SAPI5 COM engine per utterance to circumvent the
        well-known Windows SAPI5 event pump closure bug where subsequent calls drop audio.
        """
        while not self._stop_event.is_set():
            try:
                # Wait for next text item with 0.2s timeout for clean shutdown responsiveness
                text = self._speech_queue.get(timeout=0.2)
                if text is None:  # Sentinel shutdown token
                    break

                try:
                    # Fresh SAPI5 instance per utterance guarantees clean COM channel
                    engine = pyttsx3.init(driverName="sapi5")
                    engine.setProperty("rate", self.voice_rate)
                    engine.setProperty("volume", self.volume)
                    engine.say(text)
                    engine.runAndWait()
                    engine.stop()
                    del engine
                except Exception as e:
                    print(f"[HRIInterface] TTS error during utterance playback: {e}")

                self._speech_queue.task_done()
            except queue.Empty:
                continue

    def speak(self, text: str, block: bool = False):
        """
        Queues a text utterance for voice synthesis.

        Args:
            text: Utterance to speak.
            block: If True, waits until speech finishes. If False (default), speaks asynchronously.
        """
        print(f"[Robot Speaks]: \"{text}\"")
        self._speech_queue.put(text)
        if block:
            self._speech_queue.join()

    # ---------------------------------------------------------------------
    # Standard HRI Protocols (Specified in Section 4 of PROJECT_SPEC.md)
    # ---------------------------------------------------------------------

    def query_novel_object(self, block: bool = False) -> str:
        """
        Protocol 1: Triggered when an out-of-distribution / novel object is detected (s < tau_low).
        """
        phrase = "I have not seen this item before. Which bin should it go to?"
        self.speak(phrase, block=block)
        return phrase

    def query_disambiguation(self, class_1: str, class_2: str, block: bool = False) -> str:
        """
        Protocol 2: Triggered when similarity is ambiguous between two candidate classes.
        """
        phrase = f"I am uncertain. Is this {class_1} or {class_2}? Press 1 or 2."
        self.speak(phrase, block=block)
        return phrase

    def confirm_action(self, class_name: str, bin_id: int, block: bool = False) -> str:
        """
        Protocol 3: Triggered when a class is identified with high confidence.
        """
        phrase = f"Confirmed. Sorting {class_name} to Bin {bin_id}."
        self.speak(phrase, block=block)
        return phrase

    def prompt_ready(self, block: bool = False) -> str:
        """Welcoming prompt when system starts."""
        phrase = "Perception active. Place an object in the box and press Space."
        self.speak(phrase, block=block)
        return phrase

    def stop(self):
        """Shuts down the worker thread cleanly after pending speech completes."""
        self._speech_queue.join()  # Wait for all queued sentences to finish speaking
        self._stop_event.set()
        self._speech_queue.put(None)
        if self._worker_thread.is_alive():
            self._worker_thread.join(timeout=1.0)
        print("[HRIInterface] Speech engine stopped cleanly.")


def run_standalone_test():
    """Interactive standalone test verifying speech synthesis and dialogue protocols."""
    print("\n" + "=" * 65)
    print(" HRI INTERFACE MODULE (OFFLINE TTS) TEST")
    print("=" * 65)

    hri = HRIInterface(voice_rate=175)

    print("\nTesting Dialogue Protocol 1: System Startup Greeting")
    hri.prompt_ready(block=True)
    time.sleep(0.5)

    print("\nTesting Dialogue Protocol 2: Novel Object Detection")
    hri.query_novel_object(block=True)
    time.sleep(0.5)

    print("\nTesting Dialogue Protocol 3: Ambiguity Resolution (Disambiguation)")
    hri.query_disambiguation("Pod", "Mug", block=True)
    time.sleep(0.5)

    print("\nTesting Dialogue Protocol 4: Confident Pick-and-Place Confirmation")
    hri.confirm_action("Pod", bin_id=2, block=True)
    time.sleep(0.5)

    hri.stop()
    print("=" * 65 + "\n")


if __name__ == "__main__":
    run_standalone_test()
