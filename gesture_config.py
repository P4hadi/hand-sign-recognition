"""Central gesture configuration for data collection.

The existing trained models intentionally remain five-class artifacts. This
module defines the broader collection vocabulary without changing those model
contracts.
"""

from __future__ import annotations


LEGACY_GESTURE_IDS = (
    "open_palm",
    "fist",
    "thumbs_up",
    "peace_sign",
    "ok_sign",
)

ADDITIONAL_GESTURE_IDS = (
    "pointing_up",
    "three_fingers",
    "call_me",
)

ALL_GESTURE_IDS = LEGACY_GESTURE_IDS + ADDITIONAL_GESTURE_IDS

GESTURE_CONFIG: dict[str, dict[str, str]] = {
    "open_palm": {
        "name": "Open Palm",
        "description": "One hand open with all five fingers extended and separated.",
    },
    "fist": {
        "name": "Fist",
        "description": "One hand closed with the fingers curled into the palm.",
    },
    "thumbs_up": {
        "name": "Thumbs Up",
        "description": "Closed hand with the thumb extended upward.",
    },
    "peace_sign": {
        "name": "Peace Sign",
        "description": "Index and middle fingers extended in a V; other fingers curled.",
    },
    "ok_sign": {
        "name": "OK Sign",
        "description": "Thumb and index finger form a circle; remaining fingers extend.",
    },
    "pointing_up": {
        "name": "Pointing Up",
        "description": "Index finger extended upward; other fingers curled.",
    },
    "three_fingers": {
        "name": "Three Fingers",
        "description": "Three fingers extended together; remaining fingers curled.",
    },
    "call_me": {
        "name": "Call Me",
        "description": "Thumb and little finger extended; middle fingers curled.",
    },
}

if tuple(GESTURE_CONFIG) != ALL_GESTURE_IDS:
    raise RuntimeError("Gesture configuration order does not match gesture IDs.")

GESTURE_NAMES = {gesture_id: details["name"] for gesture_id, details in GESTURE_CONFIG.items()}
GESTURE_DESCRIPTIONS = {
    gesture_id: details["description"]
    for gesture_id, details in GESTURE_CONFIG.items()
}
