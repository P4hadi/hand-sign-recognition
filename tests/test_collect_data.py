"""Webcam-independent tests for configurable gesture collection."""

from __future__ import annotations

import argparse
import sys
import unittest
from unittest.mock import patch

from collect_data import choose_labels, parse_args
from gesture_config import (
    ADDITIONAL_GESTURE_IDS,
    ALL_GESTURE_IDS,
    GESTURE_CONFIG,
    LEGACY_GESTURE_IDS,
)


class CollectDataConfigTests(unittest.TestCase):
    def test_configuration_contains_all_eight_gestures(self) -> None:
        self.assertEqual(len(ALL_GESTURE_IDS), 8)
        self.assertEqual(tuple(GESTURE_CONFIG), ALL_GESTURE_IDS)
        self.assertEqual(len(LEGACY_GESTURE_IDS), 5)
        self.assertEqual(len(ADDITIONAL_GESTURE_IDS), 3)
        for gesture_id in ALL_GESTURE_IDS:
            self.assertTrue(GESTURE_CONFIG[gesture_id]["name"])
            self.assertTrue(GESTURE_CONFIG[gesture_id]["description"])

    def test_all_gestures_preserves_original_five_label_behavior(self) -> None:
        args = argparse.Namespace(
            gestures=None,
            all_gestures=True,
            label=None,
        )
        self.assertEqual(choose_labels(args), list(LEGACY_GESTURE_IDS))

    def test_explicit_gesture_list_supports_all_eight(self) -> None:
        args = argparse.Namespace(
            gestures=list(ALL_GESTURE_IDS),
            all_gestures=False,
            label=None,
        )
        self.assertEqual(choose_labels(args), list(ALL_GESTURE_IDS))

    def test_explicit_single_label_remains_supported(self) -> None:
        args = argparse.Namespace(
            gestures=None,
            all_gestures=False,
            label="call_me",
        )
        self.assertEqual(choose_labels(args), ["call_me"])

    def test_duplicate_explicit_gestures_are_rejected(self) -> None:
        args = argparse.Namespace(
            gestures=["fist", "fist"],
            all_gestures=False,
            label=None,
        )
        with self.assertRaises(ValueError):
            choose_labels(args)

    def test_parse_args_accepts_explicit_gesture_list(self) -> None:
        with patch.object(
            sys,
            "argv",
            ["collect_data.py", "--gestures", "open_palm", "call_me"],
        ):
            args = parse_args()
        self.assertEqual(args.gestures, ["open_palm", "call_me"])
        self.assertFalse(args.all_gestures)

    def test_parse_args_rejects_unknown_gesture(self) -> None:
        with patch.object(
            sys,
            "argv",
            ["collect_data.py", "--gestures", "not_a_gesture"],
        ):
            with self.assertRaises(SystemExit):
                parse_args()

    def test_parse_args_rejects_conflicting_selection_flags(self) -> None:
        with patch.object(
            sys,
            "argv",
            ["collect_data.py", "--all-gestures", "--gestures", "call_me"],
        ):
            with self.assertRaises(SystemExit):
                parse_args()


if __name__ == "__main__":
    unittest.main()
