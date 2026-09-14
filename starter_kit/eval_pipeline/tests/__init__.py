"""Seeded dummy-data generation + hand-built edge-case fixtures for the
RETECO evaluation pipeline unit tests and smoke checks.

No trained model and no network access are required.
"""
import os
import random

TEST_ROOT = os.path.dirname(os.path.abspath(__file__))
PACKAGE_ROOT = os.path.dirname(TEST_ROOT)


def run_suite(verbosity=2):
    import unittest
    loader = unittest.TestLoader()
    suite = loader.discover(start_dir=TEST_ROOT, pattern="test_*.py")
    runner = unittest.TextTestRunner(verbosity=verbosity)
    return runner.run(suite)