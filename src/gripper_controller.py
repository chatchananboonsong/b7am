#!/usr/bin/env python3
"""Shim for backward compatibility: Gripper has been replaced by GimbalScanController."""
from .gimbal_controller import GimbalScanController

# Simple alias for any legacy imports
SimpleGripperController = GimbalScanController