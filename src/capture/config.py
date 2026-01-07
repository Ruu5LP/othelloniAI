"""Configuration settings for screen capture and region extraction."""

import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
CROPS_DIR = DATA_DIR / "crops"

# Crop subdirectories
BOARD_DIR = CROPS_DIR / "board"
HANDS_DIR = CROPS_DIR / "hands"
CELLS_DIR = CROPS_DIR / "cells"

# Screen capture settings
MONITOR_NUMBER = 1  # 0 for all monitors, 1 for primary, 2+ for secondary
CAPTURE_FPS = 3  # Target frames per second for continuous capture

# Board extraction settings
BOARD_GRID_SIZE = 6  # 8x8 Othello board
BOARD_REGION = (36, 264, 380, 380)

# Hand extraction settings
PLAYER_HAND_REGION = (0, 860, 462, 183)  # x, y, w, h inside phone image
HANDS_REGIONS = {
    "player": PLAYER_HAND_REGION,
    "opponent": None,
}

# Image processing settings
SAVE_RAW_FRAMES = True  # Save raw captured frames
SAVE_BOARD_CROPS = True  # Save extracted board region
SAVE_HAND_CROPS = True  # Save extracted hand regions
SAVE_CELL_CROPS = True  # Save individual cell crops

# File naming
FRAME_FILENAME_FORMAT = "frame_{timestamp}.png"
BOARD_FILENAME_FORMAT = "board_{timestamp}.png"
HAND_FILENAME_FORMAT = "hand_{player}_{timestamp}.png"
CELL_FILENAME_FORMAT = "cell_{row}_{col}_{timestamp}.png"

# AirServer phone region (for 1920x1080 screenshot sample)
PHONE_REGION = (729, 32, 462, 1043)  # x, y, w, h
SAVE_PHONE_FRAMES = True
PHONE_DIR = CROPS_DIR / "phone"
PHONE_FILENAME_FORMAT = "phone_{timestamp}.png"
PHONE_SIZE = (462, 1043)

# Create directories if they don't exist
def ensure_directories():
    """Create all necessary directories for data storage."""
    for directory in [RAW_DIR, BOARD_DIR, HANDS_DIR, CELLS_DIR, PHONE_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
