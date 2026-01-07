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
BOARD_GRID_SIZE = 8  # 8x8 Othello board
BOARD_REGION = None  # (x, y, width, height) - Auto-detect if None

# Hand extraction settings
HANDS_REGIONS = {
    "player": None,  # (x, y, width, height) - Auto-detect if None
    "opponent": None,  # (x, y, width, height) - Auto-detect if None
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

# Create directories if they don't exist
def ensure_directories():
    """Create all necessary directories for data storage."""
    for directory in [RAW_DIR, BOARD_DIR, HANDS_DIR, CELLS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
