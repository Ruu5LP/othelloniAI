"""Extract and crop regions from captured frames."""

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from . import config


def extract_board(
    frame: np.ndarray,
    region: Optional[Tuple[int, int, int, int]] = None,
    save: bool = True
) -> np.ndarray:
    """
    Extract the game board region from a frame.
    
    Args:
        frame: The full captured frame.
        region: Optional (x, y, width, height) tuple defining board region.
                If None, uses config.BOARD_REGION or returns full frame.
        save: Whether to save the extracted board.
    
    Returns:
        numpy.ndarray: Extracted board region.
    """
    # Use provided region, config region, or full frame
    if region is None:
        region = config.BOARD_REGION
    
    if region is not None:
        x, y, w, h = region
        board = frame[y:y+h, x:x+w]
    else:
        # No region specified, use full frame
        board = frame.copy()
    
    # Save if requested
    if save and config.SAVE_BOARD_CROPS:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = config.BOARD_FILENAME_FORMAT.format(timestamp=timestamp)
        output_path = config.BOARD_DIR / filename
        config.BOARD_DIR.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), board)
    
    return board


def extract_hands(
    frame: np.ndarray,
    regions: Optional[Dict[str, Tuple[int, int, int, int]]] = None,
    save: bool = True
) -> Dict[str, np.ndarray]:
    """
    Extract hand regions (player and opponent pieces) from a frame.
    
    Args:
        frame: The full captured frame.
        regions: Optional dict with 'player' and 'opponent' keys,
                 each containing (x, y, width, height) tuples.
                 If None, uses config.HANDS_REGIONS.
        save: Whether to save the extracted hands.
    
    Returns:
        Dict[str, np.ndarray]: Dictionary with 'player' and 'opponent' hand crops.
    """
    # Use provided regions or config regions
    if regions is None:
        regions = config.HANDS_REGIONS
    
    hands = {}
    
    for player_type, region in regions.items():
        if region is not None:
            x, y, w, h = region
            hand = frame[y:y+h, x:x+w]
            hands[player_type] = hand
            
            # Save if requested
            if save and config.SAVE_HAND_CROPS:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                filename = config.HAND_FILENAME_FORMAT.format(
                    player=player_type,
                    timestamp=timestamp
                )
                output_path = config.HANDS_DIR / filename
                config.HANDS_DIR.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(output_path), hand)
    
    return hands


def extract_cells(
    board: np.ndarray,
    grid_size: int = 8,
    save: bool = True
) -> List[List[np.ndarray]]:
    """
    Split the board into an 8x8 grid of cells (standard Othello board).
    
    Args:
        board: The board image to split.
        grid_size: Number of rows/columns in the grid (default: 8 for Othello).
        save: Whether to save individual cell crops.
    
    Returns:
        List[List[np.ndarray]]: 2D list of cell images [row][col].
    """
    height, width = board.shape[:2]
    cell_height = height // grid_size
    cell_width = width // grid_size
    
    cells = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f") if save else None
    
    for row in range(grid_size):
        row_cells = []
        for col in range(grid_size):
            # Calculate cell boundaries
            y_start = row * cell_height
            y_end = y_start + cell_height
            x_start = col * cell_width
            x_end = x_start + cell_width
            
            # Extract cell
            cell = board[y_start:y_end, x_start:x_end]
            row_cells.append(cell)
            
            # Save if requested
            if save and config.SAVE_CELL_CROPS:
                filename = config.CELL_FILENAME_FORMAT.format(
                    row=row,
                    col=col,
                    timestamp=timestamp
                )
                output_path = config.CELLS_DIR / filename
                config.CELLS_DIR.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(output_path), cell)
        
        cells.append(row_cells)
    
    return cells


def process_frame(
    frame: np.ndarray,
    extract_board_region: bool = True,
    extract_hand_regions: bool = True,
    extract_board_cells: bool = True
) -> Dict[str, any]:
    """
    Process a frame and extract all relevant regions.
    
    Args:
        frame: The captured frame to process.
        extract_board_region: Whether to extract the board region.
        extract_hand_regions: Whether to extract hand regions.
        extract_board_cells: Whether to split board into cells.
    
    Returns:
        Dict containing extracted regions:
        - 'board': Board region image (if extracted)
        - 'hands': Dict of hand images (if extracted)
        - 'cells': 2D list of cell images (if extracted)
    """
    results = {}
    
    # Extract board
    if extract_board_region:
        board = extract_board(frame)
        results['board'] = board
    else:
        board = frame
    
    # Extract hands
    if extract_hand_regions:
        hands = extract_hands(frame)
        results['hands'] = hands
    
    # Extract cells from board
    if extract_board_cells:
        cells = extract_cells(board, grid_size=config.BOARD_GRID_SIZE)
        results['cells'] = cells
    
    return results
