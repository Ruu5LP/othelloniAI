"""
Example script demonstrating basic usage of the OthelloniAI capture tool.

This script shows how to:
1. Capture a screenshot
2. Extract the board region
3. Split the board into 8x8 cells
4. Extract hand regions

Note: This is a demonstration script. In production, you'll need to:
- Configure proper screen regions in config.py
- Adjust for your specific game window position and size
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.capture import capture_screen, save_frame
from src.capture.extract_regions import process_frame
from src.capture import config


def main():
    print("OthelloniAI - Screen Capture Example")
    print("=" * 50)
    
    # Ensure directories exist
    config.ensure_directories()
    print(f"✓ Data directories ready")
    print(f"  Raw frames: {config.RAW_DIR}")
    print(f"  Board crops: {config.BOARD_DIR}")
    print(f"  Hand crops: {config.HANDS_DIR}")
    print(f"  Cell crops: {config.CELLS_DIR}")
    print()
    
    # Capture a screenshot
    print("Capturing screenshot...")
    try:
        frame = capture_screen()
        print(f"✓ Screenshot captured: {frame.shape}")
    except Exception as e:
        print(f"✗ Error capturing screenshot: {e}")
        print("\nNote: Screen capture requires a display. If running in a")
        print("headless environment, this is expected to fail.")
        return
    
    # Save the raw frame
    saved_path = save_frame(frame)
    print(f"✓ Frame saved to: {saved_path}")
    print()
    
    # Process the frame (extract all regions)
    print("Processing frame...")
    results = process_frame(frame)
    
    if 'board' in results:
        print(f"✓ Board extracted: {results['board'].shape}")
    
    if 'hands' in results:
        print(f"✓ Hands extracted: {len(results['hands'])} regions")
    
    if 'cells' in results:
        cells = results['cells']
        print(f"✓ Cells extracted: {len(cells)}x{len(cells[0])} grid")
        print(f"  Total cells: {sum(len(row) for row in cells)}")
    
    print()
    print("=" * 50)
    print("Example completed successfully!")
    print()
    print("Next steps:")
    print("1. Configure board and hand regions in src/capture/config.py")
    print("2. Position your game window and run this script again")
    print("3. Check the data/ directory for extracted images")


if __name__ == "__main__":
    main()
