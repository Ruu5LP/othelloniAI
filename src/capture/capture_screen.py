"""Screen capture functionality using mss library."""

import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional, Tuple

import cv2
import mss
import numpy as np

from . import config


def capture_screen(monitor: Optional[int] = None) -> np.ndarray:
    """
    Capture a screenshot from the specified monitor.
    
    Args:
        monitor: Monitor number to capture (0 for all, 1+ for specific).
                 If None, uses config.MONITOR_NUMBER.
    
    Returns:
        numpy.ndarray: Captured screenshot in BGR format (OpenCV compatible).
    """
    monitor_num = monitor if monitor is not None else config.MONITOR_NUMBER
    
    with mss.mss() as sct:
        # Validate monitor number
        if monitor_num < 0 or monitor_num >= len(sct.monitors):
            raise ValueError(
                f"Monitor {monitor_num} not found. Available monitors: 0-{len(sct.monitors)-1}"
            )
        
        # Get monitor info
        if monitor_num == 0:
            monitor_info = sct.monitors[0]  # All monitors
        else:
            monitor_info = sct.monitors[monitor_num]
        
        # Capture the screen
        screenshot = sct.grab(monitor_info)
        
        # Convert to numpy array (BGRA format)
        img = np.array(screenshot)
        
        # Convert BGRA to BGR (OpenCV format)
        img_bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        
        return img_bgr


def save_frame(
    frame: np.ndarray,
    output_dir: Optional[Path] = None,
    filename: Optional[str] = None
) -> Path:
    """
    Save a captured frame to disk.
    
    Args:
        frame: The image frame to save (numpy array).
        output_dir: Directory to save the frame. If None, uses config.RAW_DIR.
        filename: Filename for the saved frame. If None, generates from timestamp.
    
    Returns:
        Path: Full path to the saved file.
    """
    # Set output directory
    if output_dir is None:
        output_dir = config.RAW_DIR
    
    # Ensure directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate filename if not provided
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = config.FRAME_FILENAME_FORMAT.format(timestamp=timestamp)
    
    # Save the frame
    output_path = output_dir / filename
    cv2.imwrite(str(output_path), frame)
    
    return output_path


def continuous_capture(
    duration: Optional[float] = None,
    fps: Optional[int] = None,
    callback: Optional[Callable[[np.ndarray, float], bool]] = None
) -> None:
    """
    Continuously capture screenshots at a specified frame rate.
    
    Args:
        duration: How long to capture in seconds. If None, captures indefinitely.
        fps: Target frames per second. If None, uses config.CAPTURE_FPS.
        callback: Optional callback function to process each frame.
                  Signature: callback(frame: np.ndarray, timestamp: float) -> bool
                  Return False to stop capturing.
    """
    target_fps = fps if fps is not None else config.CAPTURE_FPS
    frame_delay = 1.0 / target_fps
    
    start_time = time.time()
    
    try:
        while True:
            frame_start = time.time()
            
            # Capture frame
            frame = capture_screen()
            
            # Process with callback if provided
            if callback is not None:
                should_continue = callback(frame, frame_start)
                if not should_continue:
                    break
            else:
                # Default behavior: save frame
                if config.SAVE_RAW_FRAMES:
                    save_frame(frame)
            
            # Check duration limit
            if duration is not None:
                elapsed = time.time() - start_time
                if elapsed >= duration:
                    break
            
            # Maintain frame rate
            frame_time = time.time() - frame_start
            if frame_time < frame_delay:
                time.sleep(frame_delay - frame_time)
    
    except KeyboardInterrupt:
        print("\nCapture stopped by user.")

if __name__ == "__main__":
    config.ensure_directories()
    print("[capture] start (duration=10s)")
    continuous_capture(duration=10)


