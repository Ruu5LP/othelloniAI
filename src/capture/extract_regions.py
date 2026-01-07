"""Extract and crop regions from captured frames."""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
        if config.BOARD_REGION is not None:
            region = config.BOARD_REGION
        else:
            region = auto_detect_board_region(frame)
            save_board_detect_debug(frame, region)

    
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

    # remove outer border (non-symmetric)
    top = 10
    left = 4
    right = 10
    bottom = 32   # 下だけ大きく削る

    pts = np.array([
        [18, 10],    # tl
        [board.shape[1]-18, 10],  # tr
        [board.shape[1]-6, board.shape[0]-6],  # br
        [6, board.shape[0]-6],    # bl
    ])

    # remove outer border
    top, left, right, bottom = 10, 10, 10, 26
    board = board[top:-bottom, left:-right]

    # ✅ NEW: perspective rectify (台形補正)
    board = rectify_board_by_contour(board, out_size=480)

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

def save_grid_debug(board: np.ndarray, grid_size: int = 6, out_path="debug_grid.png"):
    dbg = board.copy()
    h, w = dbg.shape[:2]
    for i in range(1, grid_size):
        x = int(w * i / grid_size)
        y = int(h * i / grid_size)
        cv2.line(dbg, (x, 0), (x, h), (0, 0, 255), 1)
        cv2.line(dbg, (0, y), (w, y), (0, 0, 255), 1)
    cv2.imwrite(out_path, dbg)
    print("[debug] wrote", out_path)


def extract_cells(board: np.ndarray, grid_size: int = 8, save: bool = True):
    height, width = board.shape[:2]

    # ✅ 境界を等分（端の余りも全部含める）
    xs = np.linspace(0, width,  grid_size + 1).astype(int)
    ys = np.linspace(0, height, grid_size + 1).astype(int)

    cells = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f") if save else None

    for row in range(grid_size):
        row_cells = []
        for col in range(grid_size):
            x_start, x_end = xs[col], xs[col+1]
            y_start, y_end = ys[row], ys[row+1]

            cell = board[y_start:y_end, x_start:x_end]
            row_cells.append(cell)

            if save and config.SAVE_CELL_CROPS:
                filename = config.CELL_FILENAME_FORMAT.format(
                    row=row, col=col, timestamp=timestamp
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
) -> Dict[str, Any]:
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

    save_grid_debug(board, grid_size=config.BOARD_GRID_SIZE)

    # Extract cells from board
    if extract_board_cells:
        cells = extract_cells(board, grid_size=config.BOARD_GRID_SIZE)
        results['cells'] = cells
    
    return results

def main():
    # 1) pick latest phone/raw frame automatically
    # Prefer phone crops if you created them, else raw frames
    phone_dir = getattr(config, "PHONE_DIR", None)
    candidates = []

    if phone_dir and Path(phone_dir).exists():
        candidates = sorted(Path(phone_dir).glob("*.png")) + sorted(Path(phone_dir).glob("*.jpg"))
    if not candidates:
        candidates = sorted(config.RAW_DIR.glob("*.png")) + sorted(config.RAW_DIR.glob("*.jpg"))

    if not candidates:
        raise FileNotFoundError("No input frames found in data/raw or data/crops/phone")

    latest = candidates[-1]
    print("[extract] using:", latest)

    frame = cv2.imread(str(latest))
    if frame is None:
        raise RuntimeError(f"Failed to read image: {latest}")

    # 2) process & save
    config.ensure_directories()

    results = process_frame(
        frame,
        extract_board_region=True,
        extract_hand_regions=True,
        extract_board_cells=True,
    )

    board = results["board"]
    state = detect_board_state(board, grid_size=config.BOARD_GRID_SIZE, sample=10)
    save_state_debug(board, state)
    print(state)


    print("[extract] saved board:", config.BOARD_DIR)
    print("[extract] saved hands:", config.HANDS_DIR)
    print("[extract] saved cells:", config.CELLS_DIR)

def auto_detect_board_region(phone_img: np.ndarray) -> Tuple[int, int, int, int]:
    """Detect the board square region (x,y,w,h) from a phone screenshot."""
    gray = cv2.cvtColor(phone_img, cv2.COLOR_BGR2GRAY)

    # Blur a bit to reduce noise
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Edge detect
    edges = cv2.Canny(blur, 50, 150)

    # Close gaps in edges
    kernel = np.ones((5, 5), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best = None
    best_score = 0

    h_img, w_img = phone_img.shape[:2]

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h

        if area < 20000:
            continue

        # square-ish check
        aspect = w / float(h)
        if aspect < 0.85 or aspect > 1.15:
            continue

        # board is usually central and large
        center_x = x + w / 2
        center_y = y + h / 2

        # penalize too close to edges
        if center_x < w_img * 0.2 or center_x > w_img * 0.8:
            continue
        if center_y < h_img * 0.15 or center_y > h_img * 0.85:
            continue

        score = area
        if score > best_score:
            best_score = score
            best = (x, y, w, h)

    if best is None:
        raise RuntimeError("Board region not found automatically")

    return best

def save_board_detect_debug(phone_img: np.ndarray, region: Tuple[int,int,int,int], out="debug_board_detect.png"):
    dbg = phone_img.copy()
    x, y, w, h = region
    cv2.rectangle(dbg, (x, y), (x+w, y+h), (0, 0, 255), 2)
    cv2.imwrite(out, dbg)
    print("[debug] wrote", out)

def detect_board_state(board: np.ndarray, grid_size: int = 6, sample: int = 10):
    h, w = board.shape[:2]
    xs = np.linspace(0, w, grid_size + 1).astype(int)
    ys = np.linspace(0, h, grid_size + 1).astype(int)

    state = np.zeros((grid_size, grid_size), dtype=np.int32)

    for rr in range(grid_size):
        for cc in range(grid_size):
            x0, x1 = xs[cc], xs[cc+1]
            y0, y1 = ys[rr], ys[rr+1]

            cell = board[y0:y1, x0:x1]
            ch, cw = cell.shape[:2]

            cx0 = max(0, cw//2 - sample//2)
            cx1 = min(cw, cw//2 + sample//2)
            cy0 = max(0, ch//2 - sample//2)
            cy1 = min(ch, ch//2 + sample//2)

            patch = cell[cy0:cy1, cx0:cx1]

            hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
            H = hsv[:,:,0].mean()
            S = hsv[:,:,1].mean()
            V = hsv[:,:,2].mean()

            # ✅ r,g,b は行番号と被るので名前を変える
            bb = patch[:,:,0].mean()
            gg = patch[:,:,1].mean()
            rr_val = patch[:,:,2].mean()

            # black
            if V < 85:
                state[rr, cc] = 1

            # white (stricter)
            elif V > 235 and S < 25 and abs(rr_val-gg) < 12 and abs(gg-bb) < 12:
                state[rr, cc] = 2

            # empty
            else:
                state[rr, cc] = 0

    return state


def save_state_debug(board: np.ndarray, state: np.ndarray, out="debug_state.png"):
    dbg = board.copy()
    h, w = dbg.shape[:2]
    grid = state.shape[0]

    xs = np.linspace(0, w, grid + 1).astype(int)
    ys = np.linspace(0, h, grid + 1).astype(int)

    for r in range(grid):
        for c in range(grid):
            x0, x1 = xs[c], xs[c+1]
            y0, y1 = ys[r], ys[r+1]
            cx = (x0 + x1)//2
            cy = (y0 + y1)//2

            v = state[r, c]
            txt = "." if v == 0 else ("B" if v == 1 else "W")
            cv2.putText(dbg, txt, (cx-6, cy+6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

    cv2.imwrite(out, dbg)
    print("[debug] wrote", out)

def warp_board_perspective(board: np.ndarray, pts: np.ndarray, out_size: int = 480) -> np.ndarray:
    """
    Warp a trapezoid board image into a perfect square.
    pts: np.array of shape (4,2) in order [tl, tr, br, bl]
    """
    dst = np.array([
        [0, 0],
        [out_size - 1, 0],
        [out_size - 1, out_size - 1],
        [0, out_size - 1]
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(pts.astype(np.float32), dst)
    warped = cv2.warpPerspective(board, M, (out_size, out_size))
    return warped

def rectify_board_by_contour(board: np.ndarray, out_size: int = 480) -> np.ndarray:
    gray = cv2.cvtColor(board, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(blur, 30, 120)
    kernel = np.ones((5, 5), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=3)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    h, w = board.shape[:2]
    img_area = h * w

    best_quad = None
    best_score = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < img_area * 0.20:
            continue

        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

        if len(approx) != 4:
            continue

        x, y, bw, bh = cv2.boundingRect(approx)
        aspect = bw / float(bh)

        if aspect < 0.85 or aspect > 1.15:
            continue

        cx = x + bw / 2
        cy = y + bh / 2
        if cx < w * 0.2 or cx > w * 0.8:
            continue
        if cy < h * 0.2 or cy > h * 0.9:
            continue

        score = area
        if score > best_score:
            best_score = score
            best_quad = approx

    if best_quad is None:
        raise RuntimeError("Board contour not found")

    # ✅ デバッグ: 何を拾ったか描画
    dbg = board.copy()
    cv2.polylines(dbg, [best_quad], True, (0,0,255), 3)
    cv2.imwrite("debug_contour.png", dbg)
    print("[debug] wrote debug_contour.png")

    # ✅ここからあなたのコードを貼る場所！
    pts = best_quad.reshape(4, 2).astype(np.float32)

    def order_points(pts):
        pts = np.array(pts, dtype=np.float32)
        y_sorted = pts[np.argsort(pts[:, 1])]
        top = y_sorted[:2]
        bottom = y_sorted[2:]

        top = top[np.argsort(top[:, 0])]
        tl, tr = top

        bottom = bottom[np.argsort(bottom[:, 0])]
        bl, br = bottom

        return np.array([tl, tr, br, bl], dtype=np.float32)

    src = order_points(pts)

    dst = np.array([
        [0, 0],
        [out_size - 1, 0],
        [out_size - 1, out_size - 1],
        [0, out_size - 1]
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(board, M, (out_size, out_size))

    # ✅ warp結果も保存（必須）
    cv2.imwrite("debug_warp.png", warped)
    print("[debug] wrote debug_warp.png")

    return warped

def order_points(pts):
    # pts: (4,2)
    pts = np.array(pts, dtype=np.float32)

    # yでソートして上2点・下2点に分ける
    y_sorted = pts[np.argsort(pts[:, 1])]
    top = y_sorted[:2]
    bottom = y_sorted[2:]

    # 上2点をxでソート → tl,tr
    top = top[np.argsort(top[:, 0])]
    tl, tr = top

    # 下2点をxでソート → bl,br
    bottom = bottom[np.argsort(bottom[:, 0])]
    bl, br = bottom

    return np.array([tl, tr, br, bl], dtype=np.float32)


if __name__ == "__main__":
    main()