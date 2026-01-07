# OthelloniAI

Real-time screen capture and cropping tool for mobile Othello game boards. This tool captures game screens, extracts regions of interest (board, hands, cells), and prepares data for ML-based game analysis.

## Features

- **Real-time Screen Capture**: Uses `mss` for efficient screen capturing
- **Region Extraction**: Automatically extracts game board and hand regions
- **8x8 Cell Grid**: Splits Othello board into individual cell images
- **Organized Storage**: Saves frames and crops in structured directories
- **Docker Support**: Ready for FastAPI/ML service deployment

## Project Structure

```
othelloniAI/
├── src/
│   ├── capture/
│   │   ├── __init__.py
│   │   ├── capture_screen.py    # Screen capture using mss
│   │   ├── extract_regions.py   # Region extraction and cell splitting
│   │   └── config.py             # Configuration settings
│   └── utils/
│       └── __init__.py
├── data/
│   ├── raw/                      # Raw captured frames
│   └── crops/
│       ├── board/                # Extracted board regions
│       ├── hands/                # Extracted hand regions (player/opponent)
│       └── cells/                # Individual 8x8 cell crops
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```

## Windows Setup

### Prerequisites

- Python 3.11 or higher
- pip (Python package manager)

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Ruu5LP/othelloniAI.git
   cd othelloniAI
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Tool

#### Basic Screen Capture

```python
from src.capture import capture_screen, save_frame

# Capture a single screenshot
frame = capture_screen()
save_frame(frame)
```

#### Extract Board and Cells

```python
from src.capture import capture_screen, extract_board, extract_cells

# Capture and process
frame = capture_screen()
board = extract_board(frame)
cells = extract_cells(board)

print(f"Extracted {len(cells)}x{len(cells[0])} cells")
```

#### Continuous Capture

```python
from src.capture.capture_screen import continuous_capture

# Capture for 10 seconds at 30 FPS
continuous_capture(duration=10, fps=30)
```

#### Process Complete Frame

```python
from src.capture import capture_screen
from src.capture.extract_regions import process_frame

# Capture and extract all regions
frame = capture_screen()
results = process_frame(frame)

print(f"Board shape: {results['board'].shape}")
print(f"Hands extracted: {list(results.get('hands', {}).keys())}")
print(f"Cells grid: {len(results['cells'])}x{len(results['cells'][0])}")
```

### Configuration

Edit `src/capture/config.py` to customize:

- Monitor selection
- Capture frame rate
- Board region coordinates (for auto-cropping)
- Hand region coordinates
- Save options for different crop types

## Docker Setup

### Build and Run with Docker

```bash
# Build the Docker image
docker build -t othelloniai:latest .

# Run with docker-compose
docker-compose up
```

The Docker setup is prepared for future FastAPI/ML service integration.

## Development

### Project Dependencies

- **mss**: Fast cross-platform screen capture
- **opencv-python**: Image processing and manipulation
- **numpy**: Numerical operations on image arrays

### Future Enhancements

- FastAPI REST API for remote capture control
- ML model integration for game state recognition
- Real-time game analysis and move suggestions
- WebSocket support for live streaming

## License

This project is open source and available under the MIT License.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.