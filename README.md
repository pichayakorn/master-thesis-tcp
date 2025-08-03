# Video Processing Tool

A versatile command-line tool for video processing that combines two main functions:
1. Extracting frames from videos at specific intervals
2. Creating collage layouts from extracted frames with customizable options

## Requirements

- Python 3.6+
- OpenCV (`cv2`) library
- NumPy

To install required libraries:

```bash
pip install opencv-python numpy
```

## Usage

The script provides two main modes of operation: `extract` and `collage`.

### Extract Mode

Extract frames from videos at specified intervals.

```powershell
python video_processing.py extract --input-dir "path/to/videos" --output-dir "path/to/output/frames" [options]
```

#### Options for Extract Mode

| Option | Description |
|--------|-------------|
| `--frame-interval N` | Save every Nth frame (default: 1) |

### Collage Mode

Create collages from previously extracted frames.

```powershell
python video_processing.py collage --input-dir "path/to/frames" --output-dir "path/to/output/collages" [options]
```

#### Options for Collage Mode

| Option | Description |
|--------|-------------|
| `--grid ROWSxCOLS` | Grid layout (default: 2x2). Examples: 2x2, 3x2 |
| `--with-frame-numbers` | Show frame numbers (default: enabled) |
| `--no-frame-numbers` | Hide frame numbers |
| `--with-labels` | Add black backgrounds for frame numbers (default: enabled) |
| `--no-labels` | Remove black backgrounds for frame numbers |
| `--randomize` | Randomize the order of frames in collages (default: disabled) |

## Examples

### Extract Frames at 3 FPS (assuming 30 FPS video)

```powershell
python video_processing.py extract --input-dir "data/videos" --output-dir "data/frames/3fps" --frame-interval 10
```

### Create Standard 2x2 Collages

```powershell
python video_processing.py collage --input-dir "data/frames/3fps" --output-dir "data/collages/2x2" --grid 2x2
```

### Create 3x2 Collages Without Frame Numbers

```powershell
python video_processing.py collage --input-dir "data/frames/3fps" --output-dir "data/collages/3x2-no-numbers" --grid 3x2 --no-frame-numbers
```

### Create Randomized 2x2 Collages

```powershell
python video_processing.py collage --input-dir "data/frames/3fps" --output-dir "data/collages/2x2-random" --grid 2x2 --randomize
```

### Create Clean Collages (No Labels or Frame Numbers)

```powershell
python video_processing.py collage --input-dir "data/frames/3fps" --output-dir "data/collages/clean" --no-labels --no-frame-numbers
```

## Directory Structure

The script preserves the directory structure from the input to the output. For example:

- Videos in `data/videos/project1/video1.mp4`
- Extracted frames will be in `data/frames/3fps/project1/video1/`
- Collages will be in `data/collages/2x2/project1/video1/`

## Notes

- The script automatically handles nested directories and preserves their structure
- Frame numbers in collages are 1-based (first frame is 1)
- If a collage grid has more cells than available frames, empty cells will be left blank

## Accident Detection with AI Model

The project includes an AI-powered accident detection system that analyzes video frames to identify and classify accidents.

### Requirements for Accident Detection

- OpenAI API key (set in a `.env` file)
- Additional Python libraries:
  ```bash
  pip install openai python-dotenv loguru scikit-learn
  ```

### Usage

```powershell
python main.py --model <MODEL_NAME> --data_path <DATA_DIRECTORY> [--img_detail <low|high>]
```

#### Parameters

| Parameter | Description |
|-----------|-------------|
| `--model` | OpenAI model to use (e.g., gpt-4.1-2025-04-14, o3-2025-04-16) |
| `--data_path` | Path to the data directory containing norm, ped, and col subdirectories |
| `--img_detail` | Image detail level (default: high, options: low, high) |

#### Example

```powershell
python main.py --model "gpt-4.1-2025-04-14" --data_path "data/collages/2x2" --img_detail high
```

### Data Structure

The accident detection system expects data in the following structure:

```
data_path/
  ├── norm/  (normal traffic scenarios)
  │   ├── video1/
  │   │   ├── frame1.jpg
  │   │   ├── frame2.jpg
  │   └── video2/
  ├── ped/   (pedestrian accidents)
  │   └── ...
  └── col/   (vehicle collision accidents)
      └── ...
```

### Output

The script generates:
- Classification reports showing accuracy metrics
- Confusion matrix
- Detailed logs stored in the `logs` directory
