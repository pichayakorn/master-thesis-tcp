import os
import cv2
import numpy as np
import argparse
import random

def extract_frames(video_path, output_dir, frame_interval=1):
    """
    Extract frames from a video file and save them as images.
    
    :param video_path: Path to the video file.
    :param output_dir: Directory where extracted frames will be saved.
    :param frame_interval: Interval of frames to save (default is every frame).
    """
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception(f"Error opening video file: {video_path}")
    
    # Get the total number of frames in the video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Determine the padding length based on the total number of frames
    padding_length = len(str(total_frames))
    
    frame_count = 0
    saved_frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Save frame at the specified interval
        if frame_count % frame_interval == 0:
            frame_filename = os.path.join(output_dir, f"frame_{saved_frame_count:0{padding_length}d}.jpg")
            cv2.imwrite(frame_filename, frame)
            saved_frame_count += 1
        
        frame_count += 1
    
    cap.release()
    print(f"Saved {saved_frame_count} frames to '{output_dir}'")

def find_mp4_directories(data_dir):
    """
    Find all .mp4 files in the given directory and its subdirectories.
    
    :param data_dir: Root directory to search for .mp4 files.
    :return: List of paths to .mp4 files.
    """
    mp4_paths = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.mp4'):
                mp4_paths.append(os.path.join(root, file))
    return mp4_paths

def process_videos(data_dir, output_base_dir, frame_interval=1):
    """
    Process all .mp4 videos in the given directory and its subdirectories,
    extracting frames and saving them in a new directory structure.
    
    :param data_dir: Root directory containing subdirectories with .mp4 files.
    :param output_base_dir: Base directory where extracted frames will be saved.
    :param frame_interval: Interval of frames to save (default is every frame).
    """
    mp4_files = find_mp4_directories(data_dir)
    
    for video_path in mp4_files:
        # Derive the output directory path
        relative_path = os.path.relpath(video_path, data_dir)
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        output_dir = os.path.join(output_base_dir, os.path.dirname(relative_path), video_name)
        
        # Extract frames from the video
        extract_frames(video_path, output_dir, frame_interval)

def create_collages(input_dir, output_dir, grid_rows=2, grid_cols=2, with_frame_numbers=True, with_labels=True, randomize_frames=False):
    """
    Create collages from frames in input_dir.
    
    :param input_dir: Directory containing frames
    :param output_dir: Directory to save collages
    :param grid_rows: Number of rows in the collage grid
    :param grid_cols: Number of columns in the collage grid
    :param with_frame_numbers: Whether to display frame numbers
    :param with_labels: Whether to add black label backgrounds
    :param randomize_frames: Whether to randomize frame order
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Loop through all subfolders in the main directory
    for subfolder in os.listdir(input_dir):
        subfolder_path = os.path.join(input_dir, subfolder)
        if os.path.isdir(subfolder_path):
            # Loop through the next level of subfolders
            for inner_subfolder in os.listdir(subfolder_path):
                inner_subfolder_path = os.path.join(subfolder_path, inner_subfolder)
                if os.path.isdir(inner_subfolder_path):
                    # Get all frame files within this inner subfolder
                    frame_files = sorted(os.listdir(inner_subfolder_path))
                    total_frames = len(frame_files)
                    frames_per_collage = grid_rows * grid_cols
                    
                    # Calculate the number of collages needed
                    num_collages = total_frames // frames_per_collage + (1 if total_frames % frames_per_collage > 0 else 0)
                    padding_width = len(str(num_collages))

                    # Handle randomization if requested
                    if randomize_frames:
                        all_indices = list(range(total_frames))
                        random.shuffle(all_indices)
                    else:
                        all_indices = list(range(total_frames))

                    # Loop through each set of frames
                    for collage_index in range(num_collages):
                        start_idx = collage_index * frames_per_collage
                        end_idx = min(start_idx + frames_per_collage, total_frames)
                        group_indices = all_indices[start_idx:end_idx]

                        # Get frames for this collage
                        frames = []
                        frame_numbers = []
                        for idx in group_indices:
                            frame_path = os.path.join(inner_subfolder_path, frame_files[idx])
                            frame = cv2.imread(frame_path)
                            if frame is None:
                                print(f"Warning: Could not read frame {frame_path}")
                                continue
                            frames.append(frame)
                            frame_numbers.append(idx + 1)  # 1-based frame number

                        # Create collage only if there are frames
                        if frames:
                            # Determine the size of each frame
                            frame_height, frame_width = frames[0].shape[:2]
                            padding = 15
                            
                            # Calculate collage dimensions
                            collage_height = grid_rows * frame_height + (grid_rows + 1) * padding
                            collage_width = grid_cols * frame_width + (grid_cols + 1) * padding

                            # Create a blank image for the collage with padding
                            collage = np.full((collage_height, collage_width, 3), 255, dtype=np.uint8)  # White background

                            # Calculate positions for each frame in the grid
                            positions = []
                            for row in range(grid_rows):
                                for col in range(grid_cols):
                                    y = padding + row * (frame_height + padding)
                                    x = padding + col * (frame_width + padding)
                                    positions.append((y, x))

                            # Place frames in the collage with padding
                            for i, (frame, pos, frame_num) in enumerate(zip(frames, positions, frame_numbers)):
                                y, x = pos
                                collage[y:y+frame_height, x:x+frame_width] = frame
                                
                                if with_labels:
                                    # Add black highlight for text
                                    cv2.rectangle(collage, (x, y), (x + 80, y + 50), (0, 0, 0), -1)
                                
                                if with_frame_numbers:
                                    # Add frame number
                                    cv2.putText(
                                        collage,
                                        f'{frame_num}',  # Frame number
                                        (x + 10, y + 40),  # Position
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        1,
                                        (255, 255, 255),  # White color for text
                                        2,
                                        cv2.LINE_AA  # Thickness
                                    )

                            # Save the collage preserving the subfolder structure
                            output_subfolder = os.path.join(output_dir, subfolder, inner_subfolder)
                            os.makedirs(output_subfolder, exist_ok=True)
                            output_path = os.path.join(output_subfolder, f'collage_{collage_index + 1:0{padding_width}d}.jpg')
                            cv2.imwrite(output_path, collage)
                            print(f"Saved collage to '{output_path}'")


def main():
    parser = argparse.ArgumentParser(description='Video Processing and Collage Creation Tool')
    
    # Define mode: extract frames or create collages
    parser.add_argument('mode', choices=['extract', 'collage'], 
                        help='Mode: extract frames from videos or create collages from frames')
    
    # Common arguments
    parser.add_argument('--input-dir', required=True, help='Input directory containing videos or frames')
    parser.add_argument('--output-dir', required=True, help='Output directory for frames or collages')
    
    # Arguments for extract mode
    parser.add_argument('--frame-interval', type=int, default=1, 
                        help='For extract mode: Save every Nth frame (default: 1)')
    
    # Arguments for collage mode
    parser.add_argument('--grid', default='2x2', 
                        help='For collage mode: Grid layout as ROWSxCOLUMNS, e.g., 2x2, 3x2 (default: 2x2)')
    parser.add_argument('--with-frame-numbers', action='store_true', default=True,
                        help='For collage mode: Show frame numbers in collage (default: True)')
    parser.add_argument('--no-frame-numbers', action='store_false', dest='with_frame_numbers',
                        help='For collage mode: Do not show frame numbers in collage')
    parser.add_argument('--with-labels', action='store_true', default=True,
                        help='For collage mode: Add black label backgrounds (default: True)')
    parser.add_argument('--no-labels', action='store_false', dest='with_labels',
                        help='For collage mode: Do not add black label backgrounds')
    parser.add_argument('--randomize', action='store_true', default=False,
                        help='For collage mode: Randomize frame order (default: False)')
    
    args = parser.parse_args()
    
    if args.mode == 'extract':
        process_videos(args.input_dir, args.output_dir, args.frame_interval)
    elif args.mode == 'collage':
        # Parse grid dimensions
        try:
            rows, cols = map(int, args.grid.split('x'))
        except ValueError:
            print(f"Invalid grid format: {args.grid}. Must be ROWSxCOLUMNS, e.g., 2x2, 3x2")
            return
        
        create_collages(args.input_dir, args.output_dir, 
                        grid_rows=rows, 
                        grid_cols=cols,
                        with_frame_numbers=args.with_frame_numbers,
                        with_labels=args.with_labels,
                        randomize_frames=args.randomize)


if __name__ == "__main__":
    main()
