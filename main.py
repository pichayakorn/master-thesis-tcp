import os
import time
import sys
import argparse
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path
from loguru import logger
import cv2
import base64
import json
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Process video frames for accident detection using OpenAI models')
    parser.add_argument('--model', help='OpenAI model to use (e.g., gpt-4.1-2025-04-14, o3-2025-04-16)')
    parser.add_argument('--data_path', help='Path to the data directory containing norm, ped, and col subdirectories')
    parser.add_argument('--img_detail', default='high', choices=['low', 'high'], help='Image detail level (default: high)')
    
    args = parser.parse_args()
    
    # Load environment variables from .env file
    load_dotenv()

    # Access the API key using the variable name defined in the .env file
    api_key = os.getenv("OPENAI_API_KEY")

    # Check if the API key is loaded correctly
    if not api_key:
        logger.error("API key not found. Please ensure it's set in the .env file.")
        sys.exit(1)

    # Set the API key and model name from command line arguments
    MODEL = args.model
    client = OpenAI(api_key=api_key)
    img_detail = args.img_detail
    base_dir = args.data_path

    # Directories for each class
    directories = {
        "norm": os.path.join(base_dir, "norm"),
        "ped": os.path.join(base_dir, "ped"),
        "col": os.path.join(base_dir, "col")
    }

    # Validate that directories exist
    for label, dir_path in directories.items():
        if not os.path.exists(dir_path):
            logger.error(f"Directory not found: {dir_path}")
            sys.exit(1)

    # Define the log directory
    log_dir = Path("logs")

    # Create the log directory if it doesn't exist
    log_dir.mkdir(parents=True, exist_ok=True)

    # Define the log file name with the current timestamp
    log_file = log_dir / f"{MODEL}_{base_dir.replace('/', '_').replace('\\', '_')}_{img_detail}_log_{datetime.now():%Y-%m-%d-%H-%M}.log"

    # Configure loguru to save the log to the log file
    logger.add(log_file, format="{time} {level} {message}", level="DEBUG")
    logger.info('Start running the script.')
    logger.info(f'Model: {MODEL}')
    logger.info(f'Data path: {base_dir}')
    logger.info(f'Image detail: {img_detail}')

    # Function to process frames from a directory
    def process_frames_from_directory(directory_path):
        logger.info(f"Start processing frames from directory: {directory_path}")
        base64Frames = []
        for root, _, files in os.walk(directory_path):
            for file in sorted(files):
                if file.endswith(('.jpg', '.jpeg', '.png')):
                    frame_path = os.path.join(root, file)
                    frame = cv2.imread(frame_path)
                    if frame is None:
                        logger.warning(f"Could not read image: {frame_path}")
                        continue
                    _, buffer = cv2.imencode(".jpg", frame)

                    # Check the size of the image buffer
                    if buffer.nbytes > 20 * 1024 * 1024:
                        logger.error("Frame size exceeds 20 MB. Skipping frame.")
                        continue

                    base64Frames.append(base64.b64encode(buffer).decode("utf-8"))

        logger.debug(f"Processed {len(base64Frames)} frames")
        return base64Frames

    # Define the question
    QUESTION = """Your task is to first identify whether an accident occurs in the video. You need to classify it as either "Normal" or "Accident". If it's "Normal", you don't need to take any action. However, if it's an "Accident", please also specify the type of accident with the reason in detail. There are only two types of accidents: Type A: a car crashes into people who are crossing the street. Type B: a car crashes with another vehicle. Let's think step-by-step"""

    # Function to call OpenAI API for visual Q&A
    def chat_with_gpt(base64Frames):
        nonlocal PROMPT_TOKENS, COMPLETION_TOKENS
        retry_attempts = 5
        for attempt in range(retry_attempts):
            try:
                # Check if the model is o3 and use the appropriate API
                if MODEL.startswith("o3"):
                    logger.info("Start calling OpenAI API using Responses API...")
                    messages = [
                        {"role": "system", "content": """Use the video to answer the provided question. Respond in JSON with attributes: Class: {Normal|Accident}, Accident_type: {A|B|None}. Output in Markdown code block format (between ```json and ```)."""},
                        {"role":"user",
                            "content": [
                                {"type": "input_text", "text": "These are the frames from the video."},
                                *[
                                    {"type": "input_image", "image_url": f"data:image/jpg;base64,{x}"}
                                    for x in base64Frames
                                ],
                                {"type": "input_text", "text": QUESTION}
                            ]
                        }
                    ]
                    qa_visual_response = client.responses.create(
                        model=MODEL,
                        input=messages,
                        store=False,
                        reasoning={
                            "effort": "medium",
                            "summary": "detailed"
                        }
                    )

                    # Log the full qa_visual_response object as JSON
                    full_response_dict = qa_visual_response.model_dump() if hasattr(qa_visual_response, 'model_dump') else qa_visual_response
                    logger.info("--- Full OpenAI Responses API Response (JSON) ---")
                    logger.info(json.dumps(full_response_dict, indent=2))
                    response = qa_visual_response.output_text
                    input_tokens = qa_visual_response.usage.input_tokens
                    output_tokens = qa_visual_response.usage.output_tokens
                else:
                    logger.info("Start calling OpenAI API using Chat Completions API...")
                    messages = [
                        {"role": "system", "content": """Use the video to answer the provided question. Respond in JSON with attributes: Class: {Normal|Accident}, Accident_type: {A|B|None}. Output in Markdown code block format (between ```json and ```)."""},
                        {"role":"user","content":[
                          "These are the frames from the video.",
                          *map(lambda x:{"type":"image_url","image_url":{"url":f'data:image/jpg;base64,{x}',"detail":f"{img_detail}"}}, base64Frames),
                          QUESTION
                          ],
                        }
                    ]

                    qa_visual_response = client.chat.completions.create(
                        model=MODEL,
                        messages=messages,
                    )

                    response = qa_visual_response.choices[0].message.content
                    input_tokens = qa_visual_response.usage.prompt_tokens
                    output_tokens = qa_visual_response.usage.completion_tokens

                logger.debug(response)
                logger.debug(f"Prompt tokens: {input_tokens}")
                logger.debug(f"Completion tokens: {output_tokens}")
                
                # Extract response
                extracted_response = extract_response(response)
                PROMPT_TOKENS += input_tokens
                COMPLETION_TOKENS += output_tokens
                return extracted_response
            except Exception as e:
                logger.error(f"An error occurred during OpenAI API call: {e}", exc_info=True)
                if attempt < retry_attempts - 1:
                    logger.info("Retrying...")
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    logger.error("Max retry attempts reached. Skipping this video.")
                    return None

    # Extract response from GPT
    def extract_response(response):
        logger.info("Start extracting response")
        response = response.lower()
        try:
            # Try to parse the response directly as JSON
            json_obj = json.loads(response)
            return json_obj
        except json.JSONDecodeError:
            # If parsing fails, extract the JSON part of the response
            start_idx = response.find("```json")
            if start_idx != -1:
                start_idx += 7
                end_idx = response.rfind("```")
                json_str = response[start_idx:end_idx].strip()
                return json.loads(json_str)
            else:
                logger.error("Response does not contain valid JSON")
                raise ValueError("Response does not contain valid JSON")

    # Function to get label index
    def get_label_index(label):
        return {"norm": 0, "ped": 1, "col": 2}[label]

    true_labels = []
    pred_labels = []
    PROMPT_TOKENS = 0
    COMPLETION_TOKENS = 0

    # Process each directory and frames
    for label, dir_path in directories.items():
        logger.info(f"Processing label: {label}, directory: {dir_path}")
        subfolders = [f.path for f in os.scandir(dir_path) if f.is_dir()]
        for subfolder in subfolders:
            logger.info(f"Processing subfolder: {subfolder}")
            base64Frames = process_frames_from_directory(subfolder)
            if not base64Frames:
                logger.warning(f"No valid frames found in subfolder: {subfolder}")
                continue  # Skip if no valid frames were extracted
            response = chat_with_gpt(base64Frames)
            if response is None:
                logger.warning(f"No valid response received for subfolder: {subfolder}")
                continue  # Skip if no valid response was received
            prediction = response
            logger.debug("Strip json response:")
            logger.debug(prediction)

            true_labels.append(get_label_index(label))
            if prediction["class"] == "normal":
                pred_labels.append(get_label_index("norm"))
            else:
                if prediction["accident_type"] == "a":
                    pred_labels.append(get_label_index("ped"))
                elif prediction["accident_type"] == "b":
                    pred_labels.append(get_label_index("col"))

    # Ensure there are predictions and true labels before generating reports
    if true_labels and pred_labels:
        # Generate confusion matrix and classification report
        conf_matrix = confusion_matrix(true_labels, pred_labels, labels=[0, 1, 2])
        class_report = classification_report(true_labels, pred_labels, target_names=["Normal", "Pedestrian Accident", "Collision"])

        logger.info("Confusion Matrix:")
        logger.info(conf_matrix)
        logger.info("Classification Report:")
        logger.info(class_report)

        print("Confusion Matrix:")
        print(conf_matrix)
        print("Classification Report:")
        print(class_report)
    else:
        logger.error("No predictions or true labels to generate reports.")

    logger.debug(f"Total Input tokens: {PROMPT_TOKENS}")
    logger.debug(f"Total Output tokens: {COMPLETION_TOKENS}")

if __name__ == "__main__":
    main()
