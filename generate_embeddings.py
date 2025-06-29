import os
import pickle
import cv2
import numpy as np
import argparse
import logging
import yaml
from tqdm import tqdm
from pathlib import Path
from deepface import DeepFace
from retinaface import RetinaFace

# -----------------------------
# Utility: Set up logging
# -----------------------------
def setup_logging(log_path):
    """Create logging directory and set up log format and handlers."""
    os.makedirs(log_path, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_path, 'embedding_generation.log')),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

# -----------------------------
# Utility: Load config from YAML
# -----------------------------
def load_config(config_path="config.yaml"):
    """Load configuration parameters from a YAML file."""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

# -----------------------------
# Step 1: Detect face using RetinaFace
# -----------------------------
def detect_face(image, detector_threshold=0.5):
    """Detect a single face in an image using RetinaFace."""
    try:
        faces = RetinaFace.detect_faces(image, threshold=detector_threshold)
        if not faces or isinstance(faces, tuple):
            return None
        
        # Use the first detected face
        face_data = next(iter(faces.values()))
        facial_area = face_data['facial_area']
        x1, y1, x2, y2 = facial_area
        
        # Crop the face region from image
        face_img = image[y1:y2, x1:x2]
        return face_img
    except Exception as e:
        return None

# -----------------------------
# Step 2: Extract embedding from face using DeepFace
# -----------------------------
def get_embedding(image, model_name="Facenet512"):
    """Generate facial embedding for a face image using DeepFace."""
    try:
        embedding = DeepFace.represent(
            img_path=image, 
            model_name=model_name,
            enforce_detection=False,
            detector_backend="retinaface"
        )
        return embedding[0]["embedding"]
    except Exception as e:
        return None

# -----------------------------
# Step 3: Loop through students and process all face images
# -----------------------------
def process_student_images(registered_path, detector_threshold=0.5):
    """Loop through all student image folders and extract average embeddings."""
    embeddings = {}

    # Get subdirectories for each student (each student has a folder)
    student_dirs = [d for d in os.listdir(registered_path) if os.path.isdir(os.path.join(registered_path, d))]
    
    for student in tqdm(student_dirs, desc="Processing students"):
        student_path = os.path.join(registered_path, student)
        student_embeddings = []

        # Load all image files for the current student
        image_files = [f for f in os.listdir(student_path) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        for img_file in image_files:
            try:
                # Read image from disk
                img_path = os.path.join(student_path, img_file)
                image = cv2.imread(img_path)
                
                if image is None:
                    continue

                # Detect and crop face
                face_img = detect_face(image, detector_threshold)
                if face_img is None:
                    continue

                # Extract embedding
                embedding = get_embedding(face_img)
                if embedding is not None:
                    student_embeddings.append(embedding)
            except Exception as e:
                continue
        
        # Average embeddings for this student (robust against variation)
        if student_embeddings:
            embeddings[student] = np.mean(student_embeddings, axis=0)
    
    return embeddings

# -----------------------------
# Main: Parse config, extract embeddings, and save
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Generate face embeddings for registered students")
    parser.add_argument('--config', type=str, default="config.yaml", help="Path to config file")
    args = parser.parse_args()
    
    # Load YAML config
    config = load_config(args.config)
    
    # Set up logging
    logger = setup_logging(config.get('log_path', 'logs'))
    logger.info("Starting embedding generation")

    # Extract embeddings for all registered students
    embeddings = process_student_images(
        config.get('registered_path'), 
        config.get('detector_threshold', 0.5)
    )

    # Save embeddings as a pickle file
    embeddings_path = config.get('embeddings_path')
    os.makedirs(os.path.dirname(embeddings_path), exist_ok=True)
    
    with open(embeddings_path, 'wb') as f:
        pickle.dump(embeddings, f)
    
    logger.info(f"Embeddings generated for {len(embeddings)} students and saved to {embeddings_path}")

# Run main if script is executed
if __name__ == "__main__":
    main()
