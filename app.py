import streamlit as st
import os
import cv2
import pandas as pd
import numpy as np
import json
import torch
import torchvision
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import sys
import tempfile
from deepface import DeepFace
from retinaface import RetinaFace
from classroom_timeline import ClassroomTimelineAnalyzer
from groq_agent import GroqAgent
import os
import json
from datetime import datetime

# Set page configuration
st.set_page_config(
    page_title="Face Recognition & Analysis System",
    page_icon="👤",
    layout="wide"
)

# App title and description
st.title("Face Recognition, Emotion Analysis & Head Pose Estimation")
st.write("Upload an image to recognize faces, analyze emotions, and estimate head pose.")

# Add sidebar for configuration
st.sidebar.title("Settings")

def update_app_sidebar():
    # Add timeline folder setting in sidebar
    timeline_folder = st.sidebar.text_input(
        "Timeline Images Folder",
        value="D:\\BTP\\CompleteApp\\Dataset\\classroom_timeline",
        help="Path to folder containing chronological classroom images"
    )
    
    return timeline_folder

# Database path setting
db_path = st.sidebar.text_input(
    "Database Path",
    value="D:\\BTP\\CompleteApp\\Dataset\\registered_students",
    help="Path to folder containing registered student images in subfolders"
)

# Hopenet model path setting
hopenet_model_path = st.sidebar.text_input(
    "Hopenet Model Path",
    value="D:\\BTP\\CompleteApp\\models\\hopenet_robust_alpha1.pkl",
    help="Path to Hopenet model file"
)

# Setup Hopenet model for head pose estimation
@st.cache_resource
def setup_hopenet(model_path):
    """Set up the Hopenet model for head pose estimation"""
    if not model_path or not os.path.exists(model_path):
        st.sidebar.warning("Hopenet model path not specified or invalid. Head pose estimation will be skipped.")
        return None
    
    try:
        # Try to dynamically add deep-head-pose code directory
        model_dir = os.path.dirname(model_path)
        if model_dir not in sys.path:
            sys.path.append(model_dir)
        
        # Add the code subdirectory if it exists
        code_dir = os.path.join(model_dir, 'code')
        if os.path.exists(code_dir) and code_dir not in sys.path:
            sys.path.append(code_dir)
        
        try:
            # Import Hopenet now that path is added
            from hopenet import Hopenet
        except ImportError:
            st.sidebar.error("Could not import Hopenet module. Make sure the code is accessible.")
            return None
        
        # Model parameters
        num_bins = 66
        
        # Initialize ResNet-based Hopenet model
        model = Hopenet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], num_bins)
        
        # Load pre-trained weights from checkpoint
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        
        # Set model to evaluation mode
        model.eval()
        
        st.sidebar.success("Hopenet model loaded successfully!")
        return model
    except Exception as e:
        st.sidebar.error(f"Error setting up Hopenet: {str(e)}")
        st.sidebar.warning("Head pose estimation will be skipped.")
        return None

# Setup image transformation for Hopenet
def get_head_pose_transform():
    """Get image transformation pipeline for head pose estimation"""
    return transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])

# Head pose prediction function
def predict_head_pose(model, pil_image, transformations):
    """Predict head pose (yaw, pitch, roll) from a PIL image"""
    if model is None:
        return None, None, None
    
    try:
        # Prepare input
        input_tensor = transformations(pil_image).unsqueeze(0)
        
        # Get model predictions
        with torch.no_grad():  # No need to track gradients
            yaw, pitch, roll = model(input_tensor)
        
        # Process outputs
        softmax = torch.nn.Softmax(dim=1)
        idx_tensor = torch.arange(66, dtype=torch.float).unsqueeze(0)
        
        # Convert from bin index to angles
        yaw_pred = torch.sum(softmax(yaw) * idx_tensor, dim=1) * 3 - 99
        pitch_pred = torch.sum(softmax(pitch) * idx_tensor, dim=1) * 3 - 99
        roll_pred = torch.sum(softmax(roll) * idx_tensor, dim=1) * 3 - 99
        
        return yaw_pred.item(), pitch_pred.item(), roll_pred.item()
    except Exception as e:
        st.error(f"Error in head pose prediction: {str(e)}")
        return None, None, None

def debug_database(db_path):
    """Print information about the database to help diagnose issues"""
    st.subheader("Database Information")
    
    if not os.path.exists(db_path):
        st.error(f"ERROR: Database path does not exist!")
        return False

    # Check person folders
    person_folders = [f for f in os.listdir(db_path)
                     if os.path.isdir(os.path.join(db_path, f))]

    if not person_folders:
        st.error(f"ERROR: No person folders found in database!")
        return False

    st.success(f"Found {len(person_folders)} registered students")

    # Show database structure
    col1, col2 = st.columns(2)
    
    total_images = 0
    with col1:
        st.write("Students in database:")
        for person in person_folders:
            person_path = os.path.join(db_path, person)
            images = [f for f in os.listdir(person_path)
                    if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

            st.write(f"- {person}: {len(images)} images")
            total_images += len(images)

    with col2:
        st.metric("Total students", len(person_folders))
        st.metric("Total images", total_images)

    # Check for representations file
    repr_file = os.path.join(db_path, "representations_vgg_face.pkl")
    if os.path.exists(repr_file):
        st.write(f"Representations file exists: {repr_file}")
        st.write(f"Size: {os.path.getsize(repr_file) / (1024*1024):.2f} MB")
    else:
        st.write("Representations file does not exist - it will be created when needed")

    return True

def recognize_faces_analyze_emotions_and_pose(image, db_path, hopenet_model, transform, output_folder=None):
    """
    Recognize faces in an image by comparing with the database,
    analyze emotions and head pose for each detected face
    """
    # Create a progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    if output_folder is None:
        output_folder = tempfile.mkdtemp()
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    img_rgb = np.array(image)
    img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)  # Convert to BGR for OpenCV functions
    
    temp_filename = "temp_image.jpg"
    temp_filepath = os.path.join(output_folder, temp_filename)
    cv2.imwrite(temp_filepath, img)
    
    status_text.text("Processing image...")
    progress_bar.progress(10)
    
    annotated_img = img.copy()

    results = {
        'image_file': temp_filename,
        'faces': []
    }

    try:
        status_text.text("Detecting faces using RetinaFace...")
        raw_faces = RetinaFace.detect_faces(img_rgb)
        min_det_score = 0.9
        faces = {
            key: data
            for key, data in raw_faces.items()
            if data.get("score", 1.0) >= min_det_score
        }
        progress_bar.progress(30)

        if not faces:
            status_text.text("No faces detected in the image")
            results['total_faces'] = 0
            progress_bar.progress(100)
            return results

        status_text.text(f"Detected {len(faces)} faces. Processing...")
        results['total_faces'] = len(faces)

        face_counter = 0
        for face_key in faces.keys():
            face_counter += 1
            face_progress = 30 + (face_counter / len(faces)) * 60
            progress_bar.progress(int(face_progress))
            status_text.text(f"Processing face {face_counter} of {len(faces)}...")
            
            face_data = faces[face_key]

            # Get facial area coordinates
            x1, y1, x2, y2 = face_data["facial_area"]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            face_info = {
                'face_index': face_counter - 1,
                'position': {
                    'x': x1,
                    'y': y1,
                    'width': x2 - x1,
                    'height': y2 - y1
                }
            }

            # Add additional landmark points if available
            if "landmarks" in face_data:
                face_info['landmarks'] = {
                    k: [float(v[0]), float(v[1])] 
                    for k, v in face_data["landmarks"].items()
                }

            # Extract face
            detected_face = img_rgb[y1:y2, x1:x2]

            # FACE RECOGNITION
            identity = "Unknown"
            confidence = 0
            try:
                if os.path.exists(db_path) and len(os.listdir(db_path)) > 0:
                    status_text.text(f"Recognizing face {face_counter}...")
                    df = DeepFace.find(
                        img_path=detected_face,
                        db_path=db_path,
                        enforce_detection=False,
                        silent=True,
                        model_name="VGG-Face"
                    )

                    if len(df) > 0 and not df[0].empty:
                        match = df[0].iloc[0]
                        # Extract identity name and compute confidence
                        identity   = os.path.basename(os.path.dirname(match['identity']))
                        distance   = float(match['distance'])
                        confidence = 100 - (distance * 100)

                        # Enforce 50% threshold
                        if confidence < 30:
                            # Mark low-confidence match as unknown
                            face_info['recognition'] = {
                                'recognized': False,
                                'person': "Unknown",
                                'confidence': confidence,
                                'distance': distance,
                                'reference_image': os.path.basename(match['identity'])
                            }
                        else:
                            # High-confidence: accept recognition
                            face_info['recognition'] = {
                                'recognized': True,
                                'person': identity,
                                'confidence': confidence,
                                'distance': distance,
                                'reference_image': os.path.basename(match['identity'])
                            }
                    else:
                        # No matches at all
                        face_info['recognition'] = {
                            'recognized': False,
                            'person': "Unknown",
                            'confidence': 0.0,
                            'distance': 1.0,
                            'reference_image': None
                        }

            except Exception as e:
                st.error(f"Recognition error for face {face_counter}: {str(e)}")
                face_info['recognition'] = {
                    'recognized': False,
                    'person': "Unknown",
                    'confidence': 0.0,
                    'distance': 1.0,
                    'reference_image': None,
                    'error': str(e)
                }

            # EMOTION ANALYSIS
            try:
                status_text.text(f"Analyzing emotions for face {face_counter}...")
                emotion_analysis = DeepFace.analyze(
                    img_path=detected_face,
                    actions=['emotion'],
                    enforce_detection=False,
                    detector_backend="skip"
                )

                if emotion_analysis and isinstance(emotion_analysis, list) and len(emotion_analysis) > 0:
                    emotions = emotion_analysis[0]['emotion']
                    dominant_emotion = max(emotions, key=lambda k: emotions[k])

                    face_info['emotion'] = {
                        'dominant': dominant_emotion,
                        'scores': {emotion: float(score) for emotion, score in emotions.items()}
                    }
                else:
                    face_info['emotion'] = {
                        'dominant': "Unknown",
                        'scores': {}
                    }
            except Exception as e:
                st.error(f"Emotion analysis error for face {face_counter}: {str(e)}")
                face_info['emotion'] = {
                    'dominant': "Error",
                    'scores': {},
                    'error': str(e)
                }

            # HEAD POSE ESTIMATION
            try:
                if hopenet_model is not None:
                    status_text.text(f"Estimating head pose for face {face_counter}...")
                    # Add padding around face for better pose estimation
                    padding = int(max(x2-x1, y2-y1) * 0.2)
                    pad_x1 = max(0, x1 - padding)
                    pad_y1 = max(0, y1 - padding)
                    pad_x2 = min(img_rgb.shape[1], x2 + padding)
                    pad_y2 = min(img_rgb.shape[0], y2 + padding)
                    
                    # Extract padded face
                    padded_face = img_rgb[pad_y1:pad_y2, pad_x1:pad_x2]
                    padded_pil_face = Image.fromarray(padded_face)
                    
                    # Predict head pose
                    yaw, pitch, roll = predict_head_pose(hopenet_model, padded_pil_face, transform)
                    
                    if yaw is not None and pitch is not None and roll is not None:
                        face_info['head_pose'] = {
                            'yaw': float(yaw),    # Left/Right
                            'pitch': float(pitch), # Up/Down
                            'roll': float(roll)    # Tilt
                        }
                    else:
                        face_info['head_pose'] = {
                            'yaw': None,
                            'pitch': None,
                            'roll': None,
                            'error': "Failed to estimate head pose"
                        }
                else:
                    face_info['head_pose'] = {
                        'yaw': None,
                        'pitch': None,
                        'roll': None,
                        'error': "Hopenet model not loaded"
                    }
            except Exception as e:
                st.error(f"Head pose estimation error for face {face_counter}: {str(e)}")
                face_info['head_pose'] = {
                    'yaw': None,
                    'pitch': None,
                    'roll': None,
                    'error': str(e)
                }

            results['faces'].append(face_info)

            # Draw bounding box
            color = (0, 255, 0) if face_info['recognition'].get('recognized', False) else (0, 0, 255)
            cv2.rectangle(
                annotated_img,
                (x1, y1),
                (x2, y2),
                color,
                2
            )

            # Create label with name and confidence
            person = face_info['recognition'].get('person', 'Unknown')
            confidence = face_info['recognition'].get('confidence', 0)

            label = f"{person}"
            if person != "Unknown":
                label += f" ({confidence:.2f}%)"

            # Add label above the face
            cv2.putText(
                annotated_img,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )

            # Add emotion label below face
            emotion_text = face_info['emotion'].get('dominant', 'Unknown')
            cv2.putText(
                annotated_img,
                f"Emotion: {emotion_text}",
                (x1, y2 + 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2
            )
            
            # Add head pose info if available
            if 'head_pose' in face_info and face_info['head_pose']['yaw'] is not None:
                pose_text = f"Y:{face_info['head_pose']['yaw']:.1f} P:{face_info['head_pose']['pitch']:.1f} R:{face_info['head_pose']['roll']:.1f}"
                cv2.putText(
                    annotated_img,
                    pose_text,
                    (x1, y2 + 55),  # Position below emotion text
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2
                )
                
                # Visualize head pose with directional lines if landmarks available
                if 'landmarks' in face_info:
                    face_center = face_info['landmarks'].get('nose', [x1 + (x2-x1)//2, y1 + (y2-y1)//2])
                    cx, cy = int(face_center[0]), int(face_center[1])
                    
                    # Calculate vector lengths based on face size
                    face_size = max(x2-x1, y2-y1)
                    line_length = face_size * 0.5
                    
                    # Draw yaw line (left/right - blue)
                    yaw_rad = np.deg2rad(face_info['head_pose']['yaw'])
                    yaw_x = int(cx - line_length * np.sin(yaw_rad))
                    yaw_y = int(cy - line_length * np.cos(yaw_rad))
                    cv2.line(annotated_img, (cx, cy), (yaw_x, yaw_y), (255, 0, 0), 2)
                    
                    # Draw pitch line (up/down - green)
                    pitch_rad = np.deg2rad(face_info['head_pose']['pitch'])
                    pitch_x = int(cx + line_length * np.sin(pitch_rad))
                    pitch_y = int(cy + line_length * np.cos(pitch_rad))
                    cv2.line(annotated_img, (cx, cy), (pitch_x, pitch_y), (0, 255, 0), 2)
                    
                    # Draw roll line (tilt - red)
                    roll_rad = np.deg2rad(face_info['head_pose']['roll'])
                    roll_x = int(cx + line_length * np.cos(roll_rad))
                    roll_y = int(cy + line_length * np.sin(roll_rad))
                    cv2.line(annotated_img, (cx, cy), (roll_x, roll_y), (0, 0, 255), 2)

    except Exception as e:
        status_text.text(f"Error processing image: {str(e)}")
        results['error'] = str(e)
        progress_bar.progress(100)
        return results

    # Save annotated image
    annotated_path = os.path.join(output_folder, f"annotated_{temp_filename}")
    cv2.imwrite(annotated_path, annotated_img)
    results['annotated_image'] = annotated_path
    
    # Convert the annotated image back to RGB for display
    results['annotated_img_rgb'] = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
    
    status_text.text("Processing complete!")
    progress_bar.progress(100)
    
    return results

def display_results(results):
    """Display face recognition and analysis results in Streamlit"""
    if not results:
        st.error("No results to display.")
        return
    
    if 'error' in results:
        st.error(f"Error during image processing: {results['error']}")
        return
    
    st.header("Analysis Results")
    st.subheader(f"Detected {results['total_faces']} face(s)")
    
    # Display annotated image
    if 'annotated_img_rgb' in results:
        st.image(results['annotated_img_rgb'], caption="Detected Faces", use_column_width=True)
    
    # Display individual faces and details
    if results['faces']:
        st.subheader("Face Details")
        
        # Create tabs for each face
        face_tabs = st.tabs([f"Face {i+1}" for i in range(len(results['faces']))])
        
        for i, (tab, face) in enumerate(zip(face_tabs, results['faces'])):
            with tab:
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    # Recognition details
                    st.subheader("Recognition")
                    if 'recognition' in face:
                        rec = face['recognition']
                        recognition_status = 'Recognized' if rec.get('recognized', False) else 'Unrecognized'
                        
                        st.write(f"Status: {recognition_status}")
                        st.write(f"Identity: {rec.get('person', 'Unknown')}")
                        
                        if 'confidence' in rec:
                            st.write(f"Confidence: {rec['confidence']:.1f}%")
                        
                        if 'error' in rec:
                            st.error(f"Error: {rec['error']}")
                
                with col2:
                    # Emotion analysis details
                    st.subheader("Emotion Analysis")
                    if 'emotion' in face:
                        emo = face['emotion']
                        st.write(f"Dominant Emotion: {emo.get('dominant', 'Unknown')}")
                        
                        if 'scores' in emo and emo['scores']:
                            # Create a bar chart for emotion scores
                            emotion_data = {
                                'Emotion': list(emo['scores'].keys()),
                                'Score': [score * 100 if score <= 1 else score for score in emo['scores'].values()]
                            }
                            
                            chart_data = pd.DataFrame(emotion_data)
                            chart_data = chart_data.sort_values('Score', ascending=False)
                            
                            st.bar_chart(chart_data.set_index('Emotion'))
                        
                        if 'error' in emo:
                            st.error(f"Error: {emo['error']}")
                
                # Head pose estimation (full width)
                st.subheader("Head Pose Estimation")
                if 'head_pose' in face:
                    pose = face['head_pose']
                    if pose['yaw'] is not None:
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Yaw (Left/Right)", f"{pose['yaw']:.1f}°")
                        
                        with col2:
                            st.metric("Pitch (Up/Down)", f"{pose['pitch']:.1f}°")
                        
                        with col3:
                            st.metric("Roll (Tilt)", f"{pose['roll']:.1f}°")
                        
                        # Visualization placeholder
                        st.write("Visualization interpretation:")
                        st.write("- Yaw: Negative values indicate the person is looking left, positive values indicate looking right")
                        st.write("- Pitch: Negative values indicate the person is looking up, positive values indicate looking down")
                        st.write("- Roll: Values indicate the head tilt (clockwise/counterclockwise)")
                    else:
                        st.write("Head pose estimation not available")
                        if 'error' in pose:
                            st.error(f"Error: {pose['error']}")
                else:
                    st.write("Head pose information not available")



# Add this function to process face recognition results for Groq
def process_results_for_groq(results, class_roster=None):
    """Process face recognition results into a format suitable for Groq analysis"""
    # Current date and time
    current_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Extract recognized and unknown students
    recognized_students = []
    unknown_count = 0
    all_emotions = []
    
    for face in results.get('faces', []):
        if face.get('recognition', {}).get('recognized', False):
            student_name = face.get('recognition', {}).get('person', 'Unknown')
            recognized_students.append(student_name)
            
            # Get emotion data
            emotion_data = face.get('emotion', {})
            if emotion_data and 'scores' in emotion_data:
                dominant_emotion = emotion_data.get('dominant', 'Unknown')
                all_emotions.append(dominant_emotion)
        else:
            unknown_count += 1
    
    # Calculate most common emotion
    most_common_emotion = max(set(all_emotions), key=all_emotions.count) if all_emotions else "Unknown"
    
    # Calculate absent students if roster is provided
    absent_students = []
    if class_roster:
        absent_students = [student for student in class_roster if student not in recognized_students]
    
    # Calculate engagement scores based on emotions
    # This is a simplified model - you might want to customize this
    engagement_scores = []
    student_engagement = {}
    
    for face in results.get('faces', []):
        if face.get('recognition', {}).get('recognized', False):
            student_name = face.get('recognition', {}).get('person', 'Unknown')
            emotion = face.get('emotion', {}).get('dominant', 'neutral')
            
            # Simple engagement score calculation
            # This is very basic - you might want to use a more sophisticated approach
            engagement_score = 0.0
            if emotion == "happy":
                engagement_score = 0.9
            elif emotion == "neutral":
                engagement_score = 0.7
            elif emotion == "surprise":
                engagement_score = 0.8
            elif emotion == "sad":
                engagement_score = 0.4
            elif emotion == "angry":
                engagement_score = 0.3
            elif emotion == "fear":
                engagement_score = 0.2
            elif emotion == "disgust":
                engagement_score = 0.3
            else:
                engagement_score = 0.5
                
            engagement_scores.append(engagement_score)
            student_engagement[student_name] = engagement_score
    
    # Calculate average engagement
    avg_engagement = sum(engagement_scores) / len(engagement_scores) if engagement_scores else 0.0
    
    # Find most and least engaged students
    most_engaged_student = max(student_engagement.items(), key=lambda x: x[1])[0] if student_engagement else "None"
    least_engaged_student = min(student_engagement.items(), key=lambda x: x[1])[0] if student_engagement else "None"
    
    # Create detailed emotion data for each student
    student_details = []
    for face in results.get('faces', []):
        if face.get('recognition', {}).get('recognized', False):
            student_name = face.get('recognition', {}).get('person', 'Unknown')
            emotion_data = face.get('emotion', {}).get('scores', {})
            
            # Sort emotions by score
            sorted_emotions = sorted(emotion_data.items(), key=lambda x: x[1], reverse=True)
            
            primary_emotion = sorted_emotions[0] if len(sorted_emotions) > 0 else ("unknown", 0)
            secondary_emotion = sorted_emotions[1] if len(sorted_emotions) > 1 else ("unknown", 0)
            
            head_pose = face.get('head_pose', {})
            yaw = head_pose.get('yaw')
            pitch = head_pose.get('pitch')
            
            # Determine attention direction based on head pose
            attention_direction = "forward"
            if yaw is not None and pitch is not None:
                if abs(yaw) > 20:
                    attention_direction = "left" if yaw < 0 else "right"
                elif abs(pitch) > 20:
                    attention_direction = "up" if pitch < 0 else "down"
            
            student_details.append({
                "name": student_name,
                "primary_emotion": {
                    "emotion": primary_emotion[0],
                    "confidence": primary_emotion[1]
                },
                "secondary_emotion": {
                    "emotion": secondary_emotion[0],
                    "confidence": secondary_emotion[1]
                },
                "engagement_score": student_engagement.get(student_name, 0.0),
                "attention_direction": attention_direction
            })
    
    # Create the insights dictionary
    insights = {
        "class_session": {
            "date": current_date,
            "total_faces_detected": results.get('total_faces', 0)
        },
        "attendance": {
            "present_students": recognized_students,
            "unknown_individuals": unknown_count,
            "absent_students": absent_students
        },
        "engagement": {
            "most_common_emotion": most_common_emotion,
            "average_engagement_score": avg_engagement,
            "most_engaged_student": most_engaged_student,
            "least_engaged_student": least_engaged_student
        },
        "student_details": student_details
    }
    
    return insights

def add_timeline_tab():
    # Main app tabs - add a new "Timeline Analysis" tab
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Face Recognition", "Database Info", 
        "Classroom Analysis", "Timeline Analysis", "About"
    ])
    
    return tab1, tab2, tab3, tab4, tab5
# Add this function for the timeline analysis tab
def timeline_analysis_tab(tab, timeline_folder, db_path, hopenet_model, transform):
    with tab:
        st.header("Classroom Timeline Analysis")
        st.write("Analyze multiple classroom images over time to track engagement trends.")
        
        # Create columns for configuration
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Timeline Configuration")
            st.write(f"Current timeline folder: {timeline_folder}")
            
            if not os.path.exists(timeline_folder):
                st.warning(f"Timeline folder does not exist: {timeline_folder}")
                if st.button("Create Timeline Folder"):
                    try:
                        os.makedirs(timeline_folder)
                        st.success(f"Created timeline folder at {timeline_folder}")
                    except Exception as e:
                        st.error(f"Error creating folder: {str(e)}")
        
        with col2:
            # Add a file uploader for class roster
            roster_file = st.file_uploader("Upload Class Roster (JSON)", type=["json"], 
                                         help="Upload a JSON file with the class roster",
                                         key="class_analysis_roster")
            
            # Process the roster file
            class_roster = []
            if roster_file is not None:
                try:
                    class_roster = json.load(roster_file)
                    st.success(f"Loaded roster with {len(class_roster)} students")
                except Exception as e:
                    st.error(f"Error loading roster: {str(e)}")
        
        # Image uploader for adding images to timeline
        st.subheader("Add Images to Timeline")
        uploaded_files = st.file_uploader("Upload image(s) to timeline", type=["jpg", "jpeg", "png"], 
                                       accept_multiple_files=True)
        
        if uploaded_files:
            if st.button("Add to Timeline"):
                added_count = 0
                for uploaded_file in uploaded_files:
                    try:
                        # Create timestamp-based filename
                        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                        file_extension = os.path.splitext(uploaded_file.name)[1]
                        new_filename = f"{timestamp}{file_extension}"
                        
                        # Save file to timeline folder
                        with open(os.path.join(timeline_folder, new_filename), "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        
                        added_count += 1
                    except Exception as e:
                        st.error(f"Error saving {uploaded_file.name}: {str(e)}")
                
                st.success(f"Added {added_count} image(s) to timeline")
        
        # Camera option for adding images to timeline
        camera_option = st.checkbox("Use Camera to Add Timeline Image", value=False)
        if camera_option:
            camera_input = st.camera_input("Take a picture for timeline")
            if camera_input:
                try:
                    # Create timestamp-based filename
                    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                    new_filename = f"{timestamp}.jpg"
                    
                    # Save file to timeline folder
                    with open(os.path.join(timeline_folder, new_filename), "wb") as f:
                        f.write(camera_input.getbuffer())
                    
                    st.success(f"Added camera image to timeline as {new_filename}")
                except Exception as e:
                    st.error(f"Error saving camera image: {str(e)}")
        
        # Timeline processing section
        st.subheader("Process Timeline")
        
        # Initialize timeline analyzer
        timeline_analyzer = ClassroomTimelineAnalyzer(
            timeline_folder=timeline_folder,
            recognition_function=recognize_faces_analyze_emotions_and_pose,
            process_fn=process_results_for_groq
        )
        
        # Get list of available images
        image_files = timeline_analyzer.get_image_files()
        
        if not image_files:
            st.warning("No images found in the timeline folder")
        else:
            st.write(f"Found {len(image_files)} images in timeline folder")
            
            if st.button("Process Timeline"):
                with st.spinner("Processing timeline images..."):
                    # Process timeline
                    analysis_results = timeline_analyzer.process_timeline(
                        db_path=db_path,
                        hopenet_model=hopenet_model,
                        transform=transform,
                        class_roster=class_roster
                    )
                    
                    # Store results in session state
                    st.session_state.timeline_analysis = analysis_results
                    
                    # Generate visualizations
                    plot_path = timeline_analyzer.generate_visualizations()
                    if plot_path and os.path.exists(plot_path):
                        st.session_state.timeline_plot = plot_path
                    
                    # Export data
                    export_path = timeline_analyzer.export_timeline_data()
                    if export_path:
                        st.session_state.timeline_export = export_path
                    
                    st.success("Timeline processing complete!")
        
        # Display timeline analysis
        if 'timeline_analysis' in st.session_state:
            st.subheader("Timeline Analysis Results")
            
            # Display summary
            summary = st.session_state.timeline_analysis.get("timeline_summary", {})
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Sessions Analyzed", summary.get("num_sessions", 0))
            
            with col2:
                engagement_trend = summary.get("trends", {}).get("engagement", {})
                direction = engagement_trend.get("direction", "neutral")
                percent = engagement_trend.get("percent_change", 0)
                trend_icon = "↑" if direction == "increasing" else "↓" if direction == "decreasing" else "-"
                st.metric("Engagement Trend", f"{trend_icon} {percent:.1f}%")
            
            with col3:
                attendance_trend = summary.get("trends", {}).get("attendance", {})
                direction = attendance_trend.get("direction", "neutral")
                percent = attendance_trend.get("percent_change", 0)
                trend_icon = "↑" if direction == "increasing" else "↓" if direction == "decreasing" else "-"
                st.metric("Attendance Trend", f"{trend_icon} {percent:.1f}%")
            
            # Student insights
            st.subheader("Student Insights")
            student_insights = summary.get("student_insights", {})
            
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"Most improved student: **{student_insights.get('most_improved', 'N/A')}**")
            
            with col2:
                st.write(f"Most declined student: **{student_insights.get('most_declined', 'N/A')}**")
            
            # Display plot
            if 'timeline_plot' in st.session_state and os.path.exists(st.session_state.timeline_plot):
                st.subheader("Timeline Visualizations")
                st.image(st.session_state.timeline_plot, caption="Engagement, Attendance and Emotions Over Time")
            
            # Option to download analysis
            if 'timeline_export' in st.session_state and os.path.exists(st.session_state.timeline_export):
                with open(st.session_state.timeline_export, "rb") as f:
                    timeline_data = f.read()
                
                st.download_button(
                    label="Download Timeline Analysis (JSON)",
                    data=timeline_data,
                    file_name="classroom_timeline_analysis.json",
                    mime="application/json"
                )
        
        # Timeline image gallery
        if image_files:
            st.subheader("Timeline Image Gallery")
            
            # Create columns for gallery
            cols = st.columns(3)
            
            for i, image_file in enumerate(image_files):
                col_idx = i % 3
                with cols[col_idx]:
                    image_path = os.path.join(timeline_folder, image_file)
                    timestamp = timeline_analyzer._extract_timestamp_from_filename(image_file)
                    formatted_time = timestamp.strftime("%Y-%m-%d %H:%M:%S")
                    
                    # Get analysis result path if exists
                    result_path = os.path.join(timeline_folder, "analysis_results", f"annotated_{image_file}")
                    
                    # Show annotated image if available, otherwise original
                    display_path = result_path if os.path.exists(result_path) else image_path
                    
                    st.image(display_path, caption=f"{formatted_time}", use_column_width=True)
                    
                    # Option to view details
                    if st.button(f"View Details {i+1}", key=f"details_{i}"):
                        st.session_state.selected_timeline_image = i
            
            # Display selected image details
            if 'selected_timeline_image' in st.session_state:
                idx = st.session_state.selected_timeline_image
                if 0 <= idx < len(image_files):
                    selected_file = image_files[idx]
                    st.subheader(f"Details for {selected_file}")
                    
                    # Find corresponding result file
                    base_name = os.path.splitext(selected_file)[0]
                    result_file = os.path.join(timeline_folder, "analysis_results", f"result_{base_name}.json")
                    
                    if os.path.exists(result_file):
                        with open(result_file, 'r') as f:
                            result_data = json.load(f)
                        
                        # Display result data
                        if 'faces' in result_data:
                            st.write(f"Detected {len(result_data['faces'])} faces")
                            
                            # Create tabs for each face
                            if result_data['faces']:
                                face_tabs = st.tabs([f"Face {i+1}" for i in range(len(result_data['faces']))])
                                
                                for i, (tab, face) in enumerate(zip(face_tabs, result_data['faces'])):
                                    with tab:
                                        col1, col2 = st.columns([1, 1])
                                        
                                        with col1:
                                            st.subheader("Recognition")
                                            if 'recognition' in face:
                                                rec = face['recognition']
                                                st.write(f"Identity: {rec.get('person', 'Unknown')}")
                                                if 'confidence' in rec:
                                                    st.write(f"Confidence: {rec['confidence']:.1f}%")
                                        
                                        with col2:
                                            st.subheader("Emotion")
                                            if 'emotion' in face:
                                                emo = face['emotion']
                                                st.write(f"Dominant: {emo.get('dominant', 'Unknown')}")
                                                
                                                if 'scores' in emo and emo['scores']:
                                                    emotion_data = {
                                                        'Emotion': list(emo['scores'].keys()),
                                                        'Score': [score * 100 if score <= 1 else score for score in emo['scores'].values()]
                                                    }
                                                    
                                                    chart_data = pd.DataFrame(emotion_data)
                                                    chart_data = chart_data.sort_values('Score', ascending=False)
                                                    
                                                    st.bar_chart(chart_data.set_index('Emotion'))
                                        
                                        # Head pose
                                        if 'head_pose' in face and face['head_pose'].get('yaw') is not None:
                                            pose = face['head_pose']
                                            col1, col2, col3 = st.columns(3)
                                            
                                            with col1:
                                                st.metric("Yaw", f"{pose['yaw']:.1f}°")
                                            
                                            with col2:
                                                st.metric("Pitch", f"{pose['pitch']:.1f}°")
                                            
                                            with col3:
                                                st.metric("Roll", f"{pose['roll']:.1f}°")
                    else:
                        st.warning(f"No analysis results found for {selected_file}")

def main():
    # Global imports
    import pandas as pd
    import torch

    # Load Hopenet model based on user input
    timeline_folder = update_app_sidebar()
    hopenet_model = setup_hopenet(hopenet_model_path)
    hopenet_transform = get_head_pose_transform()
    
    # Main app tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Face Recognition", "Database Info", 
        "Classroom Analysis", "Timeline Analysis", "About"
    ])
    
    # Tab 1: Face Recognition
    with tab1:
        st.header("Upload Image for Analysis")
        uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
        
        col1, col2 = st.columns(2)
        
        with col1:
            camera_option = st.checkbox("Use Camera Instead", value=False)
        
        with col2:
            if camera_option:
                camera_input = st.camera_input("Take a picture")
                if camera_input:
                    uploaded_file = camera_input
        
        if uploaded_file is not None:
            # Display the uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Process button
            if st.button("Process Image"):
                with st.spinner("Processing..."):
                    # Process the image
                    results = recognize_faces_analyze_emotions_and_pose(
                        image, db_path, hopenet_model, hopenet_transform
                    )

                    st.session_state.face_recognition_results = results
                    
                    # Display results
                    display_results(results)
                    
                    # Option to download results as JSON
                    if results and 'annotated_image' in results:
                        # Create a clean version of results for download
                        download_results = results.copy()
                        if 'annotated_img_rgb' in download_results:
                            del download_results['annotated_img_rgb']
                        
                        # Convert to JSON string
                        results_json = json.dumps(download_results, indent=2)
                        st.download_button(
                            label="Download Results JSON",
                            data=results_json,
                            file_name="face_analysis_results.json",
                            mime="application/json"
                        )
    
    # Tab 2: Database Info
    with tab2:
        st.header("Database Information")
        
        if st.button("Scan Database"):
            debug_database(db_path)
    
    # Tab 3: About
    with tab3:
        st.header("Classroom Analysis & Chatbot")
        
        # Configuration section
        st.subheader("Configuration")
        col1, col2 = st.columns(2)
        
        with col1:
            api_key = st.text_input("Groq API Key", type="password", 
                                  help="Enter your Groq API key")
            
        with col2:
            model_options = ["llama3-8b-8192", "mixtral-8x7b-32768", "gemma-7b-it"]
            selected_model = st.selectbox("Select Groq Model", model_options)
        
        # File uploader for class roster
        roster_file = st.file_uploader("Upload Class Roster (JSON)", type=["json"], 
                                     help="Upload a JSON file with the class roster",
                                     key="timeline_analysis_roster")
        
        # Process the roster file
        class_roster = []
        if roster_file is not None:
            try:
                class_roster = json.load(roster_file)
                st.success(f"Loaded roster with {len(class_roster)} students")
            except Exception as e:
                st.error(f"Error loading roster: {str(e)}")
        
        # Display results and chatbot if results exist
        st.subheader("Class Analysis")
        
        # Check if we have results from face recognition
        if 'face_recognition_results' in st.session_state:
            results = st.session_state.face_recognition_results
            
            # Process results for GroqAgent
            insights = process_results_for_groq(results, class_roster)
            
            # Display insights summary
            st.json(insights)
            
            # Initialize GroqAgent if API key is provided
            if api_key:
                try:
                    agent = GroqAgent(api_key=api_key, model=selected_model)
                    
                    # Chatbot interface
                    st.subheader("Ask Questions About Your Class")
                    user_question = st.text_input("Ask a question about classroom attendance or engagement")
                    
                    if st.button("Get Analysis") and user_question:
                        with st.spinner("Analyzing..."):
                            analysis = agent.get_analysis(insights, user_question)
                            st.markdown("### Analysis")
                            st.markdown(analysis)
                            
                            # Save chat history
                            if 'chat_history' not in st.session_state:
                                st.session_state.chat_history = []
                            
                            st.session_state.chat_history.append({
                                "question": user_question,
                                "answer": analysis
                            })
                    
                    # Display chat history
                    if 'chat_history' in st.session_state and st.session_state.chat_history:
                        st.subheader("Chat History")
                        for i, chat in enumerate(st.session_state.chat_history):
                            with st.expander(f"Q: {chat['question'][:50]}...", expanded=i==len(st.session_state.chat_history)-1):
                                st.markdown(f"**Question:** {chat['question']}")
                                st.markdown(f"**Answer:** {chat['answer']}")
                
                except Exception as e:
                    st.error(f"Error initializing Groq Agent: {str(e)}")
            else:
                st.warning("Please enter your Groq API key to enable the chatbot")
        else:
            st.info("No face recognition results available. Please use the Face Recognition tab to analyze an image first.")
    timeline_analysis_tab(tab4, timeline_folder, db_path, hopenet_model, hopenet_transform)


if __name__ == "__main__":
    main()