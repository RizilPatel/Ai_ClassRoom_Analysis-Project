import os
import cv2
import pandas as pd
import numpy as np
import json
from datetime import datetime
import matplotlib.pyplot as plt
from PIL import Image
import streamlit as st
import shutil

class ClassroomTimelineAnalyzer:
    """
    Analyze classroom timeline using periodic image captures.
    Performs attendance, emotion recognition, engagement tracking, and visualizes trends.
    """

    def __init__(self, timeline_folder, recognition_function, process_fn):
        """
        Initialize the analyzer with timeline image folder and recognition functions.
        
        Args:
            timeline_folder (str): Folder containing classroom images captured over time.
            recognition_function (callable): Function to perform face recognition and annotate results.
            process_fn (callable): Function to extract attendance and engagement from recognition result.
        """
        self.timeline_folder = timeline_folder
        self.recognition_function = recognition_function
        self.process_results_fn = process_fn
        self.results_folder = os.path.join(timeline_folder, "analysis_results")
        self.timeline_data = []

        # Create a folder to store results
        os.makedirs(self.results_folder, exist_ok=True)

    def _is_valid_image(self, filename):
        """Check if the file is a valid image format."""
        valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        ext = os.path.splitext(filename)[1].lower()
        return ext in valid_extensions

    def _extract_timestamp_from_filename(self, filename):
        """
        Extract timestamp from filename assuming the format: YYYY-MM-DD_HH-MM-SS.jpg.
        Fallback to file creation time if the format is invalid.
        """
        try:
            base_name = os.path.splitext(filename)[0]
            timestamp = datetime.strptime(base_name, "%Y-%m-%d_%H-%M-%S")
            return timestamp
        except ValueError:
            filepath = os.path.join(self.timeline_folder, filename)
            creation_time = os.path.getctime(filepath)
            return datetime.fromtimestamp(creation_time)

    def get_image_files(self):
        """Return a sorted list of valid image files from the timeline folder."""
        if not os.path.exists(self.timeline_folder):
            return []
        files = [
            f for f in os.listdir(self.timeline_folder)
            if os.path.isfile(os.path.join(self.timeline_folder, f)) and self._is_valid_image(f)
        ]
        # Sort by extracted timestamp
        files.sort(key=lambda f: self._extract_timestamp_from_filename(f))
        return files

    def process_timeline(self, db_path, hopenet_model=None, transform=None, class_roster=None):
        """
        Process all timeline images and generate analysis results.
        
        Args:
            db_path (str): Path to student face database.
            hopenet_model: Optional head pose model.
            transform: Image preprocessing transformation.
            class_roster (list): List of valid student names.
        """
        image_files = self.get_image_files()
        if not image_files:
            return {"error": "No images found in timeline folder"}

        progress_bar = st.progress(0)
        status_text = st.empty()
        self.timeline_data = []

        for i, image_file in enumerate(image_files):
            progress = (i + 1) / len(image_files)
            progress_bar.progress(progress)
            status_text.text(f"Processing image {i+1}/{len(image_files)}: {image_file}")

            image_path = os.path.join(self.timeline_folder, image_file)
            image = Image.open(image_path)
            timestamp = self._extract_timestamp_from_filename(image_file)

            # Run face recognition and emotion analysis
            result = self.recognition_function(
                image=image,
                db_path=db_path,
                hopenet_model=hopenet_model,
                transform=transform
            )

            # Save analysis result JSON
            result_filename = f"result_{os.path.splitext(image_file)[0]}.json"
            result_path = os.path.join(self.results_folder, result_filename)

            save_result = result.copy()
            if 'annotated_img_rgb' in save_result:
                del save_result['annotated_img_rgb']

            with open(result_path, 'w') as f:
                json.dump(save_result, f, indent=2)

            # Save annotated image
            if 'annotated_img_rgb' in result:
                annotated_img = cv2.cvtColor(result['annotated_img_rgb'], cv2.COLOR_RGB2BGR)
                annotated_path = os.path.join(self.results_folder, f"annotated_{image_file}")
                cv2.imwrite(annotated_path, annotated_img)

            # Process attendance and engagement data
            insights = self.process_results_fn(result, class_roster)
            timeline_entry = {
                "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                "filename": image_file,
                "insights": insights
            }
            self.timeline_data.append(timeline_entry)

        status_text.text("Timeline processing complete!")
        return self.analyze_timeline()

    def analyze_timeline(self):
        """
        Analyze collected timeline data to extract trends like engagement, attendance, emotion.
        Also detects most improved and most declined students.
        """
        if not self.timeline_data:
            return {"error": "No timeline data available"}

        timestamps = []
        avg_engagement_scores = []
        attendance_counts = []
        emotion_counts = {e: [] for e in ["happy", "neutral", "sad", "angry", "fear", "surprise", "disgust"]}
        student_engagement_history = {}

        for entry in self.timeline_data:
            timestamp = entry["timestamp"]
            insights = entry["insights"]

            timestamps.append(timestamp)
            avg_engagement_scores.append(insights["engagement"]["average_engagement_score"])
            attendance_counts.append(len(insights["attendance"]["present_students"]))

            # Count emotions per session
            emotion_count = {e: 0 for e in emotion_counts}
            for student in insights["student_details"]:
                name = student["name"]
                if name not in student_engagement_history:
                    student_engagement_history[name] = []
                student_engagement_history[name].append({
                    "timestamp": timestamp,
                    "engagement_score": student["engagement_score"],
                    "primary_emotion": student["primary_emotion"]["emotion"],
                    "attention_direction": student.get("attention_direction", "forward")
                })
                primary_emotion = student["primary_emotion"]["emotion"]
                if primary_emotion in emotion_count:
                    emotion_count[primary_emotion] += 1

            for e in emotion_counts:
                emotion_counts[e].append(emotion_count[e])

        # Trend analysis
        engagement_trend = self._calculate_trend(avg_engagement_scores)
        attendance_trend = self._calculate_trend(attendance_counts)

        # Student improvement analysis
        student_trends = {}
        for student, records in student_engagement_history.items():
            if len(records) >= 2:
                scores = [r["engagement_score"] for r in records]
                student_trends[student] = self._calculate_trend(scores)

        most_improved = max(student_trends, key=student_trends.get) if student_trends else "N/A"
        most_declined = min(student_trends, key=student_trends.get) if student_trends else "N/A"

        return {
            "timeline_summary": {
                "num_sessions": len(self.timeline_data),
                "time_period": {
                    "start": timestamps[0] if timestamps else "N/A",
                    "end": timestamps[-1] if timestamps else "N/A"
                },
                "trends": {
                    "engagement": {
                        "direction": "increasing" if engagement_trend > 0 else "decreasing",
                        "percent_change": abs(engagement_trend) * 100
                    },
                    "attendance": {
                        "direction": "increasing" if attendance_trend > 0 else "decreasing",
                        "percent_change": abs(attendance_trend) * 100
                    }
                },
                "student_insights": {
                    "most_improved": most_improved,
                    "most_declined": most_declined
                }
            },
            "session_data": self.timeline_data,
            "engagement_data": {
                "timestamps": timestamps,
                "avg_scores": avg_engagement_scores,
                "attendance": attendance_counts,
                "emotions": emotion_counts
            },
            "student_data": student_engagement_history
        }

    def _calculate_trend(self, values):
        """Simple linear trend: (end - start) / start. Handles 0-division."""
        if not values or len(values) < 2:
            return 0
        start, end = values[0], values[-1]
        return (end - start) / start if start != 0 else (1 if end > 0 else 0)

    def generate_visualizations(self):
        """
        Generate and save matplotlib plots of engagement, attendance, and emotion over time.
        Returns:
            Path to saved plot image.
        """
        if not self.timeline_data:
            return None

        timestamps = [entry["timestamp"] for entry in self.timeline_data]
        short_timestamps = [datetime.strptime(ts, "%Y-%m-%d %H:%M:%S").strftime("%m-%d %H:%M") for ts in timestamps]

        data = {
            "Timestamp": short_timestamps,
            "Engagement": [entry["insights"]["engagement"]["average_engagement_score"] for entry in self.timeline_data],
            "Attendance": [len(entry["insights"]["attendance"]["present_students"]) for entry in self.timeline_data]
        }

        # Initialize emotion columns
        emotions = ["happy", "neutral", "sad", "angry", "surprise", "fear", "disgust"]
        for e in emotions:
            data[e.capitalize()] = []

        for entry in self.timeline_data:
            emotion_count = {e: 0 for e in emotions}
            for student in entry["insights"]["student_details"]:
                emo = student["primary_emotion"]["emotion"]
                if emo in emotion_count:
                    emotion_count[emo] += 1
            for e in emotions:
                data[e.capitalize()].append(emotion_count[e])

        df = pd.DataFrame(data)

        fig, axes = plt.subplots(3, 1, figsize=(10, 15))

        # Plot Engagement
        axes[0].plot(df["Timestamp"], df["Engagement"], marker='o', color='blue')
        axes[0].set_title("Class Engagement Over Time")
        axes[0].set_ylabel("Engagement Score")
        axes[0].grid(True)

        # Plot Attendance
        axes[1].plot(df["Timestamp"], df["Attendance"], marker='s', color='green')
        axes[1].set_title("Attendance Over Time")
        axes[1].set_ylabel("Number of Students")
        axes[1].grid(True)

        # Plot Emotions
        for e in emotions:
            axes[2].plot(df["Timestamp"], df[e.capitalize()], marker='.', label=e.capitalize())
        axes[2].set_title("Emotions Over Time")
        axes[2].set_ylabel("Count")
        axes[2].legend()
        axes[2].grid(True)

        plt.tight_layout()
        plot_path = os.path.join(self.results_folder, "timeline_analysis.png")
        plt.savefig(plot_path)
        plt.close()
        return plot_path

    def export_timeline_data(self):
        """
        Export timeline analysis to a JSON file.
        Returns:
            Path to exported JSON file.
        """
        if not self.timeline_data:
            return None
        export_path = os.path.join(self.results_folder, "timeline_data.json")
        with open(export_path, 'w') as f:
            json.dump(self.timeline_data, f, indent=2)
        return export_path
