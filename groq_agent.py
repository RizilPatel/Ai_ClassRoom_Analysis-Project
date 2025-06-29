import json
import requests
from typing import Dict, Any, Optional

class GroqAgent:
    """
    AI Teaching Assistant using Groq API for analyzing classroom engagement and attendance.
    """

    def __init__(self, api_key: str, model: str):
        # Initialize the agent with API key and model
        self.api_key = "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"  # Your API Key
        self.model = model
        self.base_url = "https://api.groq.com/openai/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def create_prompt(self, insights: Dict[str, Any], question: str = "") -> str:
        """
        Build the prompt for single session classroom engagement analysis.
        """
        base_prompt = (
            "Classroom Analysis and Engagement Assessment\n"
            "System Context\n"
            "You are an AI teaching assistant that specializes in analyzing classroom attendance "
            "and student engagement based on facial recognition and emotional analysis data. "
            "You have access to detailed information about which students were present in class, "
            "their emotional states, and engagement metrics.\n\n"
            "Available Data\n"
            "I will provide you with a JSON object containing the following information:\n"
            "- Date of the class session\n"
            "- Total number of faces detected in the classroom\n"
            "- List of identified students who were present\n"
            "- Number of unknown/unidentified people in the classroom\n"
            "- List of students who were absent (compared to the class roster)\n"
            "- Most common emotion detected across all students\n"
            "- Average engagement score for the class (0-1 scale)\n"
            "- Most engaged and least engaged students\n"
            "- Detailed emotion data for each detected face, including:\n"
            "   * Primary emotion and its confidence score\n"
            "   * Secondary emotion and its confidence score\n\n"
            "Response Guidelines\n"
            "- Respond concisely and directly to the specific question asked\n"
            "- Support your insights with specific data from the analysis\n"
            "- When discussing individual students, maintain a supportive and constructive tone\n"
            "- Avoid making absolute judgments about student performance based solely on emotional data\n"
            "- Suggest practical next steps for the teacher when appropriate\n"
            "- When analyzing engagement, consider both the dominant emotion and its intensity\n"
            "- Remember that \"happy\" and \"neutral\" emotions typically indicate good engagement, "
            "while \"sad,\" \"angry,\" or \"fearful\" emotions might indicate disengagement or confusion\n\n"
            "Here is the classroom data to analyze:\n"
        )
        base_prompt += json.dumps(insights, indent=2)
        if question:
            base_prompt += f"\n\nQuestion: {question}\nPlease analyze the data and answer the question."
        else:
            base_prompt += "\n\nPlease provide a brief summary of the class attendance and engagement."
        return base_prompt

    def get_analysis(self, insights: Dict[str, Any], question: str = "") -> str:
        """
        Send classroom insights to Groq API and retrieve analysis response.
        """
        prompt = self.create_prompt(insights, question)
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a helpful teaching assistant that analyzes classroom data."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7,
            "max_tokens": 1000
        }
        try:
            resp = requests.post(self.base_url, headers=self.headers, json=payload)
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except Exception as e:
            return f"Error getting AI analysis: {e}"

    # ---------------------------
    # Timeline Analysis Methods
    # ---------------------------

    def create_timeline_prompt(self, timeline_analysis: Dict[str, Any], question: str = "") -> str:
        """
        Build the prompt for analyzing multi-session classroom trends over time.
        """
        base = (
            "Classroom Timeline Analysis and Engagement Assessment\n"
            "System Context\n"
            "You are an AI teaching assistant that specializes in analyzing classroom attendance and "
            "student engagement trends over time based on facial recognition and emotional analysis data.\n\n"
            "Available Data\n"
            "I will provide you with a JSON object containing timeline analysis of multiple sessions including:\n"
            "- Time period covered\n"
            "- Number of sessions analyzed\n"
            "- Engagement trends and percent change\n"
            "- Attendance trends and percent change\n"
            "- Most improved and most declined students\n"
            "- Session-by-session emotional and engagement metrics\n\n"
            "Here is the timeline data:\n"
        )
        base += json.dumps(timeline_analysis, indent=2)
        if question:
            base += f"\n\nQuestion: {question}\nPlease analyze the timeline data and answer the question."
        else:
            base += "\n\nPlease provide an analysis of engagement and attendance trends over time."
        return base

    def get_timeline_analysis(self, timeline_analysis: Dict[str, Any], question: str = "") -> str:
        """
        Send timeline-based classroom data to Groq API and retrieve trend analysis.
        """
        prompt = self.create_timeline_prompt(timeline_analysis, question)
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a helpful teaching assistant that analyzes classroom trends over time."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7,
            "max_tokens": 1500
        }
        try:
            resp = requests.post(self.base_url, headers=self.headers, json=payload)
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except Exception as e:
            return f"Error getting timeline analysis: {e}"
