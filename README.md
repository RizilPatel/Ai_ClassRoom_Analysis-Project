# Ai_Classroom_Analysis

## Summary 
My project is a Streamlit‑powered classroom analytics app that lets educators quantify and visualize student engagement in real time. In a single snapshot, it will:

    Detect every face in the frame (via RetinaFace)

    Recognize each student by matching against our registered‑students database (using DeepFace/VGG‑Face)

    Analyze their emotions to compute an engagement score (DeepFace’s emotion model)

    Estimate head pose (Hopenet) and overlay orientation axes for attention cues

Beyond individual images, my app also provides:

    A Database Info panel to audit your student‑photo repository and cached embeddings

    A Classroom Analysis dashboard turning per‑student metrics into attendance/engagement insights—with a Groq‑powered chat agent to query the data

    A Timeline Analysis feature where you upload a series of classroom photos to track trends (attendance shifts, engagement changes, most/least improved students) over time, complete with plots and downloadable reports

It’s designed to empower teachers with data‑driven insights, all packaged into a simple, user‑friendly web interface.


## Important note

There is Huge hopenet_robust_alpha.pkl model in the this project which must be downloaded from = "https://drive.google.com/file/d/1m25PrSE7g9D2q2XJVMR6IA7RaCvWSzCR/view" and must be placed in models folder

