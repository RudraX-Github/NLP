🎙️ Voice NLP Suite Pro (v6.0)

Voice NLP Suite Pro is an expert-level audio processing framework that bridges the gap between Acoustic Signal Processing, Natural Language Understanding (NLU), and Generative Voice Cloning.

Available as both a Rich CLI application and a modern Streamlit Web Dashboard, this suite offers a comprehensive workflow for analyzing, cleaning, transcribing, and cloning human speech.

🚀 Key Features

1. 🎧 Advanced Audio Acquisition & Preprocessing

Multi-Modal Input: Record live audio or upload files (WAV, MP3).

Video Extraction: Automatically extracts audio tracks from video files (.mp4, .mov, .mkv) using moviepy.

Intelligent Noise Reduction: User-selectable spectral gating (Soft, Moderate, Robust) powered by noisereduce and librosa.

2. 📊 Comprehensive Voice Analysis

Leveraging Parselmouth (Praat) and Librosa, the suite performs deep-dive analytics:

Acoustic: Pitch (F0), Formants (F1/F2), Intensity (RMS), and Spectral Analysis.

Biometric: Zero-shot gender estimation and speaker identity profiling.

Emotional: Prosody analysis including Jitter (frequency variation) and Shimmer (amplitude variation) to detect vocal stability.

Linguistic: Sentiment polarity, subjectivity, and Named Entity Recognition (NER) via spaCy and TextBlob.

3. 🧬 Generative Voice Cloning & Translation

Zero-Shot Cloning: Synthesize new speech using a reference audio sample (5-10s) without training a new model.

Multilingual Support: Clone voices in 16+ languages (En, Fr, De, Es, It, Pt, Pl, Zh-cn, etc.).

Smart Chunking: Proprietary logic to handle long-form text synthesis by intelligently splitting text based on punctuation and semantic breaks.

Auto-Translation: Automatically translates input text to the target language before synthesis.

4. 📈 Visualization & Reporting

Interactive Dashboard: Streamlit-based UI with Plotly waveforms, spectrograms, and sentiment gauges.

PDF Reports: Auto-generates professional PDF summaries of the analysis session.

Waveform Comparison: Visual overlay of Original vs. Cloned audio signals.

🛠️ Installation

Prerequisites

Python 3.9 or higher

FFmpeg (required for audio processing and pydub/moviepy)

Setup

Clone the repository:

git clone [https://github.com/yourusername/voice-nlp-suite.git](https://github.com/yourusername/voice-nlp-suite.git)
cd voice-nlp-suite


Install dependencies:
It is recommended to use a virtual environment.

pip install -r requirements.txt


Note: If you encounter issues installing parselmouth, ensure you have the correct system build tools.

Download Language Models:
The application will automatically download the necessary models (en_core_web_sm for spaCy, XTTS v2 for Coqui) on the first run.

💻 Usage

Option A: The Streamlit Web App (Recommended)

The v7.0 interface offers a modern, glassmorphism-styled dashboard.

streamlit run streamapp.py


Access the dashboard at http://localhost:8501

Features: Drag-and-drop upload, interactive charts, and PDF export.

Option B: The Pro CLI

For power users who prefer a terminal interface with Rich text formatting.

python app.py


Follow the interactive prompts to record, analyze, or clone voices.

🔍 Analysis Modules Explained

The suite performs analysis across five distinct dimensions:

Module

Metrics & Description

Technology

Acoustic

Pitch, Tone, Formants: Identifies fundamental frequency and vowel characteristics.



SNR: Signal-to-Noise ratio assessment.

Librosa, Parselmouth

Linguistic

Transcription: Whisper-based Speech-to-Text.



Sentiment: Polarity (-1 to +1) and Subjectivity.



NER: Extraction of names, dates, and locations.

OpenAI Whisper, spaCy, TextBlob

Emotional

Jitter & Shimmer: Micro-fluctuations in pitch and loudness often correlated with stress or emotion.



Prosody: Rhythm and stress patterns.

Parselmouth (Praat)

Biometric

Gender Estimation: Pitch-threshold based estimation.



Identity: Speaker verification profiling.

Numpy, Scipy

Technical

Fluency: Speaking rate (WPM) calculation.



Language ID: Automatic language detection.

Python Native

🗺️ Project Roadmap

This project has evolved from a simple script to a full-stack application.

v2.0: Introduction of Core Voice Cloning (Coqui TTS).

v3.x: Implementation of Spectral Gating (Noise Reduction) and Rich CLI.

v4.0: Dynamic Prosody Control and Emotion Analysis.

v5.0: Multilingual support and Biometric analysis.

v6.0 (Current Core): Advanced Workflow, Video file support, Smart Text Chunking.

v7.0 (Current Web): Streamlit Web Application with PDF reporting.

📂 Project Structure

├── app.py                # Main CLI Application (Rich UI)
├── streamapp.py          # Streamlit Web Application
├── Voice_Analysis.txt    # Documentation on analysis metrics
├── Voice NLP Suit.html   # Interactive Timeline Visualization
├── requirements.txt      # Python dependencies
└── README.md             # Documentation


⚠️ Ethical Disclaimer

Voice NLP Suite Pro utilizes powerful voice cloning technology. This tool is intended for educational, research, and accessibility purposes only.

Do not clone the voice of any individual without their explicit consent.

Do not use this tool to generate deepfakes for misleading, fraudulent, or malicious purposes.

The developers assume no liability for the misuse of this software.

🤝 Contributing

Contributions are welcome! Please fork the repository and submit a pull request for any enhancements or bug fixes.

Fork the Project

Create your Feature Branch (git checkout -b feature/AmazingFeature)

Commit your Changes (git commit -m 'Add some AmazingFeature')