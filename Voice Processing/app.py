# -*- coding: utf-8 -*-
"""
Voice NLP Suite (PRO Edition v6.0 - Advanced Workflow)
Author: AI/ML Engineer
Date: 2025-10-10
Description: An expert-level suite for voice processing. This version introduces a new
             workflow with file selection (including video), automatic translation for
             synthesis, smart text chunking, and selective saving of artifacts.
"""
# --- Suppress all warnings for a clean, user-friendly output ---
import warnings
warnings.filterwarnings("ignore")

# --- Core Libraries ---
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import speech_recognition as sr
import librosa
import spacy
from textblob import TextBlob
import time
import os
import sys
import shutil
import soundfile as sf
import re

# --- Initialize UI Early to Catch Import Errors ---
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.prompt import Prompt, Confirm
    from rich.table import Table
    from rich.live import Live
    RICH_UI_ENABLED = True
    console = Console()
except ImportError:
    RICH_UI_ENABLED = False
    class FallbackConsole:
        def print(self, text): print(text)
    console = FallbackConsole()

# --- Optional/New Libraries ---
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    VISUALIZATION_ENABLED = True
except ImportError:
    VISUALIZATION_ENABLED = False

try:
    import torch
    from TTS.api import TTS
    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import XttsAudioConfig, XttsArgs
    from TTS.tts.configs.shared_configs import BaseDatasetConfig, BaseTTSConfig
    import parselmouth
    from parselmouth.praat import call as praat_call
    import noisereduce as nr
    from moviepy.editor import VideoFileClip
    import translators as ts
except ImportError as e:
    console.print(f"❌ [bold red]A required library is missing. Please ensure 'moviepy' and 'translators' are installed ('pip install moviepy translators'). Error: {e}[/bold red]")
    exit()

# --- Configuration ---
SAMPLE_RATE = 22050
INPUT_FILENAME = "user_voice_input.wav"
CLEANED_FILENAME = "cleaned_user_voice.wav"
CLONED_FILENAME = "cloned_voice_output.wav"
FIGURE_FILENAME = "voice_comparison.png"

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    console.print("[bold bright_yellow]spaCy model 'en_core_web_sm' not found. Downloading...[/bold bright_yellow]")
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

os.environ['COQUI_TOS_AGREED'] = '1'

# --- 1. Voice Capture & Preprocessing ---
def record_audio(duration=10):
    # ... (code is unchanged)
    console.print(f"\n🎙️ [bold bright_cyan]Starting recording for {duration} seconds...[/bold bright_cyan]")
    recording = sd.rec(int(duration * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='int16')
    if RICH_UI_ENABLED:
        with Live(console=console, screen=False, auto_refresh=False) as live:
            for i in range(duration, 0, -1):
                live.update(f"   [bold yellow]Recording... {i}s remaining[/bold yellow]", refresh=True)
                time.sleep(1)
    else:
        for i in range(duration, 0, -1):
            print(f"   Recording... {i}s remaining", end='\r')
            time.sleep(1)
        print()
    sd.wait()
    wav.write(INPUT_FILENAME, SAMPLE_RATE, recording)
    console.print(f"✅ [bold green]Audio successfully saved as {INPUT_FILENAME}[/bold green]")
    return INPUT_FILENAME

def select_and_process_file():
    filepath = Prompt.ask("[bold]Please enter the path to your audio/video file[/bold]")
    if not os.path.exists(filepath):
        console.print(f"❌ [bold red]File not found at '{filepath}'[/bold red]")
        return None
    
    try:
        console.print(f"[italic]Processing file: {os.path.basename(filepath)}...[/italic]")
        _, ext = os.path.splitext(filepath)
        if ext.lower() in ['.mp4', '.mov', '.avi', '.mkv']:
            console.print("[italic]Video file detected. Extracting audio...[/italic]")
            video = VideoFileClip(filepath)
            video.audio.write_audiofile(INPUT_FILENAME, samplerate=SAMPLE_RATE, logger=None)
        else:
            shutil.copy(filepath, INPUT_FILENAME)
        
        console.print(f"✅ [bold green]Audio successfully prepared as {INPUT_FILENAME}[/bold green]")
        return INPUT_FILENAME
    except Exception as e:
        console.print(f"❌ [bold red]Failed to process file. Error: {e}[/bold red]")
        return None

def run_preprocessing_menu(input_file, output_file):
    # ... (code is unchanged)
    console.print(Panel("[bold]🧹 Optional Audio Preprocessing[/bold]", style="bright_cyan", title="Noise Reduction"))
    if not Confirm.ask("[bold]Do you want to apply noise reduction?[/bold]"):
        console.print("[bold yellow]Skipping noise reduction.[/bold yellow]")
        return input_file
    prompt_text = ("[bold]Choose a method:[/bold]\n[1] Soft\n[2] Moderate\n[3] Robust")
    console.print(prompt_text)
    choice = Prompt.ask("[bold]Enter your choice[/bold]", choices=["1", "2", "3"], default="2")
    try:
        y, sr = librosa.load(input_file, sr=SAMPLE_RATE)
        if choice == '1':
            reduced_noise = nr.reduce_noise(y=y, sr=sr, prop_decrease=0.8)
        elif choice == '2':
            non_silent = librosa.effects.split(y, top_db=30)
            noise_clip = y[0:non_silent[0][0]] if len(non_silent) > 0 and non_silent[0][0] > 0 else y[0:int(sr*0.5)]
            reduced_noise = nr.reduce_noise(y=y, y_noise=noise_clip, sr=sr, prop_decrease=0.95)
        elif choice == '3':
            reduced_noise, _ = librosa.effects.hpss(y)
        sf.write(output_file, reduced_noise, sr)
        console.print(f"✅ [bold green]Noise reduction complete.[/bold green]")
        return output_file
    except Exception as e:
        console.print(f"❌ [bold red]Could not perform noise reduction: {e}.[/bold red]")
        return input_file

# --- 2. Speech Recognition Engine ---
def transcribe_audio(audio_file, model_size="base", language="en"):
    # ... (code is unchanged)
    console.print(f"\n🧠 [bold bright_cyan]Transcribing with Whisper ({model_size} model) for {language.upper()}[/bold bright_cyan]")
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(audio_file) as source:
            audio_data = recognizer.record(source)
        text = recognizer.recognize_whisper(audio_data, model=model_size, language=language)
        console.print(f"📝 [bold green]Transcription:[/bold green] \"{text}\"")
        return text
    except Exception as e:
        console.print(f"❌ [bold red]An error occurred during transcription: {e}[/bold red]")
        return None

# --- 3. Core Analysis Engine ---
class VoiceAnalyzer:
    # ... (code is unchanged, only analyze_linguistic is slightly adapted)
    def __init__(self, audio_file, transcribed_text, language="en"):
        if not os.path.exists(audio_file): raise FileNotFoundError(f"File not found: {audio_file}")
        self.audio_file = audio_file
        self.text = transcribed_text or ""
        self.language = language
        self.y, self.sr = librosa.load(self.audio_file, sr=SAMPLE_RATE)
        self.duration_seconds = librosa.get_duration(y=self.y, sr=self.sr)
        self.sound = parselmouth.Sound(self.audio_file)

    def analyze_acoustic(self):
        try:
            pitch = praat_call(self.sound, "To Pitch", 0.0, 75, 600)
            mean_pitch = praat_call(pitch, "Get mean", 0, 0, "Hertz")
            formant = praat_call(self.sound, "To Formant (burg)", 0.0, 5, 5500, 0.025, 50)
            f1_mean = praat_call(formant, "Get mean", 1, 0, 0, "Hertz")
            f2_mean = praat_call(formant, "Get mean", 2, 0, 0, "Hertz")
            rms = librosa.feature.rms(y=self.y)[0]
            return {"Pitch & Tone": {"Mean Pitch (Hz)": f"{mean_pitch:.2f}"}, "Formant Analysis": {"Mean F1 (Hz)": f"{f1_mean:.2f}", "Mean F2 (Hz)": f"{f2_mean:.2f}"}, "Energy & Loudness": {"Avg Intensity (RMS)": f"{np.mean(rms):.4f}"}}
        except Exception as e: return {"Error": f"Analysis failed: {e}"}

    def analyze_linguistic(self):
        if not self.text: return {"Error": "No text to analyze."}
        base_analysis = {"Transcription": {"Full Text": self.text}, "Language Model": {"Acoustic Model": "Whisper"}}
        if self.language == "en":
            doc = nlp(self.text)
            blob = TextBlob(self.text)
            base_analysis["Semantic Content"] = {"Keywords": [chunk.text for chunk in doc.noun_chunks] or "N/A"}
            base_analysis["Sentiment & Intent"] = {"Polarity": f"{blob.sentiment.polarity:.2f}", "Subjectivity": f"{blob.sentiment.subjectivity:.2f}"}
        else:
            base_analysis["Note"] = {"Details": "Semantic/Sentiment analysis is optimized for English."}
        return base_analysis

    def analyze_emotional(self):
        # ... (code is unchanged)
        try:
            jitter = "N/A"; shimmer = "N/A"; std_dev_pitch_str = "N/A"
            try:
                pitch = praat_call(self.sound, "To Pitch", 0.0, 75, 600)
                std_dev_pitch = praat_call(pitch, "Get standard deviation", 0, 0, "Hertz")
                std_dev_pitch_str = f"{std_dev_pitch:.2f}"
            except Exception: pass
            try:
                point_process = praat_call(self.sound, "To PointProcess (periodic, cc)", 75, 600)
                if praat_call(point_process, "Get number of points") > 1:
                    jitter = f"{praat_call(point_process, 'Get jitter (local)', 0, 0, 0.0001, 0.02, 1.3) * 100:.4f}%"
                    shimmer = f"{praat_call(point_process, 'Get shimmer (local)', 0, 0, 0.0001, 0.02, 1.3, 1.6) * 100:.4f}%"
            except Exception: pass
            return {"Prosody": {"Pitch Variation (Hz)": std_dev_pitch_str}, "Vocal Stability": {"Jitter": jitter, "Shimmer": shimmer}}
        except Exception as e: return {"Error": f"Analysis failed: {e}"}

    def analyze_biometric(self):
        # ... (code is unchanged)
        try:
            pitch = praat_call(self.sound, "To Pitch", 0.0, 75, 600)
            mean_pitch = praat_call(pitch, "Get mean", 0, 0, "Hertz")
            gender = "Undetermined"
            if 85 < mean_pitch < 180: gender = "Likely Male"
            elif 165 < mean_pitch < 255: gender = "Likely Female"
            return {"Identity Estimation": {"Estimated Gender": f"{gender}"}, "Speaker Adaptation": {"Method": "Zero-Shot Cloning"}}
        except Exception as e: return {"Error": f"Analysis failed: {e}"}

    def analyze_technical(self):
        # ... (code is unchanged)
        if not self.text: return {"Error": "No text to analyze."}
        try: language = self.language.upper()
        except: language = "N/A"
        speaking_rate = (len(self.text.split()) / self.duration_seconds) * 60 if self.duration_seconds > 0 else 0
        return {"Speech Fluency": {"Speaking Rate (WPM)": f"{speaking_rate:.2f}"}, "Content": {"Detected Language": language}}

# --- 4. Quality & Visualization Modules ---
def visualize_and_save_audio(original_file, cloned_file):
    # ... (code is largely unchanged, just uses FIGURE_FILENAME)
    if not VISUALIZATION_ENABLED: return False
    console.print("\n📊 [bold bright_cyan]Generating voice waveform comparison...[/bold bright_cyan]")
    try:
        y_orig, _ = librosa.load(original_file, sr=SAMPLE_RATE)
        y_cloned, _ = librosa.load(cloned_file, sr=SAMPLE_RATE)
        sns.set_style("whitegrid")
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, sharey=True)
        fig.suptitle('Voice Waveform Comparison', fontsize=16, weight='bold')
        librosa.display.waveshow(y_orig, sr=SAMPLE_RATE, ax=axes[0], color='royalblue'); axes[0].set_title("Original Voice")
        librosa.display.waveshow(y_cloned, sr=SAMPLE_RATE, ax=axes[1], color='darkorange'); axes[1].set_title("Synthesized Voice")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95]);
        fig.savefig(FIGURE_FILENAME)
        plt.show(block=True)
        return True
    except Exception as e:
        console.print(f"❌ [bold red]Could not generate visualization. Error: {e}[/bold red]")
        return False

# --- 5. Text-to-Speech Module ---
def speak_text_with_cloning(text_to_speak, reference_audio_file, tts_model):
    console.print(Panel("[bold]🗣️ Advanced Text-to-Speech Engine[/bold]", style="bright_cyan", title="Cloning Voice"))
    
    languages = {"1": ("Hindi", "hi"), "2": ("English", "en"), "3": ("French", "fr"), "4": ("Russian", "ru"), "5": ("Chinese", "zh-cn")}
    
    while True:
        console.print("\n[bold]Choose a language for synthesis:[/bold]")
        for key, (name, _) in languages.items(): console.print(f"[{key}] {name}")
        lang_choice = Prompt.ask("[bold]Enter your choice[/bold]", choices=languages.keys(), default="2")
        chosen_lang_name, chosen_lang_code = languages[lang_choice]

        try:
            console.print(f"\n[italic]Translating text to {chosen_lang_name}...[/italic]")
            translated_text = ts.translate_text(text_to_speak, to_language=chosen_lang_code)
            console.print(f"✅ [bold green]Translated Text:[/bold green] \"{translated_text}\"")
            
            console.print(f"\n[italic]Synthesizing '{chosen_lang_name}' speech...[/italic]")
            
            # --- NEW ROBUST CHUNKING LOGIC ---
            def create_chunks(text, limit=225):
                chunks = []
                # First, split by major sentence-ending punctuation
                sentences = re.split('([.!?])', text)
                # Combine sentences with their punctuation
                if len(sentences) > 1:
                    processed_sentences = ["".join(i) for i in zip(sentences[0::2], sentences[1::2])]
                    if len(sentences) % 2 == 1:
                        processed_sentences.append(sentences[-1])
                else:
                    processed_sentences = sentences

                for sentence in processed_sentences:
                    if len(sentence) <= limit:
                        if sentence.strip(): chunks.append(sentence.strip())
                    else:
                        # Sentence is too long, split it further by words
                        current_chunk = ""
                        words = sentence.split(' ')
                        for word in words:
                            if len(current_chunk) + len(word) + 1 <= limit:
                                current_chunk += word + " "
                            else:
                                if current_chunk.strip(): chunks.append(current_chunk.strip())
                                current_chunk = word + " "
                        if current_chunk.strip():
                            chunks.append(current_chunk.strip())
                return chunks

            sentences = create_chunks(translated_text)
            final_audio = []
            temp_files = []
            for i, sentence in enumerate(sentences):
                if not sentence: continue
                temp_filename = f"temp_chunk_{i}.wav"
                temp_files.append(temp_filename)
                tts_model.tts_to_file(text=sentence, speaker_wav=reference_audio_file, language=chosen_lang_code, file_path=temp_filename)
                y, sr = librosa.load(temp_filename, sr=SAMPLE_RATE)
                final_audio.append(y)
            
            if not final_audio:
                console.print("❌ [bold red]Text was empty after processing. Cannot synthesize.[/bold red]")
                continue

            concatenated_audio = np.concatenate(final_audio)
            sf.write(CLONED_FILENAME, concatenated_audio, SAMPLE_RATE)

            # Cleanup temp chunks
            for f in temp_files: os.remove(f)

            if os.path.exists(CLONED_FILENAME):
                console.print("🔊 [bold]Playing cloned voice...[/bold]")
                sd.play(concatenated_audio, SAMPLE_RATE); sd.wait()
                console.print("✅ [bold green]Playback finished.[/bold green]")
                
                was_viz_generated = visualize_and_save_audio(reference_audio_file, CLONED_FILENAME)
                save_results_menu(did_clean=os.path.exists(CLEANED_FILENAME), did_clone=True, did_viz=was_viz_generated)

        except Exception as e: 
            console.print(f"❌ [bold red]An error occurred during voice cloning: {e}[/bold red]")
        
        if not Confirm.ask("\n[bold]Synthesize the same text in another language?[/bold]"):
            break


# --- 6. Main Application Logic ---
def initialize_tts_model():
    console.print(Panel("[bold]🚀 Voice Cloning Model Setup[/bold]", style="bright_cyan"))
    model_name = "tts_models/multilingual/multi-dataset/xtts_v2"
    try:
        console.print(f"\n⏳ [italic]Loading model '{model_name}' (this happens once)...[/italic]")
        if hasattr(torch, 'serialization') and hasattr(torch.serialization, 'safe_globals'):
            with torch.serialization.safe_globals([XttsConfig, XttsAudioConfig, BaseDatasetConfig, XttsArgs, BaseTTSConfig]):
                tts_model = TTS(model_name, progress_bar=True, gpu=False)
        else:
            tts_model = TTS(model_name, progress_bar=True, gpu=False)
        console.print("✅ [bold green]Voice cloning model ready.[/bold green]"); return tts_model
    except Exception as e: console.print(f"❌ [bold red]Could not load model. Error: {e}[/bold red]"); return None

def print_detailed_report(report):
    # ... (code is unchanged)
    METRIC_EXPLANATIONS = {"Mean Fundamental Frequency (Hz)": "The average perceived pitch.", "Average Intensity (RMS)": "Corresponds to loudness.", "Polarity": "Sentiment from -1.0 (negative) to 1.0 (positive).", "Pitch Variation (Std Dev Hz)": "Higher values mean more expressive speech.", "Jitter": "Frequency variation, can relate to roughness.", "Shimmer": "Amplitude variation, can relate to breathiness.", "Speaking Rate (WPM)": "A measure of speech pace."}
    console.print(Panel("[bold]Full Analysis Report[/bold]", style="green", border_style="green"))
    for category, analyses in report.items():
        if not analyses: continue
        table = Table(title=f"[bold bright_cyan]{category}[/bold bright_cyan]", show_header=False, box=None, padding=(0, 2))
        table.add_column("Metric", width=35); table.add_column("Value", width=30); table.add_column("Explanation", style="dim")
        if "Error" in analyses: table.add_row("[bold red]Error[/bold red]", analyses["Error"])
        else:
            for sub_category, metrics in analyses.items():
                table.add_row(f"[bold underline]{sub_category}[/bold underline]", "")
                for metric, value in metrics.items():
                    explanation = METRIC_EXPLANATIONS.get(metric, "")
                    table.add_row(f"  {metric}", str(value), explanation)
        console.print(table)


def save_results_menu(did_clean, did_clone, did_viz):
    console.print(Panel("[bold]💾 Save Artifacts[/bold]", style="bright_cyan"))
    choices = {}
    if did_clean: choices["1"] = ("Cleaned Audio", CLEANED_FILENAME)
    if did_clone: choices[str(len(choices)+1)] = ("Cloned Voice", CLONED_FILENAME)
    if did_viz: choices[str(len(choices)+1)] = ("Comparison Figure", FIGURE_FILENAME)

    if not choices: return

    console.print("[bold]Select which files you want to save (e.g., '1' or '1,3'):[/bold]")
    for key, (name, _) in choices.items():
        console.print(f"[{key}] {name}")
    
    selections = Prompt.ask("[bold]Enter your choice(s)[/bold]").replace(" ", "").split(',')
    for sel in selections:
        if sel in choices:
            name, temp_path = choices[sel]
            save_path = Prompt.ask(f"[bold]Enter filename to save '{name}'[/bold]", default=f"saved_{os.path.basename(temp_path)}")
            shutil.copy(temp_path, save_path)
            console.print(f"✅ [bold green]{name} saved to {os.path.abspath(save_path)}[/bold green]")

def cleanup():
    # ... (code is unchanged)
    console.print("[bold green]Exiting the NLP Suite. God Speed![/bold green]")
    try:
        sd.stop()
        for f in [INPUT_FILENAME, CLEANED_FILENAME, CLONED_FILENAME, FIGURE_FILENAME]:
            if os.path.exists(f): os.remove(f)
        if Confirm.ask("[bold]Clear downloaded model cache?[/bold]"):
            tts_cache_path = os.path.join(os.path.expanduser("~"), ".local", "share", "tts")
            if os.path.exists(tts_cache_path): shutil.rmtree(tts_cache_path)
    except Exception: pass
    finally: sys.exit(0)

def main_menu():
    console.print(Panel("[bold]Welcome to the Voice NLP Suite Pro v6.0[/bold]\n[bright_cyan]Advanced Workflow Edition[/bright_cyan]", style="green", title="Voice NLP Suite"))
    tts_model = initialize_tts_model()
    is_audio_ready = False; active_audio_file = None
    
    while True:
        console.print(Panel("[bold]Select an option[/bold]", style="bright_cyan", title="Main Menu"))
        
        analyze_option = "[2] Analyze Voice" if is_audio_ready else "[dim][2] Analyze Voice (Input first)[/dim]"
        tts_option = "[3] Text-to-Speech" if is_audio_ready else "[dim][3] Text-to-Speech (Input first)[/dim]"
        
        prompt_text = f"[bold]Choose an action:[/bold]\n[1] Input Voice\n{analyze_option}\n{tts_option}\n[4] Exit"
        console.print(prompt_text)
        choice = Prompt.ask("[bold]Enter your choice[/bold]", choices=["1", "2", "3", "4"], default="1")
        
        if choice == '1':
            console.print("[bold]How would you like to provide audio?[/bold]")
            input_choice = Prompt.ask("[1] Record Voice\n[2] Select a File", choices=["1", "2"], default="1")
            if input_choice == '1':
                duration = int(Prompt.ask("[bold]Enter recording duration (5-15s recommended)[/bold]", default="10"))
                record_audio(duration)
                active_audio_file = INPUT_FILENAME
            else:
                active_audio_file = select_and_process_file()
            
            if active_audio_file:
                is_audio_ready = True
                if Confirm.ask("[bold]Do you want to apply noise reduction?[/bold]"):
                    active_audio_file = run_preprocessing_menu(active_audio_file, CLEANED_FILENAME)
        
        elif choice == '2':
            if not is_audio_ready: console.print("🔴 [bold red]Please provide a voice input first.[/bold red]"); continue
            
            console.print("\n[bold]Choose the language of the audio:[/bold]")
            lang_map = {"1": ("English", "en"), "2": ("Hindi", "hi"), "3": ("French", "fr"), "4": ("Russian", "ru"), "5": ("Chinese", "zh")}
            for k, (name, _) in lang_map.items(): console.print(f"[{k}] {name}")
            lang_choice = Prompt.ask("[bold]Enter choice[/bold]", choices=lang_map.keys(), default="1")
            _, transcription_lang_code = lang_map[lang_choice]
            
            transcribed_text = transcribe_audio(active_audio_file, language=transcription_lang_code)
            if transcribed_text:
                analyzer = VoiceAnalyzer(active_audio_file, transcribed_text, language=transcription_lang_code)
                full_report = {"🎙️ Acoustic Analysis": analyzer.analyze_acoustic(), "🧠 Linguistic Analysis": analyzer.analyze_linguistic(), "😃 Emotional Analysis": analyzer.analyze_emotional(), "🧬 Biometric Analysis": analyzer.analyze_biometric(), "🛠️ Technical Analysis": analyzer.analyze_technical()}
                print_detailed_report(full_report)
                save_results_menu(did_clean=(active_audio_file == CLEANED_FILENAME), did_clone=False, did_viz=False)

        elif choice == '3':
            if not is_audio_ready: console.print("🔴 [bold red]Please provide a voice input first.[/bold red]"); continue
            if not tts_model: console.print("🔴 [bold red]TTS is disabled.[/bold red]"); continue
            text_to_speak = Prompt.ask("[bold]Enter the text to synthesize[/bold]")
            if text_to_speak: speak_text_with_cloning(text_to_speak, active_audio_file, tts_model)
        
        elif choice == '4':
            cleanup()

if __name__ == "__main__":
    main_menu()

