# AI-cliping
Smart AI video clipper that automatically finds and cuts highlight moments from your videos. Built with Streamlit + Whisper + MoviePy.
import streamlit as st
import moviepy.editor as mp
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
import tempfile, os

st.set_page_config(page_title="Aman ClipAI", page_icon="🎬")
st.title("🎬 Aman ClipAI - Smart AI Video Clipper")

st.write("Upload your video and let AI detect highlights, generate captions, and translate them automatically!")

# Upload video
video_file = st.file_uploader("🎥 Upload your video file", type=["mp4", "mov", "mkv"])

# Language selection
lang = st.selectbox("🌍 Select Caption Language", ["English", "Urdu", "Hindi", "Arabic", "Spanish"])

if video_file:
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(video_file.read())
        video_path = tmp.name

    st.video(video_path)

    # Step 1: Transcribe
    st.info("🧠 Extracting speech from video using Whisper...")
    asr = pipeline("automatic-speech-recognition", model="openai/whisper-small")
    result = asr(video_path)
    transcript = result["text"]
    st.write("📝 Transcript Preview:", transcript[:300] + "...")

    # Step 2: Translate Captions
    if lang != "English":
        st.info(f"🌐 Translating captions into {lang}...")
        translator = pipeline("translation", model="Helsinki-NLP/opus-mt-en-mul")
        translated = translator(transcript)[0]["translation_text"]
    else:
        translated = transcript

    # Step 3: Generate Clip (simple keyword highlight)
    st.info("✂️ Detecting highlights...")
    keywords = ["important", "great", "best", "amazing", "wow", "note"]
    clips = []
    video = mp.VideoFileClip(video_path)

    for word in keywords:
        if word in transcript.lower():
            start = max(0, transcript.lower().index(word) % int(video.duration - 10))
            end = min(video.duration, start + 10)
            clips.append(video.subclip(start, end))

    if clips:
        final = mp.concatenate_videoclips(clips)
        # Step 4: Add Captions
        st.info("🪄 Adding captions to video...")
        txt = mp.TextClip(translated, fontsize=30, color='white', bg_color='black', size=video.size, method='caption')
        txt = txt.set_duration(final.duration).set_position(("center", "bottom"))
        final = mp.CompositeVideoClip([final, txt])

        output_path = "highlight_with_captions.mp4"
        final.write_videofile(output_path, codec="libx264", audio_codec="aac")

        st.success("✅ Done! Your AI Highlight Clip is ready.")
        st.video(output_path)
        with open(output_path, "rb") as f:
            st.download_button("⬇️ Download Highlight Clip", f, file_name="highlight_with_captions.mp4")
    else:
        st.warning("No highlights found. Try another video!")
