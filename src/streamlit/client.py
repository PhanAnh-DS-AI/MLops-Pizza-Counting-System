import streamlit as st
import requests
import os

BASE_URL = "http://localhost:8000"

st.set_page_config(page_title="Pizza Sales Counting System", layout="wide")
st.title("🍕 Pizza Sales Counting System")

video_file_map = {
    "1461_CH01_20250607193711_203711 - Trim.mp4": "Camera 1",
    "1465_CH02_20250607170555_172408 - Trim.mp4": "Camera 2",
    "1462_CH03_20250607192844_202844 - Trim.mp4": "Camera 3",
    "1462_CH04_20250607210159_211703 - Trim.mp4": "Camera 4",
    "1464_CH02_20250607180000_190000 - Trim.mp4": "Channel 2",
    "1467_CH04_20250607180000_190000 - Trim.mp4": "Channel 4",
}

display_to_file = {v: k for k, v in video_file_map.items()}
video_display_list = list(video_file_map.values())

if "last_video_choice" not in st.session_state:
    st.session_state["last_video_choice"] = None
if "show_stream" not in st.session_state:
    st.session_state["show_stream"] = False
if "processing_started" not in st.session_state:
    st.session_state["processing_started"] = False
if "just_stopped" not in st.session_state:
    st.session_state["just_stopped"] = False

video_choice_display = st.selectbox("Select Camera/Video", video_display_list)
video_choice_file = display_to_file[video_choice_display]
video_id = video_choice_file.split("_")[0] + "_" + video_choice_file.split("_")[1]  

if video_choice_display != st.session_state["last_video_choice"]:
    st.session_state["show_stream"] = False
    st.session_state["processing_started"] = False
    st.session_state["last_video_choice"] = video_choice_display

if video_choice_display:
    st.subheader(f"Live Detection: {video_choice_display}")

    # Start processing and streaming
    if not st.session_state["show_stream"]:
        if st.button("Start Live Detection"):
            resp = requests.post(
                f"{BASE_URL}/process",
                json={"video_path": f"data/raw_videos/cut_video_test/{video_choice_file}"}
            )
            if resp.status_code == 200:
                st.session_state["show_stream"] = True
                st.session_state["processing_started"] = True
                st.success("Processing started! Now streaming...")
            else:
                st.error("Failed to start processing.")
        else:
            st.info("Click 'Start Live Detection' to begin processing and streaming the selected video.")

    # Show stream and Stop button
    if st.session_state["show_stream"]:
        st.markdown(
            f"""
            <img src="{BASE_URL}/stream/{video_choice_file}" width="1200" />
            """,
            unsafe_allow_html=True,
        )
        if st.button("Stop Processing"):
            resp = requests.post(f"{BASE_URL}/stop/{video_id}")
            if resp.status_code == 200:
                st.success("Processing stopped!")
                st.session_state["show_stream"] = False
                st.session_state["processing_started"] = False
                st.session_state["just_stopped"] = True
                st.rerun()
            else:
                st.error("Failed to stop processing.")

if st.session_state.get("just_stopped", False):
    result_url = f"{BASE_URL}/results/{video_id}"
    resp = requests.get(result_url)
    if resp.status_code == 200 and resp.headers.get("content-type", "").startswith("text/csv"):
        st.write("Pizza Sale Events (CSV):")
        st.markdown(f"[Download CSV Result]({result_url})")
    else:
        st.warning("No CSV result found.")

    video_url = f"{BASE_URL}/video/{video_id}"  
    st.write("Processed Video Result:")
    st.markdown(f"[Download Processed Video]({video_url})")

    st.session_state["just_stopped"] = False

    # Feedback form
st.subheader("Submit Feedback")
correct_count = st.number_input("Correct Pizza Count (Optional)", min_value=0, value=0)
feedback_text = st.text_area("Your feedback (required)", placeholder="Examples: Undercounting pizzas, misidentification, ...")
if st.button("Submit Feedback"):
    if feedback_text.strip() == "":
        st.warning("You need to enter a comment before submitting!")
    else:
        feedback_data = {
            "video_id": video_id,
            "correct_count": correct_count,
            "feedback": feedback_text
        }
        resp = requests.post(f"{BASE_URL}/feedback", json=feedback_data)
        if resp.status_code == 200:
            st.success("Feedback submitted!")
        else:
            st.error("Failed to submit feedback.")