import streamlit as st
import cv2
import numpy as np
import pandas as pd
import os
import tempfile
import time
import matplotlib.pyplot as plt
import seaborn as sns
from ultralytics import YOLO
from sort import Sort
import requests

# Page Config
st.set_page_config(page_title="Construction Safety Dashboard", page_icon="🚧", layout="wide")
DEFAULT_IMAGE_URL = "https://cdn-icons-png.flaticon.com/512/9131/9131478.png"
# Paths
webcam_csv_file = 'webcam_ppe_tracking.csv'
video_csv_file = 'video_ppe_tracking.csv'

for file in [webcam_csv_file, video_csv_file]:
    if not os.path.exists(file):
        pd.DataFrame(columns=["Timestamp", "Person ID", "Equipment Worn", "Equipment Not Worn"]).to_csv(file, index=False)

# Session state initialization
if "receiver_email" not in st.session_state:
    st.session_state.receiver_email = "example@example.com"

if "profile_image" not in st.session_state:
    # Check if the image already exists locally, if not, download it
    if not os.path.exists("default_profile.jpg"):
        response = requests.get(DEFAULT_IMAGE_URL)
        with open("default_profile.jpg", "wb") as f:
            f.write(response.content)

    st.session_state.profile_image = "default_profile.jpg"

if "stop_webcam" not in st.session_state:
    st.session_state.stop_webcam = False

def load_model():
    """ Lazy load the YOLO model. Only load when needed. """
    if "yolo_model" not in st.session_state:
        st.session_state.yolo_model = YOLO('ppe.pt')
        st.session_state.tracker = Sort()

def process_frame(frame, csv_file):
    """ YOLO inference and tracking, returns processed frame and detected classes """
    load_model()
    model = st.session_state.yolo_model
    tracker = st.session_state.tracker

    results = model(frame)
    detections = []
    person_equipment = {}

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            conf = box.conf.item()
            label = model.names[int(box.cls.item())]
            if conf >= 0.4 and label == "Person":
                detections.append([x1, y1, x2, y2, conf])

    tracked_objects = tracker.update(np.array(detections))

    for obj in tracked_objects:
        x1, y1, x2, y2, obj_id = map(int, obj)
        person_equipment[obj_id] = {"worn": [], "not_worn": []}

        for result in results:
            for box in result.boxes:
                bx1, by1, bx2, by2 = box.xyxy[0]
                label = model.names[int(box.cls.item())]
                conf = box.conf.item()

                if conf >= 0.4 and label != "Person":
                    if x1 <= bx1 <= x2 and y1 <= by1 <= y2:
                        if label.startswith("NO-"):
                            person_equipment[obj_id]["not_worn"].append(label)
                        else:
                            person_equipment[obj_id]["worn"].append(label)

                    color = (0, 255, 0) if "NO-" not in label else (255, 0, 0)
                    cv2.rectangle(frame, (int(bx1), int(by1)), (int(bx2), int(by2)), color, 2)
                    cv2.putText(frame, label, (int(bx1), int(by1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(csv_file, 'a', newline='') as file:
            worn = ', '.join(person_equipment[obj_id]['worn']) or "None"
            not_worn = ', '.join(person_equipment[obj_id]['not_worn']) or "None"
            file.write(f"{timestamp},{obj_id},{worn},{not_worn}\n")

    return frame, person_equipment


EXPECTED_COLUMNS = ["Timestamp", "Person ID", "Equipment Worn", "Equipment Not Worn"]

def initialize_csv(file_path):
    """Ensure the CSV exists and has correct columns."""
    if not os.path.exists(file_path):
        st.warning(f"{file_path} not found, initializing new file.")
        pd.DataFrame(columns=EXPECTED_COLUMNS).to_csv(file_path, index=False)
        return

    # Validate the header and reset if corrupted
    try:
        df = pd.read_csv(file_path, nrows=1)
        if list(df.columns) != EXPECTED_COLUMNS:
            raise ValueError(f"{file_path} columns are incorrect.")
    except Exception:
        st.warning(f"{file_path} is corrupted or has incorrect format. Resetting file.")
        pd.DataFrame(columns=EXPECTED_COLUMNS).to_csv(file_path, index=False)

def load_and_validate_csv(file_path):
    """Load CSV with error handling; initialize if necessary."""
    initialize_csv(file_path)
    try:
        return pd.read_csv(file_path, on_bad_lines="skip")
    except Exception as e:
        st.error(f"Failed to read {file_path}: {e}")
        return pd.DataFrame(columns=EXPECTED_COLUMNS)

def plot_class_distribution(df, title):
    """ Helper function to plot class distribution from a DataFrame """
    if df.empty:
        st.warning(f"No data available for {title}. Please process some videos first.")
        return

    all_classes = []

    # Ensure missing values are replaced with empty strings before splitting
    df = df.fillna("")  # Convert NaN values to empty strings

    for worn, not_worn in zip(df["Equipment Worn"], df["Equipment Not Worn"]):
        all_classes.extend(str(worn).split(", "))  # Convert to string before splitting
        all_classes.extend(str(not_worn).split(", "))

    # Only keep non-empty class names
    all_classes = [cls.strip() for cls in all_classes if cls.strip()]

    if not all_classes:
        st.warning(f"No valid class data found for {title}.")
        return

    # Generate Class Distribution Plot
    class_counts = pd.Series(all_classes).value_counts()

    fig, ax = plt.subplots()
    sns.barplot(x=class_counts.index, y=class_counts.values, ax=ax)
    ax.set_title(f"{title} - Class Distribution")
    plt.xticks(rotation=45)
    st.pyplot(fig)


# Sidebar Navigation
with st.sidebar:
    st.image(st.session_state.profile_image, width=100)
    page = st.radio(f"Welcome!", ["Dashboard", "Set Receiver Email", "Analytics", "About Me"])

# Dashboard
if page == "Dashboard":
    st.title("🚧 Construction Safety Dashboard")
    tab1, tab2 = st.tabs(["📹 Webcam Detection", "📼 Upload Video"])

    # Webcam
    with tab1:
        if st.button("Start Webcam"):
            st.session_state.stop_webcam = False

        if st.button("Stop Webcam"):
            st.session_state.stop_webcam = True

        detected_classes = set()
        if not st.session_state.stop_webcam:
            cap = cv2.VideoCapture(0)
            placeholder = st.empty()

            while cap.isOpened() and not st.session_state.stop_webcam:
                ret, frame = cap.read()
                if not ret:
                    break
                frame, equipment = process_frame(frame, webcam_csv_file)
                for eq in equipment.values():
                    detected_classes.update(eq['worn'] + eq['not_worn'])
                placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

            cap.release()

        st.write("### Detected Classes")
        st.write(detected_classes)

    # Video
    with tab2:
        uploaded_file = st.file_uploader("Upload Video", type=["mp4", "avi"])

        if uploaded_file:
            temp_fd, temp_path = tempfile.mkstemp(suffix=".mp4")
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.read())

            cap = cv2.VideoCapture(temp_path)
            st.info("Processing video...")
            progress_bar = st.progress(0)

            detected_classes = set()
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            for i in range(total_frames):
                ret, frame = cap.read()
                if not ret:
                    break
                frame, equipment = process_frame(frame, video_csv_file)
                for eq in equipment.values():
                    detected_classes.update(eq['worn'] + eq['not_worn'])
                progress_bar.progress((i+1) / total_frames)

            cap.release()
            os.close(temp_fd)
            os.remove(temp_path)

            st.success("Processing Complete")
            st.write("### Detected Classes")
            st.write(detected_classes)

# Receiver Email
elif page == "Set Receiver Email":
    email = st.text_input("Receiver Email", st.session_state.receiver_email)
    if st.button("Save"):
        st.session_state.receiver_email = email
        st.success("Receiver Email Updated!")

# Analytics
elif page == "Analytics":
    st.title("📊 Safety Analytics")

    tab1, tab2 = st.tabs(["📹 Webcam Data", "📼 Uploaded Video Data"])

    # Webcam Data Tab
    with tab1:
        df_webcam = load_and_validate_csv(webcam_csv_file)
        if df_webcam.empty:
            st.warning("No webcam data found. Run the webcam detection first.")
        else:
            plot_class_distribution(df_webcam, "Webcam")
            st.write("### Raw Data")
            st.dataframe(df_webcam)

    # Uploaded Video Data Tab
    with tab2:
        df_video = load_and_validate_csv(video_csv_file)
        if df_video.empty:
            st.warning("No uploaded video data found. Process a video first.")
        else:
            plot_class_distribution(df_video, "Uploaded Video")
            st.write("### Raw Data")
            st.dataframe(df_video)
# About Me
elif page == "About Me":
    name = st.text_input("Name", "John Doe")
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    profile_image = st.file_uploader("Upload Profile Image")

    if profile_image:
        with open("profile.png", "wb") as f:
            f.write(profile_image.read())
        st.session_state.profile_image = "profile.png"

    st.image(st.session_state.profile_image, width=150)
