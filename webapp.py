import streamlit as st
import cv2
import numpy as np
import pandas as pd
import os
import tempfile
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from ultralytics import YOLO
from sort import Sort
import requests
import smtplib
from email.message import EmailMessage
import ssl

st.set_page_config(page_title="Construction Safety Dashboard", page_icon="🚧", layout="wide")

# Custom CSS
st.markdown("""
    <style>
    .stApp {background-color: #FFFFFF; color: #333333;}
    [data-testid="stSidebar"] {background-color: #808080; color: #FFFFFF; padding: 10px;}
    h1, h2, h3 {color: #FFA500; font-family: 'Arial', sans-serif; font-weight: bold; text-shadow: 1px 1px 2px #333333;}
    .stButton>button {background-color: #FFA500; color: #FFFFFF; border: 2px solid #333333; border-radius: 5px; font-weight: bold; box-shadow: 2px 2px 4px #808080;}
    .stButton>button:hover {background-color: #FFFF00; color: #333333; box-shadow: 4px 4px 8px #808080;}
    .stTabs [data-baseweb="tab"] {background-color: #808080; color: #FFFFFF; font-weight: bold; border-radius: 5px 5px 0 0;}
    .stTabs [data-baseweb="tab"][aria-selected="true"] {background-color: #FFA500; color: #FFFFFF;}
    .image-frame {border: 4px solid #FFA500; border-radius: 10px; padding: 5px; background-color: #808080; box-shadow: 3px 3px 6px #333333;}
    .username {text-align: center; color: #FFFF00; font-weight: bold; margin-top: 5px; text-shadow: 1px 1px 2px #333333;}
    .stDataFrame {border: 2px solid #FFA500; border-radius: 5px;}
    .highlight-container {border: 2px solid #FFA500; border-radius: 10px; padding: 10px; background-color: #F5F5F5; box-shadow: 2px 2px 5px #808080;}
    </style>
""", unsafe_allow_html=True)

DEFAULT_IMAGE_URL = "https://cdn-icons-png.flaticon.com/512/9131/9131478.png"
webcam_csv_file = 'webcam_ppe_tracking.csv'
video_csv_file = 'video_ppe_tracking.csv'

# Initialize CSV files if they don't exist
for file in [webcam_csv_file, video_csv_file]:
    if not os.path.exists(file):
        pd.DataFrame(columns=["Timestamp", "Person ID", "Equipment Worn", "Equipment Not Worn"]).to_csv(file, index=False)

# Session state initialization
if "receiver_email" not in st.session_state:
    st.session_state.receiver_email = "RECEIVER EMAIL"
if "profile_image" not in st.session_state:
    if not os.path.exists("default_profile.jpg"):
        response = requests.get(DEFAULT_IMAGE_URL)
        with open("default_profile.jpg", "wb") as f:
            f.write(response.content)
    st.session_state.profile_image = "default_profile.jpg"
if "webcam_active" not in st.session_state:
    st.session_state.webcam_active = False
if "video_active" not in st.session_state:
    st.session_state.video_active = False
if "session_data" not in st.session_state:
    st.session_state.session_data = {}
if "username" not in st.session_state:
    st.session_state.username = "SafetyInspector"
if "yolo_model" not in st.session_state:
    st.session_state.yolo_model = YOLO('ppe.pt')
    st.session_state.tracker = Sort()

def send_email_with_attachment(receiver_email, csv_file, image_file):
    sender_email = "EMAIL"
    sender_password = "PASSWORD"  # Use App Password for Gmail
    subject = "Safety Monitoring Report"
    body = "Please find attached the safety report and violation snapshot."

    msg = EmailMessage()
    msg["From"] = sender_email
    msg["To"] = receiver_email
    msg["Subject"] = subject
    msg.set_content(body)

    try:
        with open(csv_file, "rb") as f:
            msg.add_attachment(f.read(), maintype="application", subtype="csv", filename=os.path.basename(csv_file))
        with open(image_file, "rb") as f:
            msg.add_attachment(f.read(), maintype="image", subtype="jpeg", filename=os.path.basename(image_file))
        
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
            server.login(sender_email, sender_password)
            server.send_message(msg)
        st.success(f"Email sent to {receiver_email}")
    except Exception as e:
        st.error(f"Failed to send email: {str(e)}")

def save_session_data(csv_file):
    if not st.session_state.session_data:
        return
    if not os.path.exists(csv_file):
        pd.DataFrame(columns=["Timestamp", "Person ID", "Equipment Worn", "Equipment Not Worn"]).to_csv(csv_file, index=False)

    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(csv_file, 'a', newline='') as file:
        for obj_id, equipment in st.session_state.session_data.items():
            worn = ' '.join(equipment['worn']) or "None"
            not_worn = ' '.join(equipment['not_worn']) or "None"
            file.write(f"{timestamp},{obj_id},{worn},{not_worn}\n")

def process_frame(frame):
    model = st.session_state.yolo_model
    tracker = st.session_state.tracker

    results = model(frame)
    detections = []

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
        if obj_id not in st.session_state.session_data:
            st.session_state.session_data[obj_id] = {"worn": set(), "not_worn": set()}

        for result in results:
            for box in result.boxes:
                bx1, by1, bx2, by2 = box.xyxy[0]
                label = model.names[int(box.cls.item())]
                conf = box.conf.item()
                if conf >= 0.4 and label != "Person":
                    if x1 <= bx1 <= x2 and y1 <= by1 <= y2:
                        if label.startswith("NO-"):
                            st.session_state.session_data[obj_id]["not_worn"].add(label)
                        else:
                            st.session_state.session_data[obj_id]["worn"].add(label)
                    color = (0, 255, 0) if "NO-" not in label else (255, 0, 0)
                    cv2.rectangle(frame, (int(bx1), int(by1)), (int(bx2), int(by2)), color, 2)
                    cv2.putText(frame, label, (int(bx1), int(by1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(frame, f"ID: {obj_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    st.session_state.last_frame = frame
    return frame

with st.sidebar:
    st.image(st.session_state.profile_image, width=100)
    page_options = {
        "🏠 Dashboard": "Dashboard",
        "✉️ Set Receiver Email": "Set Receiver Email",
        "📈 Analytics": "Analytics",
        "👷 About Me": "About Me"
    }
    page = st.radio("Navigation", list(page_options.keys()), format_func=lambda x: x)
    selected_page = page_options[page]

if selected_page == "Dashboard":
    st.title("🚧 Construction Safety Dashboard")
    tab1, tab2 = st.tabs(["📹 Webcam", "📼 Video Upload"])
    
    with tab1:
        col1, col2 = st.columns([3, 1])
        with col1:
            fps = st.slider("Webcam FPS", 1, 30, 15, key="webcam_fps")
            if st.button("▶️ Start Webcam", key="start_webcam"):
                st.session_state.webcam_active = True
                st.session_state.session_data = {}
            if st.button("⏹️ Stop Webcam", key="stop_webcam"):
                st.session_state.webcam_active = False
                save_session_data(webcam_csv_file)
                if "last_frame" in st.session_state:
                    evidence_image = "violation_snapshot.jpg"
                    cv2.imwrite(evidence_image, st.session_state.last_frame)
                    send_email_with_attachment(st.session_state.receiver_email, webcam_csv_file, evidence_image)

            if st.session_state.webcam_active:
                cap = cv2.VideoCapture(0)
                if not cap.isOpened():
                    st.error("Error: Could not open webcam.")
                    st.session_state.webcam_active = False
                else:
                    placeholder = st.empty()
                    try:
                        while st.session_state.webcam_active:
                            ret, frame = cap.read()
                            if not ret:
                                st.error("Error: Could not read frame from webcam.")
                                break
                            frame = process_frame(frame)
                            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            placeholder.image(frame_rgb, caption=st.session_state.username, use_column_width=True)
                            time.sleep(1/fps)
                    except Exception as e:
                        st.error(f"Webcam error: {str(e)}")
                    finally:
                        cap.release()
                        st.session_state.webcam_active = False

        with col2:
            with st.container():
                st.markdown('<div class="highlight-container">', unsafe_allow_html=True)
                st.write("### 🔔 Detected Classes (Webcam)")
                detected_classes = set()
                for equipment in st.session_state.session_data.values():
                    detected_classes.update(equipment['worn'])
                    detected_classes.update(equipment['not_worn'])
                st.write(detected_classes)
                st.markdown('</div>', unsafe_allow_html=True)

    with tab2:
        col1, col2 = st.columns([3, 1])
        with col1:
            fps = st.slider("Video FPS", 1, 30, 15, key="video_fps")
            uploaded_file = st.file_uploader("📤 Upload Video", type=["mp4", "avi", "mov"])
            if uploaded_file:
                if st.button("▶️ Process Video", key="start_video"):
                    st.session_state.video_active = True
                    st.session_state.session_data = {}
                if st.button("⏹️ Stop Processing", key="stop_video"):
                    st.session_state.video_active = False
                    save_session_data(video_csv_file)
                    if "last_frame" in st.session_state:
                        evidence_image = "violation_snapshot.jpg"
                        cv2.imwrite(evidence_image, st.session_state.last_frame)
                        send_email_with_attachment(st.session_state.receiver_email, video_csv_file, evidence_image)

                if st.session_state.video_active:
                    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                    tfile.write(uploaded_file.read())
                    cap = cv2.VideoCapture(tfile.name)
                    if not cap.isOpened():
                        st.error("Error: Could not open video file.")
                        st.session_state.video_active = False
                    else:
                        placeholder = st.empty()
                        try:
                            while cap.isOpened() and st.session_state.video_active:
                                ret, frame = cap.read()
                                if not ret:
                                    break
                                frame = process_frame(frame)
                                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                placeholder.image(frame_rgb, caption=st.session_state.username, use_column_width=True)
                                time.sleep(1/fps)
                        except Exception as e:
                            st.error(f"Video processing error: {str(e)}")
                        finally:
                            cap.release()
                            os.unlink(tfile.name)
                            st.session_state.video_active = False

        with col2:
            with st.container():
                st.markdown('<div class="highlight-container">', unsafe_allow_html=True)
                st.write("### 🔔 Detected Classes (Video)")
                detected_classes = set()
                for equipment in st.session_state.session_data.values():
                    detected_classes.update(equipment['worn'])
                    detected_classes.update(equipment['not_worn'])
                st.write(detected_classes)
                st.markdown('</div>', unsafe_allow_html=True)

elif selected_page == "Set Receiver Email":
    st.title("✉️ Set Receiver Email")
    with st.container():
        st.markdown('<div class="highlight-container">', unsafe_allow_html=True)
        email = st.text_input("Receiver Email", st.session_state.receiver_email)
        if st.button("💾 Save"):
            st.session_state.receiver_email = email
            st.success("Receiver email updated!")
        st.markdown('</div>', unsafe_allow_html=True)

elif selected_page == "Analytics":
    st.title("📈 Safety Analytics")
    tab1, tab2 = st.tabs(["📹 Webcam Data", "📼 Video Data"])
    
    def plot_violations_by_person(df, title, container):
        if df.empty:
            container.warning(f"No data available for {title}.")
            return
        violation_summary = {}
        for index, row in df.iterrows():
            person_id = row["Person ID"]
            not_worn_list = str(row["Equipment Not Worn"]).split()
            if person_id not in violation_summary:
                violation_summary[person_id] = {"NO-Hardhat": 0, "NO-Mask": 0, "NO-Safety Vest": 0}
            for item in not_worn_list:
                item = item.strip()
                if item in violation_summary[person_id]:
                    violation_summary[person_id][item] += 1
        violation_df = pd.DataFrame.from_dict(violation_summary, orient="index").reset_index()
        violation_df = violation_df.rename(columns={"index": "Person ID"})
        violation_df = violation_df.fillna(0)
        long_df = violation_df.melt(id_vars=["Person ID"], var_name="Violation Type", value_name="Count")
        container.write(f"### 📊 {title} - Violations by Person ID")
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.set_style("whitegrid")
        sns.barplot(x="Person ID", y="Count", hue="Violation Type", data=long_df, ax=ax, 
                   palette={"NO-Hardhat": "#FFA500", "NO-Mask": "#FFFF00", "NO-Safety Vest": "#808080"})
        ax.set_title(f"{title} - Equipment Not Worn by Person ID", color="#FFA500")
        ax.set_xlabel("Person ID", color="#333333")
        ax.set_ylabel("Violation Count", color="#333333")
        plt.xticks(rotation=45, color="#333333")
        plt.yticks(color="#333333")
        ax.spines['top'].set_color('#808080')
        ax.spines['right'].set_color('#808080')
        ax.spines['left'].set_color('#808080')
        ax.spines['bottom'].set_color('#808080')
        plt.tight_layout()
        container.pyplot(fig)
        plt.close()

    def plot_safety_check_summary(df, title, container):
        if df.empty:
            container.warning(f"No data available for {title}.")
            return
        safety_check = {}
        for index, row in df.iterrows():
            person_id = row["Person ID"]
            not_worn_list = str(row["Equipment Not Worn"]).split()
            if person_id not in safety_check:
                safety_check[person_id] = "Pass"
            for item in not_worn_list:
                if item.startswith("NO-"):
                    safety_check[person_id] = "Fail"
                    break
        safety_df = pd.DataFrame(list(safety_check.items()), columns=["Person ID", "Safety Check"])
        container.write(f"### ✅ {title} - Safety Check Summary")
        container.dataframe(safety_df)
        summary_count = safety_df["Safety Check"].value_counts()
        container.write(f"### 📊 {title} - Pass/Fail Summary")
        fig, ax = plt.subplots()
        sns.set_style("whitegrid")
        sns.barplot(x=summary_count.index, y=summary_count.values, hue=summary_count.index, 
                   palette={"Pass": "#FFFF00", "Fail": "#FFA500"}, ax=ax, legend=False)
        ax.set_ylabel("Count", color="#333333")
        ax.set_title(f"{title} - Safety Check Result", color="#FFA500")
        plt.xticks(color="#333333")
        plt.yticks(color="#333333")
        ax.spines['top'].set_color('#808080')
        ax.spines['right'].set_color('#808080')
        ax.spines['left'].set_color('#808080')
        ax.spines['bottom'].set_color('#808080')
        plt.tight_layout()
        container.pyplot(fig)
        plt.close()

    with tab1:
        with st.container():
            st.markdown('<div class="highlight-container">', unsafe_allow_html=True)
            if os.path.exists(webcam_csv_file):
                df = pd.read_csv(webcam_csv_file)
                df["Person ID"] = df["Person ID"].astype(str)
                plot_violations_by_person(df, "Webcam Data", tab1)
                plot_safety_check_summary(df, "Webcam Data", tab1)
                tab1.write("### 📋 Webcam Data - Raw Data")
                tab1.dataframe(df)
            else:
                tab1.warning("No webcam data found. Please run detection first.")
            st.markdown('</div>', unsafe_allow_html=True)

    with tab2:
        with st.container():
            st.markdown('<div class="highlight-container">', unsafe_allow_html=True)
            if os.path.exists(video_csv_file):
                df = pd.read_csv(video_csv_file)
                df["Person ID"] = df["Person ID"].astype(str)
                plot_violations_by_person(df, "Video Data", tab2)
                plot_safety_check_summary(df, "Video Data", tab2)
                tab2.write("### 📋 Video Data - Raw Data")
                tab2.dataframe(df)
            else:
                tab2.warning("No video data found. Please run detection first.")
            st.markdown('</div>', unsafe_allow_html=True)

elif selected_page == "About Me":
    st.title("👷 About Me")
    with st.container():
        st.markdown('<div class="highlight-container">', unsafe_allow_html=True)
        username = st.text_input("Username", st.session_state.username)
        if st.button("💾 Update Username"):
            st.session_state.username = username
            st.success("Username updated!")
        st.selectbox("Gender", ["Male", "Female", "Other"])
        uploaded_profile = st.file_uploader("📸 Upload Profile Image")
        if uploaded_profile:
            with open("default_profile.jpg", "wb") as f:
                f.write(uploaded_profile.getbuffer())
            st.session_state.profile_image = "default_profile.jpg"
        st.image(st.session_state.profile_image, width=150)
        st.markdown('</div>', unsafe_allow_html=True)
