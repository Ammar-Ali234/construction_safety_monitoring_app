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
import smtplib
from email.message import EmailMessage
import ssl

st.set_page_config(page_title="Construction Safety Dashboard", page_icon="🚧", layout="wide")

DEFAULT_IMAGE_URL = "https://cdn-icons-png.flaticon.com/512/9131/9131478.png"

webcam_csv_file = 'webcam_ppe_tracking.csv'
video_csv_file = 'video_ppe_tracking.csv'

for file in [webcam_csv_file, video_csv_file]:
    if not os.path.exists(file):
        pd.DataFrame(columns=["Timestamp", "Person ID", "Equipment Worn", "Equipment Not Worn"]).to_csv(file, index=False)

if "receiver_email" not in st.session_state:
    st.session_state.receiver_email = "mammarali299@gmail.com"

if "profile_image" not in st.session_state:
    if not os.path.exists("default_profile.jpg"):
        response = requests.get(DEFAULT_IMAGE_URL)
        with open("default_profile.jpg", "wb") as f:
            f.write(response.content)
    st.session_state.profile_image = "default_profile.jpg"

if "webcam_active" not in st.session_state:
    st.session_state.webcam_active = False

if "session_data" not in st.session_state:
    st.session_state.session_data = {}

if "yolo_model" not in st.session_state:
    st.session_state.yolo_model = YOLO('ppe.pt')
    st.session_state.tracker = Sort()

def send_email_with_attachment(receiver_email, csv_file, image_file):
    sender_email = "mammarali299@gmail.com"
    sender_password = "fwouqdkyedxulbol"  # Use App Password if using Gmail

    subject = "Safety Monitoring Report"
    body = "Please find attached the safety report and violation snapshot."

    msg = EmailMessage()
    msg["From"] = sender_email
    msg["To"] = receiver_email
    msg["Subject"] = subject
    msg.set_content(body)

    with open(csv_file, "rb") as f:
        msg.add_attachment(f.read(), maintype="application", subtype="csv", filename=os.path.basename(csv_file))

    with open(image_file, "rb") as f:
        msg.add_attachment(f.read(), maintype="image", subtype="jpeg", filename=os.path.basename(image_file))

    context = ssl.create_default_context()
    with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
        server.login(sender_email, sender_password)
        server.send_message(msg)

    print(f"Email sent to {receiver_email}")

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
    page = st.radio("Welcome!", ["Dashboard", "Set Receiver Email", "Analytics", "About Me"])

if page == "Dashboard":
    st.title("🚧 Construction Safety Dashboard")

    if st.button("Start Webcam"):
        st.session_state.webcam_active = True
        st.session_state.session_data = {}

    if st.button("Stop Webcam"):
        st.session_state.webcam_active = False
        save_session_data(webcam_csv_file)

        if "last_frame" in st.session_state:
            evidence_image = "violation_snapshot.jpg"
            cv2.imwrite(evidence_image, st.session_state.last_frame)
            send_email_with_attachment(st.session_state.receiver_email, webcam_csv_file, evidence_image)

    if st.session_state.webcam_active:
        cap = cv2.VideoCapture(0)
        placeholder = st.empty()

        while cap.isOpened() and st.session_state.webcam_active:
            ret, frame = cap.read()
            if not ret:
                break

            frame = process_frame(frame)
            placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

        cap.release()

    st.write("### Detected Classes")
    detected_classes = set()
    for equipment in st.session_state.session_data.values():
        detected_classes.update(equipment['worn'])
        detected_classes.update(equipment['not_worn'])
    st.write(detected_classes)

elif page == "Set Receiver Email":
    st.title("📧 Set Receiver Email")
    email = st.text_input("Receiver Email", st.session_state.receiver_email)
    if st.button("Save"):
        st.session_state.receiver_email = email
        st.success("Receiver email updated!")

elif page == "Analytics":
    st.title("📊 Safety Analytics")

    data_source = st.radio("Select Data Source", ["📹 Webcam Data", "📼 Uploaded Video Data"])

    file = webcam_csv_file if data_source == "📹 Webcam Data" else video_csv_file
    label = data_source

    if os.path.exists(file):
        df = pd.read_csv(file)
        df["Person ID"] = df["Person ID"].astype(str)  # Ensure IDs are strings for consistency

        # Function to plot violations using Seaborn
        def plot_violations_by_person(df, title):
            if df.empty:
                st.warning(f"No data available for {title}.")
                return

            # Initialize violation tracker
            violation_summary = {}

            # Loop through each row to gather violations for each person
            for index, row in df.iterrows():
                person_id = row["Person ID"]
                not_worn_list = str(row["Equipment Not Worn"]).split()

                if person_id not in violation_summary:
                    violation_summary[person_id] = {"NO-Hardhat": 0, "NO-Mask": 0, "NO-Safety Vest": 0}

                for item in not_worn_list:
                    item = item.strip()
                    if item in violation_summary[person_id]:
                        violation_summary[person_id][item] += 1

            # Convert to DataFrame
            violation_df = pd.DataFrame.from_dict(violation_summary, orient="index").reset_index()
            violation_df = violation_df.rename(columns={"index": "Person ID"})
            violation_df = violation_df.fillna(0)

            # Convert to long format for Seaborn plotting
            long_df = violation_df.melt(id_vars=["Person ID"], var_name="Violation Type", value_name="Count")

            st.write(f"### {title} - Violations by Person ID")

            # Plot using Seaborn
            sns.set(style="whitegrid")
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.barplot(
                x="Person ID", y="Count", hue="Violation Type", data=long_df, ax=ax, palette="Reds"
            )
            ax.set_title(f"{title} - Equipment Not Worn by Person ID")
            ax.set_xlabel("Person ID")
            ax.set_ylabel("Violation Count")
            plt.xticks(rotation=45)
            st.pyplot(fig)

        # Function to calculate and display Safety Check Summary
        def plot_safety_check_summary(df, title):
            if df.empty:
                st.warning(f"No data available for {title}.")
                return

            safety_check = {}

            for index, row in df.iterrows():
                person_id = row["Person ID"]
                not_worn_list = str(row["Equipment Not Worn"]).split()

                if person_id not in safety_check:
                    safety_check[person_id] = "✅ Pass"

                for item in not_worn_list:
                    if item.startswith("NO-"):
                        safety_check[person_id] = "❌ Fail"
                        break  # One violation is enough to fail

            # Convert safety check summary to DataFrame
            safety_df = pd.DataFrame(list(safety_check.items()), columns=["Person ID", "Safety Check"])

            st.write(f"### {title} - Safety Check Summary")
            st.dataframe(safety_df)

            # Plotting Pass/Fail counts
            summary_count = safety_df["Safety Check"].value_counts()

            st.write(f"### {title} - Pass/Fail Summary")
            fig, ax = plt.subplots()
            sns.barplot(x=summary_count.index, y=summary_count.values, ax=ax, palette=["#4caf50", "#f44336"])
            ax.set_ylabel("Count")
            ax.set_title(f"{title} - Safety Check Result")
            st.pyplot(fig)

        # Show Violations Chart and Safety Check Summary
        plot_violations_by_person(df, label)
        plot_safety_check_summary(df, label)

        # Show raw data
        st.write(f"### {label} - Raw Data")
        st.dataframe(df)

    else:
        st.warning(f"No data found for {label}. Please run detection first.")


elif page == "About Me":
    st.text_input("Name", "John Doe")
    st.selectbox("Gender", ["Male", "Female", "Other"])
    st.file_uploader("Upload Profile Image")
    st.image(st.session_state.profile_image, width=150)
