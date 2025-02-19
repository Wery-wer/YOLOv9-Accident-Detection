import os
import subprocess

# Pastikan libGL.so.1 tersedia agar OpenCV tidak error
try:
    subprocess.run(["apt-get", "update"], check=True)
    subprocess.run(["apt-get", "install", "-y", "libgl1-mesa-glx"], check=True)
except Exception as e:
    print("Error installing libGL:", e)

import streamlit as st
from ultralytics import YOLO
import tempfile
import cv2
from pathlib import Path
import time
from PIL import Image
import gdown

# Fungsi untuk mengunduh model dari Google Drive jika belum ada
def download_model():
    model_url = "https://drive.google.com/file/d/1DSYPws_YxzAcNUTUdCSZMPijW6qavtNw/view?usp=sharing"  # Ganti dengan File ID model di Google Drive
    model_path = "SGD x 640 x 0.0005 x 80 15 5.pt"

    if not os.path.exists(model_path):
        gdown.download(model_url, model_path, quiet=False)

# Unduh model sebelum digunakan
download_model()

# Fungsi untuk membuat folder hasil prediksi
def create_prediction_folder(base_path="runs/detect"):
    base_dir = Path(base_path)
    base_dir.mkdir(parents=True, exist_ok=True)
    existing_folders = [f for f in base_dir.iterdir() if f.is_dir() and f.name.startswith("predict")]
    next_folder_num = len(existing_folders) + 1
    new_folder = base_dir / f"predict{next_folder_num}"
    new_folder.mkdir(exist_ok=False)
    return new_folder

# Fungsi untuk memutar video dengan FPS asli
def play_video_with_fps(file_path):
    vid = cv2.VideoCapture(file_path)
    stframe = st.empty()
    fps = int(vid.get(cv2.CAP_PROP_FPS))
    while vid.isOpened():
        ret, frame = vid.read()
        if not ret:
            break
        stframe.image(frame, channels="BGR", use_container_width=True)
        time.sleep(1 / fps)
    vid.release()

# Fungsi untuk memproses video dan menyimpan hasil
def process_video_and_save(file_path, confidence):
    model = YOLO('best_yolov9e(100)_2.pt')
    vid = cv2.VideoCapture(file_path)
    fps = int(vid.get(cv2.CAP_PROP_FPS))
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    save_dir = create_prediction_folder()
    prediction_output_path = save_dir / "prediction_output.webm"
    fourcc = cv2.VideoWriter_fourcc(*'VP08')
    out = cv2.VideoWriter(str(prediction_output_path), fourcc, fps, (width, height))

    no_accident_count, moderate_count, severe_count = 0, 0, 0
    captured_moderate_frames, captured_severe_frames = [], []

    while True:
        ret, frame = vid.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model.predict(source=frame_rgb, imgsz=640, conf=confidence)
        annotated_frame = results[0].plot()

        for result in results:
            for box in result.boxes:
                class_id = int(box.cls)
                if class_id == 1:
                    no_accident_count += 1
                elif class_id == 0:
                    moderate_count += 1
                    captured_moderate_frames.append(annotated_frame)
                elif class_id == 2:
                    severe_count += 1
                    captured_severe_frames.append(annotated_frame)

        annotated_frame_bgr = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
        annotated_frame_bgr = cv2.resize(annotated_frame_bgr, (width, height))
        out.write(annotated_frame_bgr)

    vid.release()
    out.release()
    return str(prediction_output_path), no_accident_count, moderate_count, severe_count, captured_moderate_frames, captured_severe_frames

# Fungsi untuk melakukan prediksi gambar
def predict_image(file_path, confidence):
    model = YOLO('best_yolov9e(100)_2.pt')
    img_bgr = cv2.imread(file_path)
    results = model.predict(source=img_bgr, imgsz=640, conf=confidence)

    no_accident_count, moderate_count, severe_count = 0, 0, 0
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls)
            if class_id == 1:
                no_accident_count += 1
            elif class_id == 0:
                moderate_count += 1
            elif class_id == 2:
                severe_count += 1

    annotated_image_bgr = results[0].plot()
    save_dir = create_prediction_folder()
    annotated_image_path = save_dir / "prediction_output.jpg"
    cv2.imwrite(str(annotated_image_path), annotated_image_bgr)
    return no_accident_count, moderate_count, severe_count, str(annotated_image_path)

# Fungsi utama Streamlit
def main():
    st.title('YOLOv9 Accident Detection')
    st.sidebar.title('Options')
    confidence = st.sidebar.slider('Confidence', min_value=0.0, max_value=1.0, value=0.5)
    uploaded_file = st.sidebar.file_uploader("Upload an image or video", type=["jpg", "jpeg", "png", "mp4", "avi"])

    if uploaded_file:
        file_type = uploaded_file.name.split('.')[-1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_type}") as tmp_file:
            tmp_file.write(uploaded_file.read())
            temp_file_path = tmp_file.name

        if file_type in ['mp4', 'avi']:
            st.write("Preview of uploaded video:")
            play_video_with_fps(temp_file_path)
            if st.button('Predict'):
                with st.spinner('Processing video...'):
                    output_video_path, no_accident_count, moderate_count, severe_count, captured_moderate_frames, captured_severe_frames = process_video_and_save(temp_file_path, confidence)
                st.success("Prediction completed.")
                st.write(f"No accident detected: {no_accident_count}")
                st.write(f"Moderate accidents detected: {moderate_count}")
                st.write(f"Severe accidents detected: {severe_count}")
                st.video(output_video_path)

                if captured_moderate_frames:
                    st.write("Captured frames of moderate accidents:")
                    cols = st.columns(3)
                    for i, frame in enumerate(captured_moderate_frames):
                        with cols[i % 3]:
                            st.image(frame, caption=f"Moderate Frame {i+1}", use_container_width=True)

                if captured_severe_frames:
                    st.write("Captured frames of severe accidents:")
                    cols = st.columns(3)
                    for i, frame in enumerate(captured_severe_frames):
                        with cols[i % 3]:
                            st.image(frame, caption=f"Severe Frame {i+1}", use_container_width=True)

        elif file_type in ['jpg', 'jpeg', 'png']:
            st.image(temp_file_path, caption='Uploaded Image', use_container_width=True)
            if st.button('Predict'):
                no_accident_count, moderate_count, severe_count, annotated_image_path = predict_image(temp_file_path, confidence)
                st.write(f"No accident detected: {no_accident_count}")
                st.write(f"Moderate accidents detected: {moderate_count}")
                st.write(f"Severe accidents detected: {severe_count}")
                st.image(annotated_image_path, caption="Prediction Output", use_container_width=True)

if __name__ == '__main__':
    main()
