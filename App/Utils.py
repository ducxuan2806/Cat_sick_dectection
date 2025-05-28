import tempfile
import time

import numpy as np
import cv2
import pandas as pd
import streamlit as st
from PIL import Image

class Utils:
    def __init__(self, model):
        pass
        self.model = model

    def display_detected_frames(self,  image):
        results = self.model(image)
        annotated_frame = results[0].plot()
        return annotated_frame, results[0].boxes



    def infer_uploaded_image(self):
            source_img = st.sidebar.file_uploader(
                label="Choose an image...",
                type=("jpg", "jpeg", "png", 'bmp', 'webp')
            )

            col1, col2 = st.columns(2)


            if source_img:
                uploaded_image = Image.open(source_img)
                uploaded_image = np.array(uploaded_image)

                if uploaded_image.shape[-1] == 4:
                    uploaded_image = uploaded_image[..., :3]

                with col1:
                    st.image(uploaded_image, caption="Uploaded Image", use_container_width=True)

                if st.button("Execution"):
                    with st.spinner("Running..."):
                        annotated_frame, boxes = self.display_detected_frames(uploaded_image)

                        with col2:
                            st.image(annotated_frame, caption="Detected Result", use_container_width=True)

    def infer_uploaded_video(self):
        source_video = st.sidebar.file_uploader(
            label="Choose a video...",
            type=("mp4", "avi", "flv", "mov", "wmv")
        )

        col1, col2 = st.columns(2)
        with col1:
            if source_video:
                st.video(source_video)

        if source_video:
            if st.button("Execution"):
                with st.spinner("Running..."):
                    try:
                        # Lưu video tạm
                        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                        tfile.write(source_video.read())
                        tfile.close()

                        # Đọc video
                        vid_cap = cv2.VideoCapture(tfile.name)
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        width = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        height = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                        # Tạo writer để ghi video đã gán nhãn
                        output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

                        while vid_cap.isOpened():
                            success, frame = vid_cap.read()
                            if not success:
                                break

                            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            annotated_frame, _ = self.display_detected_frames(frame_rgb)
                            frame_bgr = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
                            out.write(frame_bgr)

                        vid_cap.release()
                        out.release()
                        time.sleep(0.5)

                        # Đọc lại video để hiển thị
                        with open(output_path, "rb") as f:
                            video_bytes = f.read()

                        st.success("Execution completed. Annotated video below:")
                        st.video(video_bytes)

                        with open(output_path, "rb") as f:
                            st.download_button("Download annotated video", f, "annotated_video.mp4")

                    except Exception as e:
                        st.error(f"Error processing video: {e}")

    def infer_uploaded_webcam(self):
        try:
            st.warning("Webcam sẽ hiển thị trong thời gian thực, nhấn STOP để kết thúc.")
            stop_button = st.button("Stop")

            vid_cap = cv2.VideoCapture(0)  # Webcam
            st_frame = st.empty()

            while vid_cap.isOpened() and not stop_button:
                success, frame = vid_cap.read()
                if not success:
                    break

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                annotated_frame, _ = self.display_detected_frames(frame_rgb)
                st_frame.image(annotated_frame, channels="RGB")

            vid_cap.release()
        except Exception as e:
            st.error(f"Error accessing webcam: {e}")

