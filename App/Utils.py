import tempfile
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
            type = ("mp4", "avi", "flv", "mov", "wmv")
        )
        col1, col2 = st.columns(2)

        with col1:
            if source_video:
                st.video(source_video)

        if source_video:
            if st.button("Execution"):
                with st.spinner("Running..."):
                    try:
                        tfile = tempfile.NamedTemporaryFile()
                        tfile.write(source_video.read())
                        vid_cap = cv2.VideoCapture(
                            tfile.name)
                        while (vid_cap.isOpened()):
                            success, image = vid_cap.read()
                            if success:
                                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                                annotated_frame, _ = self.display_detected_frames(image)
                                st.image(annotated_frame, channels="RGB")
                            else:
                                vid_cap.release()
                                break
                    except Exception as e:
                        st.error(f"Error loading video: {e}")


    def infer_uploaded_webcam(self):
        try:
            flag = st.button(
                label="Stop running"
            )

            vid_cap = cv2.VideoCapture(0)  # local camera
            col1, col2 = st.columns(2)

            with col1:
                st_frame = st.empty()  # Placeholder for webcam feed

            with col2:
                label_placeholder = st.empty()
            while vid_cap.isOpened() and not flag:
                success, image = vid_cap.read()
                if success:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    st_frame.image(image, channels = "RGB")
                    annotated_frame, _ = self.display_detected_frames(image)
                    st.image(annotated_frame, channels="RGB")

                else:
                    vid_cap.release()
                    break

                if cv2.waitKey(1) & 0xFF == ord('q') or flag:
                    break
            vid_cap.release()
            cv2.destroyAllWindows()
        except Exception as e:
            st.error(f"Error loading video: {str(e)}")
