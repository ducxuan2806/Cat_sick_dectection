import streamlit as st
from pathlib import Path
from ultralytics import YOLO
from .Utils import Utils
from . import Config


class CatSickDetectionApp:
    def __init__(self):
        self.utils = None
        self.model = None
        self.model_type = None
        self.model_path = None

    def setup_page(self):
        st.set_page_config(
            page_title="Detection Cat Sicks",
            page_icon="😺",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        st.title("Interactive Interface for Cat Sicks Detection")
        st.sidebar.header("Menu")

    def load_model(self):
        task_type = st.sidebar.selectbox(
            "Select Task",
            ["Detection"]
        )

        if task_type != "Detection":
            st.error("Currently only 'Detection' function is implemented")
            return

        self.model_type = st.sidebar.selectbox(
            "Select Model",
            Config.DETECTION_MODEL_LIST
        )

        if self.model_type:
            self.model_path =   Path(Config.DETECTION_MODEL_DIR, str(self.model_type) + ".pt")
        else:
            st.error("Please select a model from the sidebar.")
            return

        try:
            print(Config.DETECTION_MODEL_DIR)
            print(self.model_path)
            self.model = YOLO(self.model_path)
            self.utils = Utils(self.model)
            st.success("Model loaded successfully!")
        except Exception as e:
            st.error(f"Unable to load model. Please check the path: {self.model_path}")
            st.exception(e)

    def select_source_and_infer(self):
        st.sidebar.header("Source Config")

        source = st.sidebar.selectbox(
            "Select Source",
            Config.SOURCES_LIST
        )

        if self.utils is None:
            st.error("Model is not loaded, cannot perform inference.")
            return

        if source == "Image":
            self.utils.infer_uploaded_image()
        elif source == "Video":
            self.utils.infer_uploaded_video()
        elif source == "Webcam":
            self.utils.infer_uploaded_webcam()
        else:
            st.error("Selected source is not supported.")

    def run(self):
        self.setup_page()
        self.load_model()
        self.select_source_and_infer()
