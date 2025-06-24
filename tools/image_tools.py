from typing import ClassVar, Optional
from langchain.tools import BaseTool
from transformers import pipeline, AutoProcessor, AutoModelForCausalLM
from PIL import Image
import torch
import os


# --- Base Class for Image Tools ---

class ImageToolBase(BaseTool):
    """
    Base class to handle common model loading and device setup for image-based tools.
    """

    _model_id: str
    _task: str
    _hf_token: Optional[str] = None
    _device: str
    _pipeline: Optional[pipeline] = None

    def __init__(self, model_id: str, task: str, **kwargs):
        super().__init__(**kwargs)
        self._model_id = model_id
        self._task = task
        self._hf_token = os.getenv("HF_API_TOKEN")
        if not self._hf_token:
            print("Warning: HF_API_TOKEN environment variable not set. Access to gated models may fail.")

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self._device}")

    def _load_pipeline(self):
        """Loads a standard Hugging Face pipeline."""
        if self._pipeline is None:
            print(f"Loading {self._model_id} pipeline for task '{self._task}'...")
            try:
                pipeline_device = 0 if self._device == "cuda" else -1

                self._pipeline = pipeline(
                    self._task,
                    model=self._model_id,
                    device=pipeline_device,
                    token=self._hf_token,
                    torch_dtype=(torch.float16 if self._device == "cuda" else torch.float32),
                    # Keep trust_remote_code=True for general compatibility, though not always needed for standard HF models
                    trust_remote_code=True
                )
                print(f"{self._model_id} pipeline loaded successfully.")
            except Exception as e:
                print(f"Error loading {self._model_id} pipeline: {e}")
                self._pipeline = None
                raise

    def _arun(self, query: str):
        raise NotImplementedError("Async execution not supported for image tools.")


# --- Image Captioning Tool ---

class ImageCaptionTool(ImageToolBase):
    name: ClassVar[str] = "Image captioner"
    description: ClassVar[str] = (
        "Use this tool to generate a caption describing the content of an image. "
        "Input: local file path to the image. Output: a descriptive caption string."
    )

    model: Optional[AutoModelForCausalLM] = None
    processor: Optional[AutoProcessor] = None

    def __init__(self, model_id: str = "Salesforce/blip-image-captioning-large"):
        super().__init__(model_id=model_id, task="image-to-text")
        print(f"Initializing ImageCaptionTool with model: {self._model_id}")
        self._load_pipeline()
        self.model = None
        self.processor = None

    def _run(self, img_path: str) -> str:
        try:
            image = Image.open(img_path).convert('RGB')
            if self._pipeline:
                result = self._pipeline(image, max_new_tokens=50)
                if isinstance(result, list) and result:
                    return result[0].get('generated_text', 'No caption generated.').strip()
                return "No caption generated."
            else:
                return "Captioning pipeline not initialized or failed to load."
        except Exception as e:
            return f"Error during captioning: {e}"


# --- Object Detection Tool ---

class ObjectDetectionTool(ImageToolBase):
    name: ClassVar[str] = "Object detector"
    description: ClassVar[str] = (
        "Use this tool to detect objects in an image. "
        "Input: local file path to the image. Output: list of [x1,y1,x2,y2] class_name score."
    )

    def __init__(self, model_id: str = "facebook/detr-resnet-50"): # *** CHANGED DEFAULT MODEL ID HERE ***
        super().__init__(model_id=model_id, task="object-detection")
        self._load_pipeline()

    def _run(self, img_path: str) -> str:
        if not self._pipeline:
            return "Detection pipeline not loaded."
        try:
            image = Image.open(img_path).convert('RGB')
            results = self._pipeline(image, threshold=0.3)

            detections = []
            for item in results:
                box = item.get('box', {})
                label = item.get('label', '')
                score = item.get('score', 0)
                if box and label:
                    detections.append(
                        f"[{int(box.get('xmin', 0))},{int(box.get('ymin', 0))},{int(box.get('xmax', 0))},{int(box.get('ymax', 0))}] {label} {score:.4f}"
                    )

            return "\n".join(detections) if detections else "No objects detected."
        except Exception as e:
            return f"Error during detection: {e}"