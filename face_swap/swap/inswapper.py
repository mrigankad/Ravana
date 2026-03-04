"""
InSwapper (InsightFace) face swap model.

This uses the pre-trained inswapper_128.onnx or inswapper_128_fp16.onnx
models from InsightFace, which are readily available.
"""

from typing import Optional
import numpy as np
import cv2

from .base import FaceSwapper
from ..core.types import AlignedFace, Embedding, SwapResult


class InSwapperModel(FaceSwapper):
    """
    InSwapper face swapping model from InsightFace.
    
    This is a practical implementation that uses readily available
    pre-trained models (inswapper_128.onnx).
    
    Model specs:
    - Input: 128x128 face image
    - ID embedding: 512-dim ArcFace embedding
    - Output: 128x128 swapped face
    """
    
    DEFAULT_MODEL_URL = "https://github.com/deepinsight/insightface/releases/download/v0.7/inswapper_128.onnx"
    
    def __init__(
        self,
        device: str = "cuda",
        model_path: str = "./models/inswapper_128.onnx",
        resolution: int = 128,
    ):
        """
        Initialize InSwapper model.
        
        Args:
            device: Device to run inference on
            model_path: Path to inswapper_128.onnx model
            resolution: Model input/output resolution (128)
        """
        super().__init__(device, resolution, use_enhancer=False)
        self.model_path = model_path
        self._session = None
    
    def load_model(self, model_path: Optional[str] = None) -> None:
        """
        Load the InSwapper ONNX model.
        
        Args:
            model_path: Path to model file (defaults to self.model_path)
        """
        import os
        
        path = model_path or self.model_path
        
        # Download model if not exists
        if not os.path.exists(path):
            self._download_model(path)
        
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError("onnxruntime is required. Install with: pip install onnxruntime-gpu")
        
        providers = ["CUDAExecutionProvider"] if self.device == "cuda" else ["CPUExecutionProvider"]
        
        self._session = ort.InferenceSession(path, providers=providers)
        
        # Get input/output info
        self._input_names = [inp.name for inp in self._session.get_inputs()]
        self._output_names = [out.name for out in self._session.get_outputs()]
        
        # Expected inputs:
        # - target: (1, 3, 128, 128) - target face
        # - source: (1, 512) - source embedding
    
    def _download_model(self, path: str) -> None:
        """Download pre-trained model if not available locally."""
        import os
        import urllib.request
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        print(f"Downloading InSwapper model to {path}...")
        urllib.request.urlretrieve(self.DEFAULT_MODEL_URL, path)
        print("Download complete.")
    
    def swap(
        self,
        target_aligned: AlignedFace,
        source_embedding: Embedding,
    ) -> SwapResult:
        """
        Generate a swapped face.
        
        Args:
            target_aligned: Aligned target face
            source_embedding: Source identity embedding
            
        Returns:
            SwapResult with swapped face and mask
        """
        if self._session is None:
            self.load_model()
        
        # Prepare target image
        target_img = target_aligned.image
        
        # Resize to 128x128
        if target_img.shape[:2] != (128, 128):
            target_img = cv2.resize(target_img, (128, 128))
        
        # Preprocess target
        target_tensor = self._preprocess_image(target_img)
        
        # Preprocess embedding
        id_tensor = self._preprocess_embedding(source_embedding)
        
        # Run inference
        inputs = {
            self._input_names[0]: target_tensor,
            self._input_names[1]: id_tensor
        }
        
        outputs = self._session.run(self._output_names, inputs)
        
        # Post-process
        swapped_face = self._postprocess_image(outputs[0])
        
        # Generate mask
        mask = self.get_mask(swapped_face)
        
        return SwapResult(
            swapped_face=swapped_face,
            mask=mask,
            source_embedding=source_embedding,
            target_aligned=target_aligned,
            quality_score=0.85  # InSwapper typically produces good quality
        )
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for InSwapper input.
        
        Args:
            image: Input image (H, W, C) BGR
            
        Returns:
            Preprocessed array (1, 3, 128, 128)
        """
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1]
        image_norm = image_rgb.astype(np.float32) / 255.0
        
        # Transpose to (C, H, W) and add batch dimension
        image_tensor = np.transpose(image_norm, (2, 0, 1))
        image_tensor = np.expand_dims(image_tensor, axis=0)
        
        return image_tensor
    
    def _preprocess_embedding(self, embedding: Embedding) -> np.ndarray:
        """
        Preprocess embedding for InSwapper.
        
        InSwapper expects 512-dim ArcFace embeddings.
        
        Args:
            embedding: Identity embedding
            
        Returns:
            Preprocessed embedding (1, 512)
        """
        vector = embedding.vector.astype(np.float32)
        
        # Ensure correct dimension
        if len(vector) != 512:
            # Pad or truncate
            if len(vector) < 512:
                vector = np.pad(vector, (0, 512 - len(vector)))
            else:
                vector = vector[:512]
        
        # Normalize if not already
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        
        # Add batch dimension
        return np.expand_dims(vector, axis=0)
    
    def _postprocess_image(self, output: np.ndarray) -> np.ndarray:
        """
        Post-process model output.
        
        Args:
            output: Model output (1, 3, 128, 128)
            
        Returns:
            Output image (128, 128, 3) BGR
        """
        # Remove batch dimension
        if output.ndim == 4:
            output = output[0]
        
        # Transpose from (C, H, W) to (H, W, C)
        if output.shape[0] == 3:
            output = np.transpose(output, (1, 2, 0))
        
        # Denormalize from [0, 1] to [0, 255]
        output = (output * 255).clip(0, 255).astype(np.uint8)
        
        # Convert RGB to BGR
        output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
        
        return output
    
    def get_mask(self, swapped_face: np.ndarray) -> np.ndarray:
        """
        Generate blending mask.
        
        Args:
            swapped_face: Swapped face image
            
        Returns:
            Soft mask (0-1)
        """
        h, w = swapped_face.shape[:2]
        
        # Create oval mask
        center = (w // 2, int(h * 0.45))  # Slightly above center for face
        axes = (w // 2 - 8, int(h * 0.45) - 8)
        
        mask = np.zeros((h, w), dtype=np.float32)
        cv2.ellipse(mask, center, axes, 0, 0, 360, 1.0, -1)
        
        # Soften edges
        mask = cv2.GaussianBlur(mask, (15, 15), 7)
        
        return mask
