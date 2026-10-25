import logging
import os
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

from rlbench_data import REPO_ROOT, frame_step, load_demo

logger = logging.getLogger(__name__)


class MultimodalInferenceModel(nn.Module):
    """
    Multimodal model that combines vision and telemetry features for robotic action prediction.

    Input: the front camera frame (ResNet-18 embedding, 512) and the 7 joint positions at that frame.
    Output: the 7 joint deltas the expert applied next. The fusion head is trained on RLBench
    demonstrations (python train.py); the inputs and outputs are standardized with statistics stored
    in the weights file.
    """

    def __init__(self, vision_embedding_dim: int = 512, telemetry_dim: int = 7,
                 hidden_dim: int = 256, output_dim: int = 7, use_vision: bool = True):
        super().__init__()
        self.use_vision = use_vision
        self.vision_embedding_dim = vision_embedding_dim if use_vision else 0
        self.telemetry_dim = telemetry_dim

        if use_vision:
            # Vision encoder (ResNet18, ImageNet weights), frozen
            self.vision_encoder = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            # Remove the final classification layer
            self.vision_encoder.fc = nn.Identity()
            for param in self.vision_encoder.parameters():
                param.requires_grad = False
            self.vision_encoder.eval()

        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        # MLP head for fusion
        fusion_dim = self.vision_embedding_dim + telemetry_dim
        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

        # Standardization learned from the training data (saved with the weights)
        self.register_buffer("input_mean", torch.zeros(fusion_dim))
        self.register_buffer("input_std", torch.ones(fusion_dim))
        self.register_buffer("output_mean", torch.zeros(output_dim))
        self.register_buffer("output_std", torch.ones(output_dim))

    def extract_vision_features(self, rgb_image: Image.Image) -> torch.Tensor:
        """Extract features from an RGB image using ResNet18."""
        return self.embed_images([rgb_image])

    def embed_images(self, images: List[Image.Image], batch_size: int = 64) -> torch.Tensor:
        if not self.use_vision:
            return torch.zeros(len(images), 0)
        out = []
        with torch.no_grad():
            for i in range(0, len(images), batch_size):
                batch = torch.stack([self.transform(img.convert("RGB")) for img in images[i:i + batch_size]])
                out.append(self.vision_encoder(batch))
        return torch.cat(out)

    def extract_telemetry_features(self, telemetry: Dict, step: int = -1) -> torch.Tensor:
        """The 7 joint positions at one frame of an episode (load_demo output)."""
        joint_positions = telemetry["joint_positions"][step]
        return torch.tensor(joint_positions[:self.telemetry_dim], dtype=torch.float32).unsqueeze(0)

    def forward(self, vision_features: torch.Tensor, telemetry_features: torch.Tensor) -> torch.Tensor:
        """Joint deltas (radians) from vision and telemetry features."""
        fused_features = torch.cat([vision_features, telemetry_features], dim=1) if self.use_vision \
            else telemetry_features
        normalized = (fused_features - self.input_mean) / self.input_std
        return self.fusion_head(normalized) * self.output_std + self.output_mean

    def predict(self, rgb_path: str, pkl_path: str) -> np.ndarray:
        """
        Main inference method: a frame (front_rgb/<step>.png) and its episode's low_dim_obs.pkl ->
        the predicted action vector for that step.
        """
        step = frame_step(rgb_path)
        telemetry = load_demo(pkl_path)
        if not 0 <= step < len(telemetry["joint_positions"]):
            raise ValueError(f"frame {step} has no telemetry ({len(telemetry['joint_positions'])} steps)")
        vision_features = self.extract_vision_features(Image.open(rgb_path)) if self.use_vision \
            else torch.zeros(1, 0)
        telemetry_features = self.extract_telemetry_features(telemetry, step)

        self.eval()
        with torch.no_grad():
            action = self.forward(vision_features, telemetry_features)
        return action.squeeze(0).numpy()


class ModelManager:
    """Loads the model a manifest describes: architecture, inputs and trained weights."""

    def __init__(self, manifest_path: str = "manifest.yaml"):
        import yaml

        self.manifest_path = manifest_path
        with open(manifest_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.model = build_model(self.config)
        weights = weights_path(self.config)
        if not os.path.exists(weights):
            logger.info(f"No trained weights at {weights}; training them now (python train.py {manifest_path})")
            from train import train_manifest
            train_manifest(manifest_path)
        # The weights file holds the trained head and its standardization; ResNet-18 comes from torchvision
        missing, unexpected = self.model.load_state_dict(
            torch.load(weights, map_location="cpu", weights_only=True), strict=False)
        if unexpected or any(not k.startswith("vision_encoder.") for k in missing):
            raise ValueError(f"{weights} doesn't match {manifest_path}: missing {missing}, unexpected {unexpected}")
        self.model.eval()

        logger.info(f"Loaded model with manifest version {self.config['version']} from {weights}")
        logger.info(f"Vision: {self.config['vision']}, Telemetry: {self.config['telemetry']}")
        logger.info(f"Policy: {self.config['policy']}, Fusion: {self.config['fusion']}")

    def get_model(self) -> MultimodalInferenceModel:
        return self.model

    def get_version(self) -> str:
        return self.config['version']

    def get_config(self) -> Dict:
        return self.config


def build_model(config: Dict) -> MultimodalInferenceModel:
    model_settings = config.get('model_settings', {})
    return MultimodalInferenceModel(
        vision_embedding_dim=model_settings.get('vision_embedding_dim', 512),
        telemetry_dim=model_settings.get('telemetry_dim', 7),
        hidden_dim=model_settings.get('hidden_dim', 256),
        output_dim=model_settings.get('output_dim', 7),
        use_vision=config.get('vision', 'resnet18') != 'none',
    )


def weights_path(config: Dict) -> str:
    return os.path.join(REPO_ROOT, config.get('weights', f"models/{config['version']}.pt"))


# Utility function to find the latest timestep image in an episode directory
def find_latest_image(rgb_dir_path: str) -> str:
    """Find the latest timestep image in the RGB directory."""
    import glob

    png_files = glob.glob(os.path.join(rgb_dir_path, "*.png"))
    if not png_files:
        raise FileNotFoundError(f"No PNG files found in {rgb_dir_path}")

    # Sort by numeric filename to get the latest timestep
    png_files.sort(key=frame_step)
    return png_files[-1]  # Return the latest timestep
