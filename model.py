import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import pickle
import numpy as np
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)

class MultimodalInferenceModel(nn.Module):
    """
    Multimodal model that combines vision and telemetry features for robotic action prediction.
    """
    
    def __init__(self, vision_embedding_dim: int = 512, telemetry_dim: int = 7, 
                 hidden_dim: int = 256, output_dim: int = 7):
        super().__init__()
        
        # Vision encoder (ResNet18)
        self.vision_encoder = models.resnet18(pretrained=True)
        # Remove the final classification layer
        self.vision_encoder.fc = nn.Identity()
        
        # Freeze vision encoder for inference only
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
        fusion_dim = vision_embedding_dim + telemetry_dim
        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        self.vision_embedding_dim = vision_embedding_dim
        self.telemetry_dim = telemetry_dim
        
    def extract_vision_features(self, rgb_image: Image.Image) -> torch.Tensor:
        """Extract features from RGB image using ResNet18."""
        # Preprocess image
        image_tensor = self.transform(rgb_image).unsqueeze(0)  # Add batch dimension
        
        # Extract features
        with torch.no_grad():
            vision_features = self.vision_encoder(image_tensor)
        
        return vision_features
    
    def extract_telemetry_features(self, low_dim_obs: Dict) -> torch.Tensor:
        """Extract telemetry features from low_dim_obs dictionary."""
        
        # Handle RLBench Demo objects that were unpickled as custom objects
        if not hasattr(low_dim_obs, 'keys'):
            logger.warning("RLBench Demo object detected, converting to dict format")
            # Convert Demo object to dict if it has useful attributes
            demo_dict = {}
            for attr in dir(low_dim_obs):
                if not attr.startswith('_'):
                    try:
                        val = getattr(low_dim_obs, attr)
                        demo_dict[attr] = val
                    except:
                        pass
            low_dim_obs = demo_dict
        
        # First check if this is actually observation data from RLBench
        # RLBench observation data should have specific keys like 'joint_velocities', 'joint_positions', etc.
        joint_positions = None
        
        # Try to find joint position data in various possible formats
        possible_keys = [
            'joint_positions', 'robot_joint_positions', 'arm_joint_positions',
            'joint_velocities', 'joint_forces', 'gripper_joint_positions'
        ]
        
        for key in possible_keys:
            if hasattr(low_dim_obs, 'keys') and key in low_dim_obs:
                candidate = low_dim_obs[key]
                if candidate is not None and hasattr(candidate, '__len__'):
                    try:
                        # Convert to numpy array for processing
                        candidate_array = np.array(candidate)
                        if candidate_array.size > 0:
                            joint_positions = candidate_array
                            logger.info(f"Using {key} for telemetry data: shape={joint_positions.shape}")
                            break
                    except Exception as e:
                        logger.debug(f"Could not use {key}: {e}")
                        continue
        
        # If still no joint positions found, this might be a Demo container rather than observation data
        if joint_positions is None:
            logger.warning("This appears to be RLBench Demo metadata rather than observation data")
            logger.warning("RLBench demos should be loaded differently - each demo contains a sequence of observations")
            logger.warning("For now, using dummy joint positions")
            joint_positions = np.zeros(self.telemetry_dim)
        else:
            # Process found joint positions
            # If we have a sequence (multiple timesteps), take the last one
            if joint_positions.ndim > 1:
                joint_positions = joint_positions[-1]  # Take last timestep
                logger.info(f"Using last timestep from sequence, final shape: {joint_positions.shape}")
            
            # Ensure we have the right dimension
            if len(joint_positions) > self.telemetry_dim:
                joint_positions = joint_positions[:self.telemetry_dim]
                logger.info(f"Truncated to {self.telemetry_dim} dimensions")
            elif len(joint_positions) < self.telemetry_dim:
                # Pad with zeros if needed
                padding = np.zeros(self.telemetry_dim - len(joint_positions))
                joint_positions = np.concatenate([joint_positions, padding])
                logger.info(f"Padded to {self.telemetry_dim} dimensions")
        
        return torch.tensor(joint_positions, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
    
    def forward(self, vision_features: torch.Tensor, telemetry_features: torch.Tensor) -> torch.Tensor:
        """Forward pass through the fusion network."""
        # Concatenate features
        fused_features = torch.cat([vision_features, telemetry_features], dim=1)
        
        # Pass through MLP head
        action = self.fusion_head(fused_features)
        
        return action
    
    def predict(self, rgb_path: str, pkl_path: str) -> np.ndarray:
        """
        Main inference method that takes file paths and returns action vector.
        """
        # Load and process RGB image
        try:
            rgb_image = Image.open(rgb_path).convert('RGB')
            vision_features = self.extract_vision_features(rgb_image)
        except Exception as e:
            logger.error(f"Error processing RGB image {rgb_path}: {e}")
            raise
        
        # Load and process telemetry data
        try:
            try:
                with open(pkl_path, 'rb') as f:
                    low_dim_obs = pickle.load(f)
            except ModuleNotFoundError as e:
                logger.warning(f"RLBench module not found (expected): {e}")
                logger.info("Attempting to load with custom unpickler...")
                
                # Create a dummy unpickler that ignores missing modules
                class RLBenchUnpickler(pickle.Unpickler):
                    def find_class(self, module, name):
                        try:
                            return super().find_class(module, name)
                        except (ModuleNotFoundError, AttributeError):
                            # Return a dummy class for missing modules
                            return type(name, (), {})
                
                with open(pkl_path, 'rb') as f:
                    low_dim_obs = RLBenchUnpickler(f).load()
                
                # Convert object to dict if necessary
                if not hasattr(low_dim_obs, 'keys'):
                    obj_dict = {}
                    for attr in dir(low_dim_obs):
                        if not attr.startswith('_'):
                            try:
                                obj_dict[attr] = getattr(low_dim_obs, attr)
                            except:
                                pass
                    low_dim_obs = obj_dict
            
            telemetry_features = self.extract_telemetry_features(low_dim_obs)
        except Exception as e:
            logger.error(f"Error processing telemetry data {pkl_path}: {e}")
            raise
        
        # Run inference
        self.eval()
        with torch.no_grad():
            action = self.forward(vision_features, telemetry_features)
        
        return action.squeeze(0).numpy()  # Remove batch dimension and convert to numpy


class ModelManager:
    """Manages model loading and configuration from manifest."""
    
    def __init__(self, manifest_path: str = "manifest.yaml"):
        import yaml
        
        with open(manifest_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize model based on manifest
        model_settings = self.config.get('model_settings', {})
        self.model = MultimodalInferenceModel(
            vision_embedding_dim=model_settings.get('vision_embedding_dim', 512),
            telemetry_dim=model_settings.get('telemetry_dim', 7),
            hidden_dim=model_settings.get('hidden_dim', 256),
            output_dim=model_settings.get('output_dim', 7)
        )
        
        logger.info(f"Loaded model with manifest version {self.config['version']}")
        logger.info(f"Vision: {self.config['vision']}, Telemetry: {self.config['telemetry']}")
        logger.info(f"Policy: {self.config['policy']}, Fusion: {self.config['fusion']}")
    
    def get_model(self) -> MultimodalInferenceModel:
        return self.model
    
    def get_version(self) -> str:
        return self.config['version']
    
    def get_config(self) -> Dict:
        return self.config


# Utility function to find the latest timestep image in an episode directory
def find_latest_image(rgb_dir_path: str) -> str:
    """Find the latest timestep image in the RGB directory."""
    import os
    import glob
    
    # Look for PNG files in the directory
    png_files = glob.glob(os.path.join(rgb_dir_path, "*.png"))
    
    if not png_files:
        raise FileNotFoundError(f"No PNG files found in {rgb_dir_path}")
    
    # Sort by numeric filename to get the latest timestep
    png_files.sort(key=lambda x: int(os.path.basename(x).split('.')[0]))
    
    return png_files[-1]  # Return the latest timestep