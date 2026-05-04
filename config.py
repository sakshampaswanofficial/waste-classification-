%%writefile config.py
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

# Standardize logging format across the pipeline
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

@dataclass(frozen=True)
class TrainingConfig:
    """Hyperparameters and configuration for the EfficientNetB0 model."""
    input_shape: Tuple[int, int, int] = (224, 224, 3)
    num_classes: int = 20
    batch_size: int = 32
    epochs_phase_1: int = 15
    epochs_phase_2: int = 15
    learning_rate_base: float = 1e-3
    learning_rate_fine: float = 1e-5
    hazardous_multiplier: float = 8.0
    hazardous_prefix: str = "hw_"
    seed: int = 42

#@dataclass(frozen=True)
#class PathConfig:
    #"""Filesystem path configurations."""
    #base_dir: Path = Path(__file__).parent.resolve()
#    data_dir: Path = base_dir / "data" / "master_dataset"
 #   model_output_dir: Path = base_dir / "artifacts"
#    best_model_path: Path = base_dir / "artifacts" / "best_production_model.keras"



# Updated PathConfig for Kaggle
@dataclass(frozen=True)
class PathConfig:
    # Point this to wherever your master dataset is mounted in Kaggle
    data_dir: Path = Path("/kaggle/input/ultimate-master-dataset/") 
    
    # Kaggle requires all outputs to go to the working directory
    model_output_dir: Path = Path("/kaggle/working/artifacts/")
    best_model_path: Path = Path("/kaggle/working/artifacts/best_production_model.keras")

config = TrainingConfig()
paths = PathConfig()
