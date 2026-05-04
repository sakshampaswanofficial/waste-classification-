%%writefile data_pipeline.py
import logging
from pathlib import Path
from typing import Tuple

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import DirectoryIterator
from tensorflow.keras.applications.efficientnet import preprocess_input

from config import config

logger = logging.getLogger(__name__)

class DataPipeline:
    """
    Manages the ingestion, augmentation, and preprocessing of the image dataset.
    """

    def __init__(self, data_directory: Path):
        """
        Args:
            data_directory: Root directory containing class subdirectories.
        """
        self.data_dir = data_directory
        self.train_generator: DirectoryIterator = None
        self.val_generator: DirectoryIterator = None
        
        self._initialize_generators()

    def _initialize_generators(self) -> None:
        """Builds and attaches the Keras ImageDataGenerators."""
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {self.data_dir}")

        logger.info("Initializing augmentations and data streams...")

        datagen = ImageDataGenerator(
            preprocessing_function=preprocess_input,
            validation_split=0.2,
            rotation_range=40,
            width_shift_range=0.3,
            height_shift_range=0.3,
            brightness_range=[0.5, 1.5],
            zoom_range=0.3,
            horizontal_flip=True,
            vertical_flip=True
        )

        target_size = (config.input_shape[0], config.input_shape[1])

        self.train_generator = datagen.flow_from_directory(
            directory=str(self.data_dir),
            target_size=target_size,
            batch_size=config.batch_size,
            class_mode='categorical',
            subset='training',
            seed=config.seed
        )

        self.val_generator = datagen.flow_from_directory(
            directory=str(self.data_dir),
            target_size=target_size,
            batch_size=config.batch_size,
            class_mode='categorical',
            subset='validation',
            seed=config.seed
        )
        
        logger.info(f"Identified {self.train_generator.num_classes} total classes.")

    def get_generators(self) -> Tuple[DirectoryIterator, DirectoryIterator]:
        """Returns the configured train and validation iterators."""
        return self.train_generator, self.val_generato
