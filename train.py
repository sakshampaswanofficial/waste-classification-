%%writefile train.py
import logging
import numpy as np
import tensorflow as tf
from typing import Dict, List
from pathlib import Path

from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from config import config, paths
from data_pipeline import DataPipeline

logger = logging.getLogger(__name__)

class WasteClassifierTrainer:
    """
    Constructs, compiles, and trains the hierarchical waste classification model.
    """

    def __init__(self, data_pipeline: DataPipeline):
        self.train_data, self.val_data = data_pipeline.get_generators()
        self.model: Model = self._build_architecture()
        self.class_weights: Dict[int, float] = self._calculate_hybrid_weights()
        
        # Ensure output directory exists before training
        paths.model_output_dir.mkdir(parents=True, exist_ok=True)

    def _calculate_hybrid_weights(self) -> Dict[int, float]:
        """
        Calculates mathematical class balance using scikit-learn, then applies
        an asymmetric penalty multiplier for hazardous classes.
        
        Returns:
            Dictionary mapping class indices to their computed hybrid weight.
        """
        logger.info("Computing hybrid class weights with safety penalties...")
        
        unique_classes = np.unique(self.train_data.classes)
        balanced_weights_array = compute_class_weight(
            class_weight='balanced',
            classes=unique_classes,
            y=self.train_data.classes
        )
        
        safe_balanced_weights = dict(zip(unique_classes, balanced_weights_array))
        final_weights: Dict[int, float] = {}

        for class_name, class_idx in self.train_data.class_indices.items():
            base_weight = safe_balanced_weights.get(class_idx, 1.0)
            
            if class_name.startswith(config.hazardous_prefix):
                final_weights[class_idx] = base_weight * config.hazardous_multiplier
                logger.info(f"Hazard Multiplier applied to: {class_name} (Index {class_idx})")
            else:
                final_weights[class_idx] = base_weight

        return final_weights

    def _build_architecture(self) -> Model:
        """Constructs the EfficientNetB0 backbone with custom top layers."""
        logger.info("Building model architecture (EfficientNetB0 Backbone)...")
        
        base_model = EfficientNetB0(
            weights='imagenet', 
            include_top=False, 
            input_shape=config.input_shape
        )
        base_model.trainable = False

        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.4)(x)
        predictions = Dense(config.num_classes, activation='softmax')(x)

        return Model(inputs=base_model.input, outputs=predictions)

    def _get_callbacks(self) -> List[tf.keras.callbacks.Callback]:
        """Instantiates the training lifecycle callbacks."""
        return [
            EarlyStopping(
                monitor='val_accuracy', 
                patience=5, 
                restore_best_weights=True, 
                verbose=1
            ),
            ModelCheckpoint(
                filepath=str(paths.best_model_path), 
                monitor='val_accuracy', 
                save_best_only=True, 
                mode='max', 
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss', 
                factor=0.5, 
                patience=2, 
                min_lr=1e-6, 
                verbose=1
            )
        ]

    def execute_training_pipeline(self) -> None:
        """Executes the two-phase training protocol."""
        callbacks = self._get_callbacks()

        # Phase 1: Feature Extraction
        logger.info("Phase 1: Training top layers (Feature Extraction)")
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=config.learning_rate_base),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        try:
            self.model.fit(
                self.train_data,
                validation_data=self.val_data,
                epochs=config.epochs_phase_1,
                class_weight=self.class_weights,
                callbacks=callbacks
            )
        except Exception as e:
            logger.error(f"Phase 1 training failed: {e}")
            raise

        # Phase 2: Deep Fine-Tuning
        logger.info("Phase 2: Deep Fine-Tuning top 50 EfficientNet layers")
        self.model.trainable = True
        for layer in self.model.layers[:-50]:
            layer.trainable = False

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=config.learning_rate_fine),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        try:
            self.model.fit(
                self.train_data,
                validation_data=self.val_data,
                epochs=config.epochs_phase_2,
                class_weight=self.class_weights,
                callbacks=callbacks
            )
            logger.info(f"Pipeline complete. Production model saved to {paths.best_model_path}")
        except Exception as e:
            logger.error(f"Phase 2 fine-tuning failed: {e}")
            raise

if __name__ == "__main__":
    try:
        pipeline = DataPipeline(data_directory=paths.data_dir)
        trainer = WasteClassifierTrainer(data_pipeline=pipeline)
        trainer.execute_training_pipeline()
    except Exception as fatal_error:
        logger.critical(f"Pipeline execution aborted due to unhandled exception: {fatal_error}")
