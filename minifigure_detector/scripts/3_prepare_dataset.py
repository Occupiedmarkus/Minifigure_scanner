import os
import yaml
import logging
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

class MinifigureModelTrainer:
    def __init__(self):
        # Load environment variables
        load_dotenv()
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Setup paths
        self.base_dir = Path(os.getenv('DATASET_PATH', 'dataset'))
        self.labels_dir = self.base_dir / 'labels'
        self.models_dir = self.base_dir / 'models'
        self.logs_dir = self.base_dir / 'logs'
        
        # Create necessary directories
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Training parameters
        self.batch_size = 32
        self.epochs = 100
        self.learning_rate = 0.001
        self.validation_split = 0.2
        
        # Load data
        self.load_data()
        
    def load_data(self):
        """Load preprocessed data"""
        try:
            # Load numpy arrays
            self.images = np.load(self.labels_dir / 'images.npy')
            self.years = np.load(self.labels_dir / 'years.npy')
            self.themes = np.load(self.labels_dir / 'themes.npy')
            
            # Load encoders mapping
            with open(self.labels_dir / 'encoders_mapping.yaml', 'r') as f:
                self.encoders_mapping = yaml.safe_load(f)
            
            # Get number of classes from the mappings
            self.num_years = len(self.encoders_mapping['year_mapping'])
            self.num_themes = len(self.encoders_mapping['theme_mapping'])
            
            # Load metadata mapping
            with open(self.labels_dir / 'metadata_mapping.yaml', 'r') as f:
                self.metadata_mapping = yaml.safe_load(f)
            
            self.logger.info(f"""
    Data loaded successfully:
    - Images shape: {self.images.shape}
    - Years shape: {self.years.shape} (unique: {self.num_years})
    - Themes shape: {self.themes.shape} (unique: {self.num_themes})
    - Year classes: {', '.join(map(str, self.encoders_mapping['year_mapping'].keys()))}
    - Theme classes: {', '.join(map(str, self.encoders_mapping['theme_mapping'].keys()))}
    """)
            
        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            raise
        
    def create_model(self):
        """Create and compile the model"""
        # Base model - EfficientNetB0
        base_model = tf.keras.applications.EfficientNetB0(
            include_top=False,
            weights='imagenet',
            input_shape=self.images[0].shape
        )
        
        # Freeze the base model
        base_model.trainable = False
        
        # Create model
        inputs = layers.Input(shape=self.images[0].shape)
        x = tf.keras.applications.efficientnet.preprocess_input(inputs)
        
        # Base model
        x = base_model(x, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.2)(x)
        
        # Common dense layers
        x = layers.Dense(512, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        
        # Two output heads
        year_output = layers.Dense(self.num_years, activation='softmax', name='year_output')(x)
        theme_output = layers.Dense(self.num_themes, activation='softmax', name='theme_output')(x)
        
        # Create model
        model = models.Model(
            inputs=inputs,
            outputs=[year_output, theme_output]
        )
        
        # Compile model
        model.compile(
            optimizer=optimizers.Adam(learning_rate=self.learning_rate),
            loss={
                'year_output': 'sparse_categorical_crossentropy',
                'theme_output': 'sparse_categorical_crossentropy'
            },
            metrics={
                'year_output': ['accuracy'],
                'theme_output': ['accuracy']
            }
        )
        
        return model

    def create_callbacks(self, model_name):
        """Create training callbacks"""
        callbacks = [
            ModelCheckpoint(
                filepath=self.models_dir / f'{model_name}_best.h5',
                monitor='val_loss',
                save_best_only=True,
                mode='min',
                verbose=1
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=5,
                min_lr=1e-6,
                verbose=1
            ),
            TensorBoard(
                log_dir=self.logs_dir / datetime.now().strftime("%Y%m%d-%H%M%S"),
                histogram_freq=1
            )
        ]
        return callbacks

    def train_model(self):
        """Train the model"""
        try:
            # Split data
            X_train, X_val, y_year_train, y_year_val, y_theme_train, y_theme_val = train_test_split(
                self.images,
                self.years,
                self.themes,
                test_size=self.validation_split,
                random_state=42
            )
            
            # Create model
            model = self.create_model()
            model_name = f"minifig_classifier_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Create callbacks
            callbacks = self.create_callbacks(model_name)
            
            # Train model
            self.logger.info("Starting model training...")
            history = model.fit(
                X_train,
                {'year_output': y_year_train, 'theme_output': y_theme_train},
                batch_size=self.batch_size,
                epochs=self.epochs,
                validation_data=(
                    X_val,
                    {'year_output': y_year_val, 'theme_output': y_theme_val}
                ),
                callbacks=callbacks,
                verbose=1
            )
            
            # Evaluate model
            self.logger.info("Evaluating model...")
            evaluation = model.evaluate(
                X_val,
                {'year_output': y_year_val, 'theme_output': y_theme_val},
                verbose=1
            )
            
            # Generate predictions
            y_pred_year, y_pred_theme = model.predict(X_val)
            y_pred_year = np.argmax(y_pred_year, axis=1)
            y_pred_theme = np.argmax(y_pred_theme, axis=1)
            
            # Generate classification reports
            year_report = classification_report(
                y_year_val,
                y_pred_year,
                target_names=[str(k) for k in self.encoders_mapping['year_mapping'].keys()]
            )
            theme_report = classification_report(
                y_theme_val,
                y_pred_theme,
                target_names=[str(k) for k in self.encoders_mapping['theme_mapping'].keys()]
            )
            
            # Save reports
            with open(self.models_dir / f'{model_name}_evaluation.txt', 'w') as f:
                f.write("Year Classification Report:\n")
                f.write(year_report)
                f.write("\nTheme Classification Report:\n")
                f.write(theme_report)
            
            # Save model architecture
            with open(self.models_dir / f'{model_name}_architecture.yaml', 'w') as f:
                f.write(model.to_yaml())
            
            # Save training history
            np.save(self.models_dir / f'{model_name}_history.npy', history.history)
            
            self.logger.info(f"""
Training completed successfully:
- Model saved as: {model_name}
- Final validation loss: {evaluation[0]:.4f}
- Year accuracy: {evaluation[3]:.4f}
- Theme accuracy: {evaluation[4]:.4f}
""")
            
            return model, history
            
        except Exception as e:
            self.logger.error(f"Error during training: {e}")
            raise

    def fine_tune_model(self, model, learning_rate=1e-5):
        """Fine-tune the model"""
        # Unfreeze the base model
        model.layers[2].trainable = True
        
        # Recompile model with lower learning rate
        model.compile(
            optimizer=optimizers.Adam(learning_rate=learning_rate),
            loss={
                'year_output': 'sparse_categorical_crossentropy',
                'theme_output': 'sparse_categorical_crossentropy'
            },
            metrics={
                'year_output': ['accuracy'],
                'theme_output': ['accuracy']
            }
        )
        
        return model

def main():
    trainer = MinifigureModelTrainer()
    
    # Train initial model
    model, history = trainer.train_model()
    
    # Ask for fine-tuning
    if input("\nWould you like to fine-tune the model? (y/n): ").lower() == 'y':
        model = trainer.fine_tune_model(model)
        trainer.train_model()  # Train again with fine-tuned model

if __name__ == "__main__":
    main()