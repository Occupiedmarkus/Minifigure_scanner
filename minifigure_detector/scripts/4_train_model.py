# scripts/4_train_model.py
from ultralytics import YOLO
from pathlib import Path
import os
from dotenv import load_dotenv
import logging
import yaml
import torch
import shutil
from datetime import datetime
import json
from tqdm import tqdm

class ModelTrainer:
    def __init__(self):
        # Load environment variables
        load_dotenv()
        
        # Setup paths
        self.base_dir = Path(os.getenv('DATASET_PATH', 'dataset'))
        self.model_dir = Path(os.getenv('MODEL_PATH', 'models/weights')).parent
        self.model_weights = self.model_dir / 'best.pt'
        
        # Training configuration
        self.setup_logging()
        self.load_training_config()
        
        # Training state
        self.current_best_metric = 0
        self.epochs_without_improvement = 0

    def setup_logging(self):
        """Setup logging configuration"""
        log_dir = Path('logs')
        log_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(log_dir / 'training.log')
            ]
        )
        self.logger = logging.getLogger(__name__)

    def load_training_config(self):
        """Load or create training configuration"""
        config_path = self.model_dir / 'training_config.yaml'
        
        if config_path.exists():
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = self.create_default_config()
            with open(config_path, 'w') as f:
                yaml.dump(self.config, f)

    def create_default_config(self):
        """Create default training configuration"""
        return {
            'model_type': 'yolov8n.pt',  # nano model
            'epochs': 100,
            'batch_size': 16,
            'img_size': 640,
            'patience': 20,  # early stopping patience
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'workers': 4,
            'resume': True,  # resume training if possible
            'save_period': 10,  # save checkpoint every N epochs
        }

    def find_latest_checkpoint(self):
        """Find the latest checkpoint"""
        checkpoints = list(self.model_dir.glob('*.pt'))
        if not checkpoints:
            return None
        
        # Sort by modification time
        return max(checkpoints, key=os.path.getmtime)

    def save_checkpoint(self, model, epoch, metrics):
        """Save training checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'metrics': metrics,
            'config': self.config
        }
        
        checkpoint_path = self.model_dir / f'checkpoint_epoch_{epoch}.pt'
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Saved checkpoint: {checkpoint_path}")

    def cleanup_old_checkpoints(self, keep_last_n=5):
        """Clean up old checkpoints keeping only the last N"""
        checkpoints = list(self.model_dir.glob('checkpoint_epoch_*.pt'))
        if len(checkpoints) > keep_last_n:
            # Sort by modification time
            checkpoints.sort(key=os.path.getmtime)
            # Remove oldest checkpoints
            for checkpoint in checkpoints[:-keep_last_n]:
                checkpoint.unlink()
                self.logger.info(f"Removed old checkpoint: {checkpoint}")

    def prepare_training(self):
        """Prepare for training"""
        # Check if CUDA is available
        if torch.cuda.is_available():
            self.logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            self.logger.info("Using CPU for training")

        # Create model directory
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # Load dataset configuration
        data_yaml = self.base_dir / 'data.yaml'
        if not data_yaml.exists():
            raise FileNotFoundError(f"Dataset configuration not found: {data_yaml}")

        return data_yaml

    def train_model(self):
        """Train the YOLO model"""
        try:
            data_yaml = self.prepare_training()
            
            # Initialize model
            if self.config['resume']:
                checkpoint = self.find_latest_checkpoint()
                if checkpoint:
                    self.logger.info(f"Resuming from checkpoint: {checkpoint}")
                    model = YOLO(checkpoint)
                else:
                    self.logger.info(f"Starting new training with {self.config['model_type']}")
                    model = YOLO(self.config['model_type'])
            else:
                model = YOLO(self.config['model_type'])

            # Training arguments
            train_args = {
                'data': str(data_yaml),
                'epochs': self.config['epochs'],
                'imgsz': self.config['img_size'],
                'batch': self.config['batch_size'],
                'device': self.config['device'],
                'workers': self.config['workers'],
                'patience': self.config['patience'],
                'project': str(self.model_dir.parent),
                'name': self.model_dir.name,
                'exist_ok': True,
                'pretrained': True,
                'optimizer': 'Adam',  # Using Adam optimizer
                'verbose': True,
                'save_period': self.config['save_period'],
                'val': True,  # Run validation
            }

            # Start training
            self.logger.info("Starting training with configuration:")
            self.logger.info(yaml.dump(train_args))

            # Train the model
            results = model.train(**train_args)

            # Save final model
            shutil.copy(self.model_dir / 'weights' / 'best.pt', self.model_weights)
            self.logger.info(f"Training completed. Best model saved to {self.model_weights}")

            # Save training results
            results_file = self.model_dir / 'training_results.json'
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=4)

            return True

        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            return False

    def validate_model(self):
        """Validate the trained model"""
        if not self.model_weights.exists():
            self.logger.error("No trained model found for validation")
            return
        
        try:
            model = YOLO(self.model_weights)
            
            # Run validation on test set
            val_results = model.val(
                data=str(self.base_dir / 'data.yaml'),
                split='test'  # Use test set for final validation
            )
            
            # Save validation results
            results_file = self.model_dir / 'validation_results.json'
            with open(results_file, 'w') as f:
                json.dump(val_results, f, indent=4)
            
            self.logger.info("Validation completed. Results saved.")
            
        except Exception as e:
            self.logger.error(f"Validation failed: {e}")

def main():
    trainer = ModelTrainer()
    
    # Ask for training configuration
    print("\nCurrent training configuration:")
    print(yaml.dump(trainer.config))
    
    if input("\nWould you like to modify the configuration? (y/n): ").lower() == 'y':
        trainer.config['epochs'] = int(input("Enter number of epochs (default 100): ") or 100)
        trainer.config['batch_size'] = int(input("Enter batch size (default 16): ") or 16)
        trainer.config['img_size'] = int(input("Enter image size (default 640): ") or 640)
        trainer.config['patience'] = int(input("Enter early stopping patience (default 20): ") or 20)
    
    # Start training
    if trainer.train_model():
        print("\nTraining completed successfully!")
        
        # Run validation
        if input("\nWould you like to run validation on the test set? (y/n): ").lower() == 'y':
            trainer.validate_model()

if __name__ == "__main__":
    main()