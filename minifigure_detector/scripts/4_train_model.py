# scripts/4_train_model.py
import os
from pathlib import Path
from dotenv import load_dotenv
import yaml
import logging
import torch
from ultralytics import YOLO
from datetime import datetime

class MinifigureTrainer:
    def __init__(self):
        load_dotenv()
        self.base_dir = Path(os.getenv('DATASET_PATH', 'dataset'))
        self.setup_logging()
        self.setup_paths()
        self.setup_config()

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

    def setup_paths(self):
        """Setup necessary directories"""
        self.data_yaml = self.base_dir / 'data.yaml'
        self.weights_dir = Path('weights')
        self.weights_dir.mkdir(exist_ok=True)

    def setup_config(self):
        """Create YAML configuration for training"""
        config = {
            'path': str(self.base_dir),
            'train': str(self.base_dir / 'train' / 'images'),
            'val': str(self.base_dir / 'val' / 'images'),
            'test': str(self.base_dir / 'test' / 'images'),
            'names': {
                0: 'minifigure'
            }
        }
        
        with open(self.data_yaml, 'w') as f:
            yaml.dump(config, f)

    def train_model(self, epochs=100, img_size=640, batch_size=16, resume=False):
        """Train the YOLOv8 model"""
        try:
            # Initialize model
            model = YOLO('yolov8n.pt')  # Load pretrained YOLOv8n model
            
            # Get device
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.logger.info(f"Using device: {device}")
            
            # Training arguments
            args = {
                'data': str(self.data_yaml),
                'epochs': epochs,
                'imgsz': img_size,
                'batch': batch_size,
                'device': device,
                'name': f'minifig_detector_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
                'resume': resume
            }
            
            # Start training
            self.logger.info("Starting training...")
            results = model.train(**args)
            
            # Save training results
            results_file = self.weights_dir / 'training_results.yaml'
            with open(results_file, 'w') as f:
                yaml.dump(results, f)
            
            self.logger.info(f"Training completed. Results saved to {results_file}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error during training: {e}")
            raise

def main():
    trainer = MinifigureTrainer()
    
    # Get training parameters from user
    try:
        print("\nEnter training parameters:")
        epochs = int(input("Number of epochs (default 100): ") or 100)
        img_size = int(input("Image size (default 640): ") or 640)
        batch_size = int(input("Batch size (default 16): ") or 16)
        resume = input("Resume training? (y/N): ").lower() == 'y'
        
        # Start training
        results = trainer.train_model(
            epochs=epochs,
            img_size=img_size,
            batch_size=batch_size,
            resume=resume
        )
        
        # Print summary
        print("\nTraining completed!")
        print(f"Final mAP@0.5: {results.maps[0]:.3f}")
        print(f"Best weights saved to: {results.best}")
        
    except ValueError as e:
        print(f"Invalid input: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()