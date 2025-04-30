import os
import yaml
import logging
import numpy as np
from pathlib import Path
from PIL import Image
import tensorflow as tf
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from typing import List, Dict, Optional, Union, Any
import io
import time
from datetime import datetime, timedelta
import redis
import jwt
from pydantic import BaseModel, Field
import psutil
import GPUtil
from prometheus_client import Counter, Histogram, Info, generate_latest
import hashlib
from contextlib import contextmanager
import signal
import threading
import queue
import json
import tempfile
import shutil
from typing_extensions import Annotated

# Custom exceptions
class ModelError(Exception):
    """Base exception for model-related errors"""
    pass

class CacheError(Exception):
    """Base exception for cache-related errors"""
    pass

class PreprocessError(Exception):
    """Base exception for preprocessing errors"""
    pass

# Pydantic models for request/response validation
class PredictionRequest(BaseModel):
    image: str = Field(..., description="Base64 encoded image")
    confidence_threshold: float = Field(0.5, ge=0.0, le=1.0)
    return_preprocessed: bool = Field(False, description="Return preprocessed image data")

class PredictionResponse(BaseModel):
    years: List[Dict[str, Union[str, float]]]
    themes: List[Dict[str, Union[str, float]]]
    timestamp: str
    model_version: str
    processing_time: float
    cache_hit: bool

class ModelMetrics(BaseModel):
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    average_latency: float = 0.0
    cache_hits: int = 0
    gpu_usage: float = 0.0
    memory_usage: float = 0.0
    last_reload_time: str = ""

# Prometheus metrics
PREDICTION_TIME = Histogram(
    'prediction_time_seconds',
    'Time spent processing prediction'
)
PREDICTION_REQUESTS = Counter(
    'prediction_requests_total',
    'Total number of prediction requests'
)
CACHE_HITS = Counter(
    'cache_hits_total',
    'Total number of cache hits'
)
MODEL_INFO = Info('model_info', 'Model information')

@contextmanager
def timeout(seconds):
    """Context manager for timeout handling"""
    def signal_handler(signum, frame):
        raise TimeoutError("Prediction timed out")
    
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
class Config:
    """Configuration management"""
    def __init__(self):
        # Load environment variables
        load_dotenv()
        
        # API Settings
        self.API_HOST = os.getenv('API_HOST', '0.0.0.0')
        self.API_PORT = int(os.getenv('API_PORT', 8000))
        self.DEBUG_MODE = os.getenv('DEBUG_MODE', 'True').lower() == 'true'
        
        # Redis Settings
        self.REDIS_HOST = os.getenv('REDIS_HOST', 'redis')
        self.REDIS_PORT = int(os.getenv('REDIS_PORT', 6379))
        
        # Model Settings
        self.MODEL_PATH = Path(os.getenv('MODEL_PATH', 'dataset/models/latest_model.h5'))
        self.MODEL_VERSION = os.getenv('MODEL_VERSION', '1.0.0')
        
        # Rebrickable API Settings
        self.REBRICKABLE_API_KEY = os.getenv('REBRICKABLE_API_KEY')
        self.REBRICKABLE_API_URL = os.getenv('REBRICKABLE_API_URL', 'https://rebrickable.com/api/v3')
        
        # Training Parameters
        self.BATCH_SIZE = int(os.getenv('BATCH_SIZE', 32))
        self.EPOCHS = int(os.getenv('EPOCHS', 50))
        self.LEARNING_RATE = float(os.getenv('LEARNING_RATE', 0.001))
        self.IMAGE_SIZE = int(os.getenv('IMAGE_SIZE', 224))
        
        # Data Collection
        self.MAX_IMAGES_PER_FIGURE = int(os.getenv('MAX_IMAGES_PER_FIGURE', 10))
        self.DOWNLOAD_THREADS = int(os.getenv('DOWNLOAD_THREADS', 4))
        
        self.validate_config()

    def validate_config(self):
        """Validate configuration settings"""
        if not self.REBRICKABLE_API_KEY:
            raise ValueError("REBRICKABLE_API_KEY must be set")
        
        if not self.MODEL_PATH.parent.exists():
            self.MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        
        if not (0.0 < self.LEARNING_RATE < 1.0):
            raise ValueError("LEARNING_RATE must be between 0 and 1")
        
        if not (16 <= self.IMAGE_SIZE <= 1024):
            raise ValueError("IMAGE_SIZE must be between 16 and 1024")

    def get_model_path(self) -> Path:
        """Get the current model path"""
        if self.MODEL_PATH.exists():
            return self.MODEL_PATH
        
        # Find latest model if specified path doesn't exist
        model_dir = self.MODEL_PATH.parent
        model_files = list(model_dir.glob('*.h5'))
        if not model_files:
            raise FileNotFoundError("No model files found")
        return max(model_files, key=lambda x: x.stat().st_mtime)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            'api': {
                'host': self.API_HOST,
                'port': self.API_PORT,
                'debug': self.DEBUG_MODE
            },
            'redis': {
                'host': self.REDIS_HOST,
                'port': self.REDIS_PORT
            },
            'model': {
                'path': str(self.MODEL_PATH),
                'version': self.MODEL_VERSION
            },
            'training': {
                'batch_size': self.BATCH_SIZE,
                'epochs': self.EPOCHS,
                'learning_rate': self.LEARNING_RATE,
                'image_size': self.IMAGE_SIZE
            },
            'data_collection': {
                'max_images_per_figure': self.MAX_IMAGES_PER_FIGURE,
                'download_threads': self.DOWNLOAD_THREADS
            }
        }
class ModelCache:
    """Model caching and version management"""
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.current_model = None
        self.model_info = {}
        self.load_queue = queue.Queue()
        self.start_monitoring()

    def start_monitoring(self):
        """Start background thread for model monitoring"""
        def monitor():
            while True:
                try:
                    # Check for new models every 60 seconds
                    self._check_new_models()
                    time.sleep(60)
                except Exception as e:
                    logging.error(f"Model monitoring error: {e}")

        thread = threading.Thread(target=monitor, daemon=True)
        thread.start()

    def _check_new_models(self):
        """Check for new model versions"""
        try:
            model_files = list(self.cache_dir.glob('*_best.h5'))
            if not model_files:
                return

            latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
            latest_info = self._get_model_info(latest_model)

            if (not self.current_model or 
                latest_info['version'] > self.model_info.get('version', 0)):
                self.load_queue.put(latest_model)
        except Exception as e:
            logging.error(f"Error checking new models: {e}")

    def _get_model_info(self, model_path: Path) -> Dict:
        """Get model information"""
        try:
            version = int(model_path.stem.split('_')[0])
            return {
                'version': version,
                'path': str(model_path),
                'size': model_path.stat().st_size,
                'modified': datetime.fromtimestamp(model_path.stat().st_mtime)
            }
        except Exception as e:
            logging.error(f"Error getting model info: {e}")
            return {}

class GPUManager:
    """GPU resource management"""
    def __init__(self):
        self.gpu_available = tf.test.is_gpu_available()
        if self.gpu_available:
            self.setup_gpu()

    def setup_gpu(self):
        """Configure GPU settings"""
        try:
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logging.info(f"GPU configuration complete. Found {len(gpus)} GPUs")
        except Exception as e:
            logging.error(f"GPU setup error: {e}")
            self.gpu_available = False

    def get_gpu_usage(self) -> Dict:
        """Get GPU usage statistics"""
        try:
            if not self.gpu_available:
                return {'available': False}

            gpu_stats = []
            for gpu in GPUtil.getGPUs():
                gpu_stats.append({
                    'id': gpu.id,
                    'memory_used': gpu.memoryUsed,
                    'memory_total': gpu.memoryTotal,
                    'temperature': gpu.temperature,
                    'load': gpu.load
                })
            return {
                'available': True,
                'gpus': gpu_stats
            }
        except Exception as e:
            logging.error(f"Error getting GPU stats: {e}")
            return {'available': False, 'error': str(e)}

class MinifigureModelServer:
    def __init__(self):
        # Load configuration
        self.config = Config()
        
        # Setup logging
        self.setup_logging()
        
        # Setup paths
        self.base_dir = Path('dataset')
        self.models_dir = self.base_dir / 'models'
        self.labels_dir = self.base_dir / 'labels'
        self.cache_dir = self.base_dir / 'cache'
        
        # Create necessary directories
        for directory in [self.models_dir, self.labels_dir, self.cache_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        # Initialize GPU management
        self.gpu_manager = GPUManager()
        
        # Initialize metrics
        self.metrics = ModelMetrics()
        
        # Initialize Redis
        self.setup_redis()
        
        # Load model and encoders
        self.load_model()
        self.load_encoders()
        
        # Initialize API
        self.app = self.create_api()

    def setup_logging(self):
        """Setup logging with configuration"""
        log_level = logging.DEBUG if self.config.DEBUG_MODE else logging.INFO
        
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f"logs/server_{datetime.now().strftime('%Y%m%d')}.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        if self.config.DEBUG_MODE:
            self.logger.debug("Debug mode enabled")

    def setup_redis(self):
        """Setup Redis connection"""
        try:
            self.redis_client = redis.Redis(
                host=self.config.REDIS_HOST,
                port=self.config.REDIS_PORT,
                db=0,
                decode_responses=True,
                socket_timeout=5,
                retry_on_timeout=True
            )
            self.redis_client.ping()
            self.logger.info("Redis connection established")
        except Exception as e:
            self.logger.error(f"Redis connection failed: {e}")
            self.redis_client = None

    def load_model(self):
        """Load the trained model"""
        try:
            model_path = Path(self.config.MODEL_PATH)
            if not model_path.exists():
                raise FileNotFoundError(f"Model not found at {model_path}")
            
            self.logger.info(f"Loading model: {model_path}")
            self.model = tf.keras.models.load_model(model_path)
            
            # Optimize for inference
            self.model.make_predict_function()
            
            # Verify model
            test_input = np.zeros((1, self.config.IMAGE_SIZE, self.config.IMAGE_SIZE, 3))
            _ = self.model.predict(test_input, verbose=0)
            
            self.logger.info("Model loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            raise

    def load_encoders(self):
        """Load label encoders"""
        try:
            encoder_path = self.labels_dir / 'encoders_mapping.yaml'
            with open(encoder_path, 'r') as f:
                self.encoders_mapping = yaml.safe_load(f)
            
            self.year_mapping = {v: k for k, v in self.encoders_mapping['year_mapping'].items()}
            self.theme_mapping = {v: k for k, v in self.encoders_mapping['theme_mapping'].items()}
            
            self.logger.info("Encoders loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Error loading encoders: {e}")
            raise

    def preprocess_image(self, image_data: bytes) -> np.ndarray:
        """Preprocess image for prediction"""
        try:
            # Open image
            img = Image.open(io.BytesIO(image_data))
            
            # Convert to RGB if necessary
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Resize
            img = img.resize((self.config.IMAGE_SIZE, self.config.IMAGE_SIZE), 
                           Image.Resampling.LANCZOS)
            
            # Convert to array and normalize
            img_array = np.array(img) / 255.0
            
            # Add batch dimension
            img_array = np.expand_dims(img_array, axis=0)
            
            return img_array
            
        except Exception as e:
            self.logger.error(f"Error preprocessing image: {e}")
            raise

    def predict(self, image_data: bytes, confidence_threshold: float = 0.5) -> Dict:
        """Make predictions for an image"""
        try:
            # Generate cache key
            cache_key = f"pred_{hashlib.sha256(image_data).hexdigest()}"
            
            # Check cache
            if self.redis_client:
                cached_result = self.redis_client.get(cache_key)
                if cached_result:
                    return json.loads(cached_result)
            
            # Preprocess image
            processed_image = self.preprocess_image(image_data)
            
            # Make prediction
            year_pred, theme_pred = self.model.predict(processed_image, verbose=0)
            
            # Filter by confidence
            year_indices = np.where(year_pred[0] >= confidence_threshold)[0]
            theme_indices = np.where(theme_pred[0] >= confidence_threshold)[0]
            
            # Sort by confidence
            year_indices = year_indices[np.argsort(year_pred[0][year_indices])[::-1]]
            theme_indices = theme_indices[np.argsort(theme_pred[0][theme_indices])[::-1]]
            
            # Format predictions
            predictions = {
                'years': [
                    {
                        'year': str(self.year_mapping[idx]),
                        'confidence': float(year_pred[0][idx])
                    }
                    for idx in year_indices
                ],
                'themes': [
                    {
                        'theme': str(self.theme_mapping[idx]),
                        'confidence': float(theme_pred[0][idx])
                    }
                    for idx in theme_indices
                ],
                'timestamp': datetime.now().isoformat(),
                'model_version': self.config.MODEL_VERSION
            }
            
            # Cache result
            if self.redis_client:
                self.redis_client.setex(cache_key, 300, json.dumps(predictions))
            
            return predictions
            
        except Exception as e:
            self.logger.error(f"Error making prediction: {e}")
            raise

    def create_api(self) -> FastAPI:
        """Create FastAPI application"""
        app = FastAPI(
            title="Minifigure Classifier API",
            description="API for classifying LEGO minifigures",
            version=self.config.MODEL_VERSION
        )
        
        # Add CORS middleware
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        @app.post("/predict")
        async def predict_endpoint(file: UploadFile = File(...)):
            try:
                contents = await file.read()
                predictions = self.predict(contents)
                return JSONResponse(content=predictions)
            except Exception as e:
                self.logger.error(f"Prediction error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.post("/predict/batch")
        async def predict_batch_endpoint(files: List[UploadFile] = File(...)):
            try:
                results = []
                for file in files:
                    contents = await file.read()
                    predictions = self.predict(contents)
                    results.append({
                        'filename': file.filename,
                        'predictions': predictions
                    })
                return JSONResponse(content={'results': results})
            except Exception as e:
                self.logger.error(f"Batch prediction error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.get("/health")
        async def health_check():
            return {
                "status": "healthy",
                "model_version": self.config.MODEL_VERSION,
                "gpu_available": self.gpu_manager.gpu_available,
                "redis_connected": bool(self.redis_client)
            }
        
        return app

    def run_server(self):
        """Run the API server"""
        try:
            uvicorn.run(
                self.app,
                host=self.config.API_HOST,
                port=self.config.API_PORT,
                log_level="debug" if self.config.DEBUG_MODE else "info"
            )
        except Exception as e:
            self.logger.error(f"Server error: {e}")
            raise

def main():
    server = MinifigureModelServer()
    server.run_server()

if __name__ == "__main__":
    main()