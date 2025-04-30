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
from typing import List, Dict, Optional, Union
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
        # Load environment variables
        load_dotenv()
        
        # Setup logging with rotating file handler
        self.setup_logging()
        
        # Initialize paths and configurations
        self.setup_paths()
        
        # Initialize GPU management
        self.gpu_manager = GPUManager()
        
        # Initialize metrics
        self.metrics = ModelMetrics()
        
        # Initialize Redis with error handling and connection pooling
        self.setup_redis()
        
        # Initialize model cache and versioning
        self.model_cache = ModelCache(self.models_dir)
        
        # Load model and encoders
        self.load_model()
        self.load_encoders()
        
        # Setup batch processing queue
        self.batch_queue = queue.Queue()
        self.start_batch_processor()
        
        # Initialize API
        self.app = self.create_api()

    def setup_logging(self):
        """Setup enhanced logging system"""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        # Create rotating file handler
        from logging.handlers import RotatingFileHandler
        
        log_file = log_dir / f"model_server_{datetime.now().strftime('%Y%m%d')}.log"
        handler = RotatingFileHandler(
            log_file,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        
        # Setup formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        
        # Setup logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(handler)
        
        # Add console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

    def setup_paths(self):
        """Setup directory structure and paths"""
        self.base_dir = Path(os.getenv('DATASET_PATH', 'dataset'))
        self.models_dir = self.base_dir / 'models'
        self.labels_dir = self.base_dir / 'labels'
        self.cache_dir = self.base_dir / 'cache'
        self.temp_dir = self.base_dir / 'temp'
        
        # Create directories
        for directory in [self.models_dir, self.labels_dir, 
                         self.cache_dir, self.temp_dir]:
            directory.mkdir(parents=True, exist_ok=True)

    def setup_redis(self):
        """Setup Redis with connection pooling and error handling"""
        try:
            # Create connection pool
            self.redis_pool = redis.ConnectionPool(
                host=os.getenv('REDIS_HOST', 'localhost'),
                port=int(os.getenv('REDIS_PORT', 6379)),
                db=0,
                decode_responses=True,
                max_connections=10,
                socket_timeout=5,
                socket_connect_timeout=5,
                retry_on_timeout=True
            )
            
            # Create Redis client
            self.redis_client = redis.Redis(
                connection_pool=self.redis_pool
            )
            
            # Test connection
            self.redis_client.ping()
            self.logger.info("Redis connection established")
            
        except redis.RedisError as e:
            self.logger.error(f"Redis connection error: {e}")
            self.logger.warning("Running without Redis cache")
            self.redis_client = None

    def load_model(self):
        """Load model with version control and optimization"""
        try:
            model_files = list(self.models_dir.glob('*_best.h5'))
            if not model_files:
                raise ModelError("No trained models found")
            
            # Get latest model version
            latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
            self.logger.info(f"Loading model: {latest_model.name}")
            
            # Load model with memory optimization
            tf.keras.backend.clear_session()
            
            with PREDICTION_TIME.time():
                self.model = tf.keras.models.load_model(latest_model)
                
                # Optimize model for inference
                self.model.make_predict_function()
                
                if self.gpu_manager.gpu_available:
                    # Move model to GPU if available
                    with tf.device('/GPU:0'):
                        self.model.predict(np.zeros((1, 224, 224, 3)))
            
            # Save model info
            self.model_info = {
                'version': latest_model.stem.split('_')[0],
                'path': str(latest_model),
                'loaded_at': datetime.now().isoformat(),
                'input_shape': self.model.input_shape,
                'output_shape': self.model.output_shape,
                'gpu_available': self.gpu_manager.gpu_available
            }
            
            # Update Prometheus metrics
            MODEL_INFO.info({
                'version': self.model_info['version'],
                'loaded_at': self.model_info['loaded_at'],
                'gpu_available': str(self.model_info['gpu_available'])
            })
            
            self.logger.info(f"Model loaded successfully: {self.model_info}")
            
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            raise ModelError(f"Failed to load model: {str(e)}")

    def load_encoders(self):
        """Load and validate label encoders"""
        try:
            encoder_path = self.labels_dir / 'encoders_mapping.yaml'
            if not encoder_path.exists():
                raise FileNotFoundError("Encoder mapping file not found")
            
            with open(encoder_path, 'r') as f:
                self.encoders_mapping = yaml.safe_load(f)
            
            # Validate encoders
            required_keys = ['year_mapping', 'theme_mapping']
            if not all(key in self.encoders_mapping for key in required_keys):
                raise ValueError("Invalid encoder mapping format")
            
            # Create reverse mappings
            self.year_mapping = {v: k for k, v in self.encoders_mapping['year_mapping'].items()}
            self.theme_mapping = {v: k for k, v in self.encoders_mapping['theme_mapping'].items()}
            
            self.logger.info("Encoders loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Error loading encoders: {e}")
            raise ModelError(f"Failed to load encoders: {str(e)}")

    def start_batch_processor(self):
        """Start background batch processing thread"""
        def process_batches():
            while True:
                try:
                    batch = []
                    batch_files = []
                    
                    # Collect batch
                    while len(batch) < 32:  # Max batch size
                        try:
                            item = self.batch_queue.get(timeout=1)
                            batch.append(item['image'])
                            batch_files.append(item['file'])
                        except queue.Empty:
                            break
                    
                    if batch:
                        # Process batch
                        batch_array = np.vstack(batch)
                        predictions = self.model.predict(batch_array)
                        
                        # Save results
                        for i, file in enumerate(batch_files):
                            result = self.format_predictions(predictions[i])
                            file['result_queue'].put(result)
                            
                except Exception as e:
                    self.logger.error(f"Batch processing error: {e}")
                    time.sleep(1)
        
        thread = threading.Thread(target=process_batches, daemon=True)
        thread.start()
    def preprocess_image(self, image_data: bytes) -> np.ndarray:
        """Preprocess image with enhanced error handling and validation"""
        try:
            # Validate input
            if not image_data:
                raise PreprocessError("Empty image data")
            
            # Open image with validation
            try:
                img = Image.open(io.BytesIO(image_data))
            except Exception as e:
                raise PreprocessError(f"Invalid image format: {e}")
            
            # Check image mode and convert if necessary
            if img.mode not in ['RGB', 'RGBA']:
                raise PreprocessError(f"Unsupported image mode: {img.mode}")
            
            if img.mode == 'RGBA':
                img = img.convert('RGB')
            
            # Validate image dimensions
            if img.size[0] < 32 or img.size[1] < 32:
                raise PreprocessError("Image too small")
            
            # Resize with high-quality settings
            img = img.resize((224, 224), Image.Resampling.LANCZOS)
            
            # Convert to numpy array with validation
            try:
                img_array = np.array(img, dtype=np.float32)
            except Exception as e:
                raise PreprocessError(f"Array conversion failed: {e}")
            
            # Normalize
            img_array = img_array / 255.0
            
            # Validate array shape and values
            if img_array.shape != (224, 224, 3):
                raise PreprocessError(f"Invalid array shape: {img_array.shape}")
            
            if not (0 <= img_array.min() <= img_array.max() <= 1.0):
                raise PreprocessError("Invalid pixel values after normalization")
            
            # Add batch dimension
            return np.expand_dims(img_array, axis=0)
            
        except PreprocessError as e:
            raise e
        except Exception as e:
            raise PreprocessError(f"Preprocessing failed: {str(e)}")

    def predict(self, image_data: bytes, confidence_threshold: float = 0.5) -> Dict:
        """Make predictions with enhanced error handling and monitoring"""
        prediction_start = time.time()
        cache_hit = False
        
        try:
            # Update metrics
            PREDICTION_REQUESTS.inc()
            
            # Generate secure cache key
            cache_key = f"pred_{hashlib.sha256(image_data).hexdigest()}_{confidence_threshold}"
            
            # Try cache first
            if self.redis_client:
                try:
                    cached_result = self.redis_client.get(cache_key)
                    if cached_result:
                        CACHE_HITS.inc()
                        cache_hit = True
                        self.metrics.cache_hits += 1
                        return json.loads(cached_result)
                except redis.RedisError as e:
                    self.logger.warning(f"Redis error: {e}")
            
            # Preprocess image
            with PREDICTION_TIME.time():
                processed_image = self.preprocess_image(image_data)
            
            # Make prediction with timeout
            with timeout(30):  # 30 seconds timeout
                if self.gpu_manager.gpu_available:
                    with tf.device('/GPU:0'):
                        predictions = self.model.predict(processed_image, verbose=0)
                else:
                    predictions = self.model.predict(processed_image, verbose=0)
            
            # Unpack predictions
            year_pred, theme_pred = predictions
            
            # Filter by confidence
            year_indices = np.where(year_pred[0] >= confidence_threshold)[0]
            theme_indices = np.where(theme_pred[0] >= confidence_threshold)[0]
            
            # Sort by confidence
            year_indices = year_indices[np.argsort(year_pred[0][year_indices])[::-1]]
            theme_indices = theme_indices[np.argsort(theme_pred[0][theme_indices])[::-1]]
            
            # Format predictions
            result = {
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
                'model_version': self.model_info['version'],
                'timestamp': datetime.now().isoformat(),
                'processing_time': time.time() - prediction_start,
                'cache_hit': cache_hit
            }
            
            # Cache result
            if self.redis_client:
                try:
                    self.redis_client.setex(
                        cache_key,
                        300,  # 5 minutes cache
                        json.dumps(result)
                    )
                except redis.RedisError as e:
                    self.logger.warning(f"Redis caching error: {e}")
            
            # Update metrics
            self.metrics.total_requests += 1
            self.metrics.successful_requests += 1
            self.metrics.average_latency = (
                (self.metrics.average_latency * (self.metrics.total_requests - 1) +
                 (time.time() - prediction_start)) / self.metrics.total_requests
            )
            
            return result
            
        except Exception as e:
            self.metrics.failed_requests += 1
            self.logger.error(f"Prediction error: {e}")
            raise

    def create_api(self) -> FastAPI:
        """Create FastAPI application with enhanced features"""
        app = FastAPI(
            title="Minifigure Classifier API",
            description="Advanced API for classifying LEGO minifigures",
            version="2.0.0",
            docs_url="/docs",
            redoc_url="/redoc"
        )
        
        # Add CORS middleware
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Authentication middleware
        @app.middleware("http")
        async def authenticate(request, call_next):
            if request.url.path in ["/docs", "/redoc", "/openapi.json", "/metrics", "/health"]:
                return await call_next(request)
            
            auth_header = request.headers.get("Authorization")
            if not auth_header:
                raise HTTPException(
                    status_code=401,
                    detail="Missing authentication token"
                )
            
            try:
                token = auth_header.split(" ")[1]
                jwt.decode(
                    token,
                    os.getenv("JWT_SECRET"),
                    algorithms=["HS256"]
                )
            except Exception:
                raise HTTPException(
                    status_code=401,
                    detail="Invalid authentication token"
                )
            
            return await call_next(request)
        
        # Prediction endpoint
        @app.post("/predict", response_model=PredictionResponse)
        async def predict_endpoint(
            file: UploadFile = File(...),
            confidence_threshold: float = 0.5,
            background_tasks: BackgroundTasks = None
        ):
            try:
                contents = await file.read()
                predictions = self.predict(contents, confidence_threshold)
                
                # Add background task for cleanup
                if background_tasks:
                    background_tasks.add_task(self.cleanup_temp_files)
                
                return predictions
            except Exception as e:
                self.logger.error(f"Prediction endpoint error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        # Batch prediction endpoint
        @app.post("/predict/batch")
        async def predict_batch_endpoint(
            files: List[UploadFile] = File(...),
            confidence_threshold: float = 0.5
        ):
            try:
                results = []
                result_queues = []
                
                # Process files in parallel
                for file in files:
                    contents = await file.read()
                    result_queue = queue.Queue()
                    result_queues.append(result_queue)
                    
                    self.batch_queue.put({
                        'image': self.preprocess_image(contents),
                        'file': {
                            'name': file.filename,
                            'result_queue': result_queue
                        }
                    })
                
                # Collect results
                for queue in result_queues:
                    try:
                        result = queue.get(timeout=30)
                        results.append(result)
                    except queue.Empty:
                        results.append({'error': 'Processing timeout'})
                
                return JSONResponse(content={'results': results})
                
            except Exception as e:
                self.logger.error(f"Batch prediction error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        # Health check endpoint
        @app.get("/health")
        async def health_check():
            """Enhanced health check endpoint"""
            try:
                health_status = {
                    "status": "healthy",
                    "timestamp": datetime.now().isoformat(),
                    "model": {
                        "version": self.model_info['version'],
                        "loaded_at": self.model_info['loaded_at'],
                        "input_shape": str(self.model_info['input_shape']),
                        "gpu_available": self.gpu_manager.gpu_available
                    },
                    "system": {
                        "cpu_usage": psutil.cpu_percent(),
                        "memory_usage": psutil.virtual_memory().percent,
                        "disk_usage": psutil.disk_usage('/').percent
                    },
                    "metrics": {
                        "total_requests": self.metrics.total_requests,
                        "successful_requests": self.metrics.successful_requests,
                        "failed_requests": self.metrics.failed_requests,
                        "average_latency": self.metrics.average_latency,
                        "cache_hits": self.metrics.cache_hits
                    }
                }

                # Add GPU stats if available
                if self.gpu_manager.gpu_available:
                    health_status["gpu"] = self.gpu_manager.get_gpu_usage()

                # Check Redis connection
                if self.redis_client:
                    try:
                        self.redis_client.ping()
                        health_status["redis"] = "connected"
                    except redis.RedisError:
                        health_status["redis"] = "disconnected"
                else:
                    health_status["redis"] = "disabled"

                # Perform quick model sanity check
                test_input = np.zeros((1, 224, 224, 3))
                _ = self.model.predict(test_input, verbose=0)
                health_status["model"]["sanity_check"] = "passed"

                return health_status

            except Exception as e:
                self.logger.error(f"Health check failed: {e}")
                return {
                    "status": "unhealthy",
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }

        # Metrics endpoint
        @app.get("/metrics")
        async def metrics():
            """Prometheus metrics endpoint"""
            return generate_latest()

        # Model information endpoint
        @app.get("/model/info")
        async def model_info():
            """Get detailed model information"""
            return {
                "model_info": self.model_info,
                "metrics": self.metrics.dict(),
                "gpu_info": self.gpu_manager.get_gpu_usage() if self.gpu_manager.gpu_available else None
            }

        # Cache management endpoints
        @app.post("/cache/clear")
        async def clear_cache():
            """Clear Redis cache"""
            try:
                if self.redis_client:
                    self.redis_client.flushdb()
                    return {"status": "success", "message": "Cache cleared"}
                return {"status": "warning", "message": "Redis not enabled"}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        return app

    def cleanup_temp_files(self):
        """Cleanup temporary files"""
        try:
            for file in self.temp_dir.glob("*"):
                if time.time() - file.stat().st_mtime > 3600:  # 1 hour old
                    file.unlink()
        except Exception as e:
            self.logger.error(f"Cleanup error: {e}")

    def run_server(self, host: str = "0.0.0.0", port: int = 8000):
        """Run the API server with enhanced configuration"""
        try:
            # Configure uvicorn logging
            log_config = uvicorn.config.LOGGING_CONFIG
            log_config["formatters"]["access"]["fmt"] = "%(asctime)s - %(levelname)s - %(message)s"
            
            # Start server with optimized settings
            config = uvicorn.Config(
                app=self.app,
                host=host,
                port=port,
                workers=4,
                loop="uvloop",
                log_config=log_config,
                limit_concurrency=100,
                limit_max_requests=10000,
                timeout_keep_alive=30,
                access_log=True
            )
            server = uvicorn.Server(config)
            
            # Start monitoring
            self.start_monitoring()
            
            # Run server
            server.run()
            
        except Exception as e:
            self.logger.error(f"Server startup error: {e}")
            raise

    def start_monitoring(self):
        """Start background monitoring"""
        def monitor_resources():
            while True:
                try:
                    # Update system metrics
                    self.metrics.memory_usage = psutil.virtual_memory().percent
                    
                    if self.gpu_manager.gpu_available:
                        gpu_stats = self.gpu_manager.get_gpu_usage()
                        if gpu_stats['available']:
                            self.metrics.gpu_usage = gpu_stats['gpus'][0]['load']
                    
                    # Log metrics if thresholds exceeded
                    if self.metrics.memory_usage > 90:
                        self.logger.warning(f"High memory usage: {self.metrics.memory_usage}%")
                    
                    if self.metrics.gpu_usage > 90:
                        self.logger.warning(f"High GPU usage: {self.metrics.gpu_usage}%")
                    
                    time.sleep(60)  # Update every minute
                    
                except Exception as e:
                    self.logger.error(f"Monitoring error: {e}")
                    time.sleep(60)

        # Start monitoring thread
        thread = threading.Thread(target=monitor_resources, daemon=True)
        thread.start()

def main():
    """Main function with enhanced error handling and setup"""
    try:
        # Set environment variables if not set
        if not os.getenv('DATASET_PATH'):
            os.environ['DATASET_PATH'] = str(Path.cwd() / 'dataset')
        
        if not os.getenv('JWT_SECRET'):
            os.environ['JWT_SECRET'] = os.urandom(32).hex()
            print("Warning: Generated temporary JWT_SECRET")
        
        # Create server instance
        server = MinifigureModelServer()
        
        # Get port from environment or default
        port = int(os.getenv('PORT', 8000))
        
        # Run server
        server.run_server(port=port)
        
    except Exception as e:
        logging.error(f"Startup error: {e}")
        raise

if __name__ == "__main__":
    main()
       