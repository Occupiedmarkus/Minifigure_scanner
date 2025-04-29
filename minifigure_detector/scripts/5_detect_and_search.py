import os
import yaml
import logging
import numpy as np
from pathlib import Path
from PIL import Image
import tensorflow as tf
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from typing import List, Dict
import io
import time
from datetime import datetime
import redis
import jwt
from pydantic import BaseModel

class PredictionRequest(BaseModel):
    image: str  # Base64 encoded image
    confidence_threshold: float = 0.5

class MinifigureModelServer:
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
        self.models_dir = self.base_dir / 'models'
        self.labels_dir = self.base_dir / 'labels'
        
        # Initialize Redis for caching
        self.redis_client = redis.Redis(
            host=os.getenv('REDIS_HOST', 'localhost'),
            port=int(os.getenv('REDIS_PORT', 6379)),
            db=0,
            decode_responses=True
        )
        
        # Load model and encoders
        self.load_model()
        self.load_encoders()
        
        # Initialize API
        self.app = self.create_api()

    def load_model(self):
        """Load the latest trained model"""
        try:
            # Load model with TF-Serving optimization
            model_files = list(self.models_dir.glob('*_best.h5'))
            if not model_files:
                raise FileNotFoundError("No trained models found")
            
            latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
            self.logger.info(f"Loading model: {latest_model.name}")
            
            # Load model with optimization
            self.model = tf.keras.models.load_model(latest_model)
            
            # Convert model for TF-Serving
            self.model_version = int(time.time())
            tf.saved_model.save(
                self.model,
                str(self.models_dir / f'serving/{self.model_version}')
            )
            
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            raise

    def load_encoders(self):
        """Load label encoders mapping"""
        try:
            with open(self.labels_dir / 'encoders_mapping.yaml', 'r') as f:
                self.encoders_mapping = yaml.safe_load(f)
            
            self.year_mapping = {v: k for k, v in self.encoders_mapping['year_mapping'].items()}
            self.theme_mapping = {v: k for k, v in self.encoders_mapping['theme_mapping'].items()}
            
        except Exception as e:
            self.logger.error(f"Error loading encoders: {e}")
            raise

    def preprocess_image(self, image_data: bytes) -> np.ndarray:
        """Preprocess image for model input"""
        try:
            # Open image from bytes
            img = Image.open(io.BytesIO(image_data))
            
            # Convert to RGB if necessary
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Resize to match model input
            img = img.resize((224, 224), Image.Resampling.LANCZOS)
            
            # Convert to numpy array and normalize
            img_array = np.array(img) / 255.0
            
            # Add batch dimension
            img_array = np.expand_dims(img_array, axis=0)
            
            return img_array
            
        except Exception as e:
            self.logger.error(f"Error preprocessing image: {e}")
            raise

    def predict(self, image_data: bytes, confidence_threshold: float = 0.5) -> Dict:
        """Make predictions for a single image"""
        try:
            # Generate cache key
            cache_key = f"pred_{hash(image_data)}_{confidence_threshold}"
            
            # Check cache
            cached_result = self.redis_client.get(cache_key)
            if cached_result:
                return yaml.safe_load(cached_result)
            
            # Preprocess image
            processed_image = self.preprocess_image(image_data)
            
            # Get predictions
            year_pred, theme_pred = self.model.predict(processed_image)
            
            # Filter predictions by confidence threshold
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
                'timestamp': datetime.now().isoformat()
            }
            
            # Cache result
            self.redis_client.setex(
                cache_key,
                300,  # Cache for 5 minutes
                yaml.dump(predictions)
            )
            
            return predictions
            
        except Exception as e:
            self.logger.error(f"Error making prediction: {e}")
            raise

    def create_api(self) -> FastAPI:
        """Create FastAPI application"""
        app = FastAPI(
            title="Minifigure Classifier API",
            description="API for classifying LEGO minifigures",
            version="1.0.0"
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
            if request.url.path in ["/docs", "/redoc", "/openapi.json"]:
                return await call_next(request)
            
            auth_header = request.headers.get("Authorization")
            if not auth_header:
                raise HTTPException(status_code=401, detail="Missing authentication token")
            
            try:
                token = auth_header.split(" ")[1]
                jwt.decode(token, os.getenv("JWT_SECRET"), algorithms=["HS256"])
            except Exception:
                raise HTTPException(status_code=401, detail="Invalid authentication token")
            
            return await call_next(request)
        
        @app.post("/predict")
        async def predict_endpoint(file: UploadFile = File(...)):
            try:
                contents = await file.read()
                predictions = self.predict(contents)
                return JSONResponse(content=predictions)
            except Exception as e:
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
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.get("/health")
        async def health_check():
            return {"status": "healthy", "model_version": self.model_version}
        
        return app

    def run_server(self, host: str = "0.0.0.0", port: int = 8000):
        """Run the API server"""
        uvicorn.run(self.app, host=host, port=port)

def main():
    # Create server instance
    server = MinifigureModelServer()
    
    # Run server
    server.run_server()

if __name__ == "__main__":
    main()