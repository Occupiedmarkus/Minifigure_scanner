Minifigure Detector
A machine learning system for detecting and classifying LEGO minifigures using computer vision and deep learning.

🎯 Features
Automated data collection from LEGO databases
Image preprocessing and augmentation
Deep learning model for minifigure classification
FastAPI-based prediction service
Interactive GUI for custom predictions and training
Docker support for easy deployment
🏗️ Project Structure
plaintext

Copy
minifigure_detector/
├── scripts/
│   ├── 1_collect_data.py      # Data collection script
│   ├── 2_preprocess_data.py   # Image preprocessing
│   ├── 3_train_model.py       # Model training
│   ├── 4_evaluate_model.py    # Model evaluation
│   ├── 5_deploy_model.py      # API service
│   └── 6_predict_and_train.py # GUI tool
├── dataset/                    # Data storage
│   ├── images/                # Raw images
│   ├── processed/             # Processed images
│   └── models/                # Trained models
├── Dockerfile                 # Docker configuration
├── docker-compose.yml         # Docker services
└── requirements.txt           # Python dependencies
🚀 Getting Started
Prerequisites
Python 3.9+
Docker and Docker Compose
NVIDIA GPU (optional, for faster training)
Installation
Clone the repository:
bash

Copy
git clone https://github.com/yourusername/minifigure_detector.git
cd minifigure_detector
Build Docker containers:
bash

Copy
docker-compose build
🔄 Usage Pipeline
1. Data Collection
Scrapes minifigure data and images:

bash

Copy
docker-compose run collect
2. Data Preprocessing
Prepares images for training:

bash

Copy
docker-compose run train python scripts/2_preprocess_data.py
3. Model Training
Trains the classification model:

bash

Copy
docker-compose run train python scripts/3_train_model.py
4. Model Evaluation
Tests model performance:

bash

Copy
docker-compose run train python scripts/4_evaluate_model.py
5. API Service
Starts the prediction API:

bash

Copy
docker-compose up api
Access the API at http://localhost:8000

6. GUI Tool (Local Use)
Interactive tool for predictions and training:

bash

Copy
# Install local dependencies
pip install -r requirements.txt

# Run GUI
python scripts/6_predict_and_train.py
🔍 Monitoring and Maintenance
Check Container Status
bash

Copy
# List running containers
docker-compose ps

# View logs
docker-compose logs

# View specific service logs
docker-compose logs api
Adding New Training Data
Use the GUI tool (Script 6) to add new images
Reprocess the dataset:
bash

Copy
docker-compose run train python scripts/2_preprocess_data.py
Retrain the model:
bash

Copy
docker-compose run train python scripts/3_train_model.py
Evaluate new model:
bash

Copy
docker-compose run train python scripts/4_evaluate_model.py
Restart the API:
bash

Copy
docker-compose restart api
📊 API Endpoints
Prediction API
POST /predict
Upload an image for prediction
Returns predicted year and theme
Documentation
/docs - Swagger UI documentation
/redoc - ReDoc documentation
🛠️ Development
Local Development
For GUI development or testing:

bash

Copy
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Unix
# or
.\venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
Docker Development
bash

Copy
# Rebuild containers
docker-compose build

# Run specific script
docker-compose run train python scripts/your_script.py

📝 Dependencies
TensorFlow for deep learning
FastAPI for API service
OpenCV for image processing
Redis for caching
Additional dependencies in requirements.txt

🤝 Contributing
Fork the repository
Create a feature branch
Commit changes
Push to the branch
Create a Pull Request

📄 License
This project is not licensed. NOT FREE to use.
🙏 Acknowledgments
LEGO® is a trademark of the LEGO Group
Thanks to the open-source community
Would you like me to:

Add more technical details about the model?
Include troubleshooting guidelines?
Add deployment instructions?
Include example API usage?
