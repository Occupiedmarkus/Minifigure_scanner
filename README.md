# 🏗️ Minifigure Detector
A machine learning system for detecting and classifying LEGO® minifigures using **computer vision** and **deep learning**.

## 🎯 Features
✅ Automated data collection from LEGO databases  
✅ Image preprocessing and augmentation  
✅ Deep learning model for minifigure classification  
✅ FastAPI-based prediction service  
✅ Interactive GUI for custom predictions and training  
✅ Docker support for easy deployment  

---

## 📂 Project Structure
```
minifigure_detector/
├── scripts/                    
│   ├── 1_collect_data.py      # Data collection script
│   ├── 2_preprocess_data.py   # Image preprocessing
│   ├── 3_train_model.py       # Model training
│   ├── 4_evaluate_model.py    # Model evaluation
│   ├── 5_deploy_model.py      # API service
│   └── 6_predict_and_train.py # GUI tool
├── dataset/                    
│   ├── images/                # Raw images
│   ├── processed/             # Processed images
│   └── models/                # Trained models
├── Dockerfile                 # Docker configuration
├── docker-compose.yml         # Docker services
└── requirements.txt           # Python dependencies
```

---

## 🚀 Getting Started

### Prerequisites
- **Python** 3.9+
- **Docker** and Docker Compose
- **NVIDIA GPU** _(optional, for faster training)_

### Installation
1️⃣ **Clone the repository**
```bash
git clone https://github.com/yourusername/minifigure_detector.git
cd minifigure_detector
```
2️⃣ **Build Docker containers**
```bash
docker-compose build
```

---

## 🔄 Usage Pipeline

### 1️⃣ Data Collection  
Scrapes minifigure data and images:  
```bash
docker-compose run collect
```

### 2️⃣ Data Preprocessing  
Prepares images for training:  
```bash
docker-compose run train python scripts/2_preprocess_data.py
```

### 3️⃣ Model Training  
Trains the classification model:  
```bash
docker-compose run train python scripts/3_train_model.py
```

### 4️⃣ Model Evaluation  
Tests model performance:  
```bash
docker-compose run train python scripts/4_evaluate_model.py
```

### 5️⃣ API Service  
Starts the prediction API:  
```bash
docker-compose up api
```
📌 Access the API at **http://localhost:8000**

### 6️⃣ GUI Tool (Local Use)  
Interactive tool for predictions and training:  
```bash
pip install -r requirements.txt
python scripts/6_predict_and_train.py
```

---

## 🔍 Monitoring and Maintenance

### Check Container Status  
```bash
# List running containers
docker-compose ps

# View logs
docker-compose logs

# View specific service logs
docker-compose logs api
```

### Adding New Training Data
Use the **GUI tool** (Script 6) to add new images, then:  
```bash
docker-compose run train python scripts/2_preprocess_data.py
docker-compose run train python scripts/3_train_model.py
docker-compose run train python scripts/4_evaluate_model.py
docker-compose restart api
```

---

## 📊 API Endpoints

### Prediction API  
**POST /predict**  
- Upload an image for prediction  
- Returns **predicted year** and **theme**

### Documentation  
- **`/docs`** - Swagger UI  
- **`/redoc`** - ReDoc  

---

## 🛠️ Development

### Local Development  
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Unix
# or
.\venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Docker Development  
```bash
# Rebuild containers
docker-compose build

# Run specific script
docker-compose run train python scripts/your_script.py
```

---

## 📝 Dependencies
- **TensorFlow** - Deep Learning  
- **FastAPI** - API Service  
- **OpenCV** - Image Processing  
- **Redis** - Caching  
🔹 Additional dependencies in `requirements.txt`

---

## 🤝 Contributing  
🔹 Fork the repository  
🔹 Create a feature branch  
🔹 Commit changes  
🔹 Push to the branch  
🔹 Create a **Pull Request**

---

## 📄 License
🚨 **This project is NOT licensed. It is NOT free to use.**  

---

## 🙏 Acknowledgments
- **LEGO®** is a trademark of the LEGO Group  
- Thanks to the **open-source community** for valuable contributions  

---

### ✅ Your README is now **structured, clear, and visually appealing!**  
Let me know if you want **additional sections or modifications!** 🚀  
