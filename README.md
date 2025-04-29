# First, make sure you're in your project directory
cd your_project

# Install required packages (if you haven't already)
pip install ultralytics opencv-python python-dotenv tqdm pyyaml

# 1. Collect initial training data
python scripts/1_collect_data.py

# 2. Label the collected images
python scripts/2_label_images.py

# 3. Prepare the dataset
python scripts/3_prepare_dataset.py

# 4. Train the model
python scripts/4_train_model.py

# Put your custom images in the custom_imgs directory
your_project/custom_imgs/
├── minifig1.jpg
├── minifig2.jpg
└── ...

# Run detection on your custom images
python scripts/5_detect_and_search.py
# Choose option 1 or 2 from the menu and point to custom_imgs directory