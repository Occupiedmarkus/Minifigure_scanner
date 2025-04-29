import tkinter as tk
from tkinter import filedialog, ttk
import cv2
import numpy as np
from PIL import Image, ImageTk
import tensorflow as tf
import os

class MinifigureGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Minifigure Detector")
        
        # Layout
        self.setup_ui()
        
        # Model loading
        self.model = self.load_model()
        
    def setup_ui(self):
        # Main frame
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Image preview
        self.preview_label = ttk.Label(self.main_frame, text="No image selected")
        self.preview_label.grid(row=0, column=0, columnspan=2, pady=10)
        
        # Buttons
        ttk.Button(self.main_frame, text="Load Image", 
                  command=self.load_image).grid(row=1, column=0, pady=5)
        ttk.Button(self.main_frame, text="Predict", 
                  command=self.predict).grid(row=1, column=1, pady=5)
        
        # Results
        self.result_label = ttk.Label(self.main_frame, text="")
        self.result_label.grid(row=2, column=0, columnspan=2, pady=10)
        
    def load_model(self):
        try:
            model_path = "dataset/models/latest_model.h5"
            return tf.keras.models.load_model(model_path)
        except:
            print("No model found. Please train the model first.")
            return None
            
    def load_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if file_path:
            # Load and display image
            image = Image.open(file_path)
            image = image.resize((224, 224))  # Resize for display
            photo = ImageTk.PhotoImage(image)
            self.preview_label.configure(image=photo)
            self.preview_label.image = photo
            self.current_image_path = file_path
            
    def predict(self):
        if not hasattr(self, 'current_image_path'):
            self.result_label.configure(text="Please load an image first")
            return
            
        if self.model is None:
            self.result_label.configure(text="Model not loaded")
            return
            
        # Load and preprocess image
        img = cv2.imread(self.current_image_path)
        img = cv2.resize(img, (224, 224))
        img = img / 255.0
        img = np.expand_dims(img, axis=0)
        
        # Predict
        prediction = self.model.predict(img)
        # Convert prediction to class name (implement based on your classes)
        result = f"Predicted: {prediction}"
        self.result_label.configure(text=result)

def main():
    root = tk.Tk()
    app = MinifigureGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()