"""
Handwritten Digit Recognition GUI
Draw digits on canvas and get real-time predictions from trained CNN model
"""

import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageDraw
import numpy as np
import tensorflow as tf
from tensorflow import keras
import os

class DigitRecognizerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Handwritten Digit Recognition")
        self.root.geometry("450x900")
        self.root.resizable(True, True)
        
        # Configure root background
        self.root.config(bg="#ecf0f1")
        
        # Load the trained model
        self.model = None
        self.load_model()
        
        # Canvas settings
        self.canvas_size = 400
        self.brush_width = 20
        
        # Create PIL image for drawing
        self.image = Image.new("L", (self.canvas_size, self.canvas_size), color=0)
        self.draw = ImageDraw.Draw(self.image)
        
        # Setup GUI
        self.setup_ui()
        
    def load_model(self):
        """Load the trained CNN model"""
        model_path = "models/mnist_cnn_model.keras"
        alt_model_path = "models/mnist_cnn_model.h5"
        
        try:
            if os.path.exists(model_path):
                self.model = keras.models.load_model(model_path)
                print(f"Model loaded successfully from {model_path}")
            elif os.path.exists(alt_model_path):
                self.model = keras.models.load_model(alt_model_path)
                print(f"Model loaded successfully from {alt_model_path}")
            else:
                messagebox.showerror("Error", 
                    "Model not found! Please train the model first using the Jupyter notebook.")
                self.root.destroy()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {str(e)}")
            self.root.destroy()
    
    def setup_ui(self):
        """Setup the user interface"""
        # Create main container with scrollbar
        main_container = tk.Frame(self.root, bg="#ecf0f1")
        main_container.pack(fill=tk.BOTH, expand=True)
        
        # Create canvas with scrollbar for scrolling content
        scroll_canvas = tk.Canvas(main_container, bg="#ecf0f1", highlightthickness=0)
        scrollbar = tk.Scrollbar(main_container, orient="vertical", command=scroll_canvas.yview)
        scrollable_frame = tk.Frame(scroll_canvas, bg="#ecf0f1")
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: scroll_canvas.configure(scrollregion=scroll_canvas.bbox("all"))
        )
        
        scroll_canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        scroll_canvas.configure(yscrollcommand=scrollbar.set)
        
        scroll_canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Enable mouse wheel scrolling
        def _on_mousewheel(event):
            scroll_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        scroll_canvas.bind_all("<MouseWheel>", _on_mousewheel)
        
        # Title
        title_frame = tk.Frame(scrollable_frame, bg="#2c3e50", height=70)
        title_frame.pack(fill=tk.X)
        title_frame.pack_propagate(False)
        
        title_label = tk.Label(title_frame, 
                              text="✍️ Digit Recognition",
                              font=("Arial", 20, "bold"),
                              bg="#2c3e50",
                              fg="white")
        title_label.pack(pady=18)
        
        # Instructions
        instruction_frame = tk.Frame(scrollable_frame, bg="#ecf0f1")
        instruction_frame.pack(fill=tk.X, pady=5)
        
        instruction_label = tk.Label(instruction_frame,
                                    text="Draw a digit (0-9) on the canvas below",
                                    font=("Arial", 11),
                                    bg="#ecf0f1",
                                    fg="#34495e")
        instruction_label.pack(pady=3)
        
        # Canvas frame with border
        canvas_container = tk.Frame(scrollable_frame, bg="#34495e", padx=2, pady=2)
        canvas_container.pack(pady=8)
        
        canvas_frame = tk.Frame(canvas_container, bg="#ecf0f1", padx=5, pady=5)
        canvas_frame.pack()
        
        # Drawing canvas
        self.canvas = tk.Canvas(canvas_frame,
                               width=self.canvas_size,
                               height=self.canvas_size,
                               bg="black",
                               cursor="cross",
                               highlightthickness=2,
                               highlightbackground="#3498db")
        self.canvas.pack()
        
        # Bind mouse events
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        
        # Buttons - placed right after canvas
        button_frame = tk.Frame(scrollable_frame, bg="#ecf0f1")
        button_frame.pack(pady=8)
        
        self.predict_btn = tk.Button(button_frame,
                                     text="🔍 Predict Digit",
                                     command=self.predict_digit,
                                     font=("Arial", 11, "bold"),
                                     bg="#3498db",
                                     fg="white",
                                     width=16,
                                     height=2,
                                     cursor="hand2",
                                     relief=tk.RAISED,
                                     borderwidth=2)
        self.predict_btn.grid(row=0, column=0, padx=8)
        
        self.clear_btn = tk.Button(button_frame,
                                   text="�️ Clear Canvas",
                                   command=self.clear_canvas,
                                   font=("Arial", 11, "bold"),
                                   bg="#e74c3c",
                                   fg="white",
                                   width=16,
                                   height=2,
                                   cursor="hand2",
                                   relief=tk.RAISED,
                                   borderwidth=2)
        self.clear_btn.grid(row=0, column=1, padx=8)
        
        # Prediction display
        result_frame = tk.Frame(scrollable_frame, bg="#ecf0f1")
        result_frame.pack(pady=8)
        
        self.result_label = tk.Label(result_frame,
                                     text="Prediction: -",
                                     font=("Arial", 18, "bold"),
                                     bg="#ecf0f1",
                                     fg="#27ae60")
        self.result_label.pack(pady=3)
        
        self.confidence_label = tk.Label(result_frame,
                                        text="Confidence: -",
                                        font=("Arial", 12),
                                        bg="#ecf0f1",
                                        fg="#7f8c8d")
        self.confidence_label.pack()
        
        # Separator
        separator = tk.Frame(scrollable_frame, bg="#bdc3c7", height=2)
        separator.pack(fill=tk.X, padx=50, pady=8)
        
        # Probability bars frame
        self.prob_frame = tk.Frame(scrollable_frame, bg="#ecf0f1")
        self.prob_frame.pack(pady=5, padx=20)
    
    def paint(self, event):
        """Draw on canvas"""
        x1, y1 = (event.x - self.brush_width), (event.y - self.brush_width)
        x2, y2 = (event.x + self.brush_width), (event.y + self.brush_width)
        
        # Draw on tkinter canvas
        self.canvas.create_oval(x1, y1, x2, y2, 
                               fill="white", 
                               outline="white",
                               width=self.brush_width)
        
        # Draw on PIL image
        self.draw.ellipse([x1, y1, x2, y2], fill=255)
    
    def on_release(self, event):
        """Auto-predict when mouse is released (optional)"""
        # Auto-prediction disabled - use Predict button instead
        pass
    
    def clear_canvas(self):
        """Clear the canvas and reset predictions"""
        self.canvas.delete("all")
        self.image = Image.new("L", (self.canvas_size, self.canvas_size), color=0)
        self.draw = ImageDraw.Draw(self.image)
        
        self.result_label.config(text="Prediction: -")
        self.confidence_label.config(text="Confidence: -")
        
        # Clear probability bars
        for widget in self.prob_frame.winfo_children():
            widget.destroy()
    
    def preprocess_image(self):
        """Preprocess the drawn image for model prediction"""
        # Resize to 28x28 (MNIST size)
        img_resized = self.image.resize((28, 28), Image.Resampling.LANCZOS)
        
        # Convert to numpy array
        img_array = np.array(img_resized)
        
        # Normalize pixel values to [0, 1]
        img_array = img_array.astype('float32') / 255.0
        
        # Reshape for model input (1, 28, 28, 1)
        img_array = img_array.reshape(1, 28, 28, 1)
        
        return img_array, img_resized
    
    def predict_digit(self):
        """Predict the drawn digit"""
        if self.model is None:
            messagebox.showerror("Error", "Model not loaded!")
            return
        
        # Check if canvas is empty
        if not self.canvas.find_all():
            messagebox.showwarning("Warning", "Please draw a digit first!")
            return
        
        try:
            # Preprocess image
            processed_img, resized_img = self.preprocess_image()
            
            # Make prediction
            predictions = self.model.predict(processed_img, verbose=0)
            predicted_digit = np.argmax(predictions[0])
            confidence = predictions[0][predicted_digit] * 100
            
            # Update result labels
            self.result_label.config(text=f"Prediction: {predicted_digit}")
            self.confidence_label.config(text=f"Confidence: {confidence:.2f}%")
            
            # Display probability bars
            self.display_probabilities(predictions[0])
            
        except Exception as e:
            messagebox.showerror("Error", f"Prediction failed: {str(e)}")
    
    def display_probabilities(self, probabilities):
        """Display probability bars for all digits"""
        # Clear previous probability bars
        for widget in self.prob_frame.winfo_children():
            widget.destroy()
        
        # Create title
        title = tk.Label(self.prob_frame,
                        text="Probability Distribution:",
                        font=("Arial", 11, "bold"),
                        bg="#ecf0f1",
                        fg="#34495e")
        title.grid(row=0, column=0, columnspan=3, pady=(5, 10))
        
        # Create probability bars
        max_prob = np.max(probabilities)
        for digit in range(10):
            prob = probabilities[digit] * 100
            
            # Label
            label = tk.Label(self.prob_frame,
                           text=f"{digit}:",
                           font=("Arial", 10, "bold"),
                           bg="#ecf0f1",
                           width=3)
            label.grid(row=digit+1, column=0, sticky='e', padx=5, pady=2)
            
            # Progress bar - highlight the max probability
            if prob == max_prob * 100:
                progress = ttk.Progressbar(self.prob_frame,
                                          length=300,
                                          mode='determinate',
                                          style='Highlight.Horizontal.TProgressbar')
            else:
                progress = ttk.Progressbar(self.prob_frame,
                                          length=300,
                                          mode='determinate',
                                          style='Custom.Horizontal.TProgressbar')
            progress['value'] = prob
            progress.grid(row=digit+1, column=1, pady=2, padx=5)
            
            # Percentage label
            pct_color = "#27ae60" if prob == max_prob * 100 else "#7f8c8d"
            pct_label = tk.Label(self.prob_frame,
                               text=f"{prob:.1f}%",
                               font=("Arial", 9, "bold" if prob == max_prob * 100 else "normal"),
                               bg="#ecf0f1",
                               fg=pct_color,
                               width=7)
            pct_label.grid(row=digit+1, column=2, padx=5, pady=2)


def main():
    """Main function to run the application"""
    root = tk.Tk()
    
    # Configure style for progress bars
    style = ttk.Style()
    style.theme_use('default')
    
    # Regular progress bar style
    style.configure('Custom.Horizontal.TProgressbar',
                   background='#95a5a6',
                   troughcolor='#ecf0f1',
                   bordercolor='#bdc3c7',
                   lightcolor='#95a5a6',
                   darkcolor='#7f8c8d',
                   thickness=20)
    
    # Highlighted progress bar style for max probability
    style.configure('Highlight.Horizontal.TProgressbar',
                   background='#27ae60',
                   troughcolor='#ecf0f1',
                   bordercolor='#229954',
                   lightcolor='#27ae60',
                   darkcolor='#1e8449',
                   thickness=20)
    
    app = DigitRecognizerGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
