import cv2
import numpy as np
from keras.models import load_model
from tkinter import filedialog
from tkinter import messagebox
from tkinter import ttk
import tkinter as tk
from PIL import Image, ImageTk

class GUI(tk.Tk):
    frame_styles = {"relief": "groove",
                    "bd": 3, "bg": "#BEB2A7",
                    "fg": "#073bb3", "font": ("Arial", 15, "bold")}

    def __init__(self):
        tk.Tk.__init__(self)
        self.title("Handwritten Character Recognition")
        self.geometry("800x600")
        self.configure(bg="#BEB2A7")

        # Load the pre-trained model
        self.model = load_model('D:/Downloads/handwritten-character-recognition-source-code/handwritten_character_recog_model.h5')
        self.words = {0:'A',1:'B',2:'C',3:'D',4:'E',5:'F',6:'G',7:'H',8:'I',9:'J',10:'K',11:'L',12:'M',13:'N',14:'O',15:'P',16:'Q',17:'R',18:'S',19:'T',20:'U',21:'V',22:'W',23:'X', 24:'Y',25:'Z'}

        # Create a frame for the image and prediction display
        self.image_frame = tk.LabelFrame(self, text="Uploaded Image", **self.frame_styles)
        self.image_frame.place(relx=0.05, rely=0.05, relwidth=0.9, relheight=0.6)

        # Create labels for the image and prediction
        self.image_label = tk.Label(self.image_frame)
        self.image_label.pack(side="top", pady=10)

        self.prediction_label = tk.Label(self.image_frame, text="", **self.frame_styles)
        self.prediction_label.pack(side="top", pady=10)

        # Create a button to upload an image
        self.upload_button = ttk.Button(self, text="Upload Image", command=self.upload_image)
        self.upload_button.place(relx=0.4, rely=0.7, relwidth=0.2, relheight=0.1)

        # Create a button to predict the character
        self.predict_button = ttk.Button(self, text="Predict", command=self.predict_character)
        self.predict_button.place(relx=0.4, rely=0.85, relwidth=0.2, relheight=0.1)

        # Initialize image variable
        self.image = None

    def upload_image(self):
        # Open a file dialog to select an image file
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp")])
        if file_path:
            # Read the image using OpenCV
            image = cv2.imread(file_path)
            # Convert image to RGB format
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # Resize image to fit the label
            image = cv2.resize(image, (300, 300))
            # Display the image
            self.show_image(image)

    def show_image(self, image):
        # Convert image to RGB format
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Resize image to fit the label
        image = cv2.resize(image, (300, 300))
        # Convert image to PIL Image
        image_pil = Image.fromarray(image)
        # Convert PIL Image to Tkinter PhotoImage
        photo = ImageTk.PhotoImage(image=image_pil)
        # Update the image label
        self.image_label.config(image=photo)
        self.image_label.image = photo
        # Store the image as a NumPy array
        self.image = image

    def predict_character(self):
        # Check if an image has been uploaded
        if self.image is not None:
            # Preprocess the image for prediction
            image_copy = cv2.GaussianBlur(self.image, (7, 7), 0)
            gray_image = cv2.cvtColor(image_copy, cv2.COLOR_RGB2GRAY)
            _, img_thresh = cv2.threshold(gray_image, 100, 255, cv2.THRESH_BINARY_INV)
            final_image = cv2.resize(img_thresh, (28, 28))
            final_image = np.reshape(final_image, (1, 28, 28, 1))
            # Predict using the model
            prediction = self.words[np.argmax(self.model.predict(final_image))]
            # Display the prediction
            self.prediction_label.config(text="Predicted Letter: " + prediction)
        else:
            messagebox.showwarning("No Image", "Please upload an image first.")

app = GUI()
app.mainloop()
