import torch
from torchvision import transforms
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, messagebox

from model import CNN

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print("Program is starting...")

# Weights Path Configuration
MODEL_PATH = './checkpoints/100.pth'

# Model Initialization
try:
    print("Loading Model Weights...")
    model = CNN(224, 224).to(device)
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict['state_dict'])
    model.eval()
    print("Model initialized")
except Exception as e:
    print(f"Loading failed: {e}")
    exit()

# Data Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# UI Class
class CatDogApp:
    def __init__(self, master):
        self.master = master
        master.title("Cat or Dog Classifier")
        master.geometry("400x500")

        self.label = tk.Label(master, text="Upload a picture of a cat or dog", font=("Arial", 12))
        self.label.pack(pady=20)

        self.button = tk.Button(master, text="Choose Image", command=self.load_image)
        self.button.pack()

        self.result_label = tk.Label(master, text="", font=("Arial", 14))
        self.result_label.pack(pady=20)

        self.image_label = tk.Label(master)
        self.image_label.pack()

    def load_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp")]
        )
        if file_path == '':
            return
        if not file_path or not file_path.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
            messagebox.showerror("Error", "Please select an image file（jpg/jpeg/png/bmp）")
            return

        try:
            image = Image.open(file_path).convert('RGB')
            input_tensor = transform(image)
            input_tensor = input_tensor.to(device).unsqueeze(0)
            with torch.no_grad():
                outputs = model(input_tensor)
                confidence = float(outputs)

            if confidence >= 0.5:
                self.result_label.config(
                    text=f"Prediction: Cat ({confidence * 100:.2f}%)",
                    fg="green"
                )
            else:
                self.result_label.config(
                    text=f"Prediction: Dog ({(1 - confidence) * 100:.2f}%)",
                    fg="green"
                )

            image.thumbnail((250, 250))
            img_tk = ImageTk.PhotoImage(image)
            self.image_label.config(image=img_tk)
            self.image_label.image = img_tk

        except Exception as e:
            messagebox.showerror("Error", f"Image processing failed: {str(e)}")

if __name__ == '__main__':
    root = tk.Tk()
    app = CatDogApp(root)
    root.mainloop()
