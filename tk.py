import customtkinter as ctk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, UnidentifiedImageError
import torch
import torchvision
import torchvision.transforms as transforms

model = torchvision.models.resnet18()
model.fc = torch.nn.Linear(in_features=512, out_features=3)
model.load_state_dict(torch.load("src/covid_classifier.pt", map_location=torch.device('cpu')))
model.eval()

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def load_and_classify_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.png")])
    if not file_path:
        return
    
    try:
        
        img = Image.open(file_path)
        if img.mode != 'RGB':
            img = img.convert("RGB")
        
        img_resized = img.resize((400, 400))
        img_display = ImageTk.PhotoImage(img_resized)
        image_label.configure(image=img_display, text="")
        image_label.image = img_display
        update_status("Image loaded successfully!")

        img_tensor = transform(img).unsqueeze(0)
        with torch.no_grad():
            output = model(img_tensor)
            _, predicted = torch.max(output.data, 1)

        result = "COVID-19 Positive" if predicted.item() == 2 else "COVID-19 Negative"
        result_label.configure(text=result, text_color="green" if "Negative" in result else "red")
        update_status("Classification completed!")

    except UnidentifiedImageError:
        messagebox.showerror("Error", "Invalid image file. Please select a valid image.")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")

def update_status(message):
    status_label.configure(text=message)

ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")

window = ctk.CTk()
window.title("COVID-19 Image Classifier")
window.geometry("700x700")

load_button = ctk.CTkButton(window, text="Load and Classify Image", command=load_and_classify_image, width=200)
image_label = ctk.CTkLabel(window, text="No Image Loaded", width=400, height=400, fg_color="gray", corner_radius=10)
result_label = ctk.CTkLabel(window, text="Result will be displayed here.", font=("Arial", 18))
status_label = ctk.CTkLabel(window, text="Status: Waiting for input...", font=("Arial", 14), text_color="blue")

load_button.pack(pady=20)
image_label.pack(pady=20)
result_label.pack(pady=20)
status_label.pack(pady=10)

window.mainloop()
