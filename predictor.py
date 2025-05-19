import torch
import torchvision
import numpy as np
from PIL import Image

# Define the class names
class_names = ['normal', 'viral', 'covid']

# Load the trained model and set it to evaluation mode
def load_model(model_path):
    model = torchvision.models.resnet18()
    model.fc = torch.nn.Linear(in_features=512, out_features=3)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

# Define the image transformations (same as used during training)
test_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize(size=(224, 224)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Prediction function
def predict_image_class(model, image_path):
    image = Image.open(image_path).convert('RGB')
    image = test_transform(image)
    image = image.unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        output = model(image)[0]
        probabilities = torch.nn.functional.softmax(output, dim=0)
        probabilities = probabilities.cpu().detach().numpy()
        predicted_class_index = np.argmax(probabilities)
        predicted_class_name = class_names[predicted_class_index]
    
    return probabilities, predicted_class_index, predicted_class_name

# Load model and make a prediction
if __name__ == "__main__":
    # Path to the .pt file and the input X-ray image
    model_path = '''C:\\Users\\senga\\OneDrive\\Desktop\\MinorProject\\COVID-19_Radiography_Dataset''' # Replace with your model file path
    image_path = 'src/viral.png'       # Replace with your X-ray image file path

    # Load model and predict
    model = load_model(model_path)
    probabilities, predicted_class_index, predicted_class_name = predict_image_class(model, image_path)

    # Output results
    print('Probabilities:', probabilities)
    print('Predicted class index:', predicted_class_index)
    print('Predicted class name:', predicted_class_name)
