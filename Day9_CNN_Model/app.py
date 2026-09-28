app_py_content = 
import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image


# ==========================================
# 1. CNN MODEL ARCHITECTURE
# ==========================================

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(SimpleCNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),

            # For 128x128 input:
            # 128 -> 64 -> 32
            nn.Linear(32 * 32 * 32, 128),

            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# ==========================================
# 2. CONFIGURATION
# ==========================================

IMAGE_SIZE = (128, 128)

CLASS_NAMES = ['Cat', 'Dog']

MODEL_PATH = 'simple_cnn_model.pth'


# ==========================================
# 3. LOAD TRAINED MODEL
# ==========================================

@st.cache_resource
def load_model():

    model = SimpleCNN(num_classes=2)

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=torch.device('cpu')
        )
    )

    model.eval()

    return model


# ==========================================
# 4. IMAGE TRANSFORMATION
# ==========================================

def get_transforms():

    return transforms.Compose([

        transforms.Resize(IMAGE_SIZE),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
    ])


# ==========================================
# 5. PREDICTION FUNCTION
# ==========================================

def predict(image, model, transform):

    image = image.convert('RGB')

    image_tensor = transform(image)

    image_tensor = image_tensor.unsqueeze(0)

    with torch.no_grad():

        output = model(image_tensor)

        probabilities = torch.softmax(output, dim=1)

        predicted_idx = torch.argmax(output, dim=1).item()

    predicted_class = CLASS_NAMES[predicted_idx]

    confidence = probabilities[0][predicted_idx].item()

    return predicted_class, confidence


# ==========================================
# 6. STREAMLIT APP
# ==========================================

st.title("🐱 Cat vs Dog Image Classifier")

st.write(
    "Upload an image and the CNN model will predict "
    "whether it is a Cat or a Dog."
)


# ==========================================
# 7. IMAGE UPLOAD
# ==========================================

uploaded_file = st.file_uploader(
    "Choose a cat or dog image...",
    type=["jpg", "jpeg", "png"]
)


# ==========================================
# 8. PREDICTION
# ==========================================

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    st.write("Classifying...")

    # Load model
    model = load_model()

    # Get transformations
    transform = get_transforms()

    # Predict
    predicted_class, confidence = predict(
        image,
        model,
        transform
    )

    # Display result
    if predicted_class == "Cat":

        st.success(
            f"🐱 Prediction: Cat | "
            f"Confidence: {confidence:.2%}"
        )

    else:

        st.success(
            f"🐶 Prediction: Dog | "
            f"Confidence: {confidence:.2%}"
        )

