import torch
import streamlit as st 
import numpy
import pandas
import torchvision.transforms as transforms
from PIL import Image


from torchvision.models.densenet import densenet201

# Find the correct key for the model in the checkpoint dictionary
# Once you identify the correct key name, replace 'model_key' with that name

import requests

model_url = 'https://drive.google.com/file/d/1-XPxv9g1D-k9J_Qb7HJ7mWrln6Q-vfmi/view?usp=sharing'  
output_file = 'epochcheckpoint9.pth'

response = requests.get(model_url)
with open(output_file, 'wb') as f:
    f.write(response.content)

checkpoint_path = 'epochcheckpoint9.pth'
checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))

# Create an instance of the Densenet-201 model
model = densenet201(pretrained=False, num_classes=39)  # Set the number of output classes to 39

# Load the state_dict to the model
model.load_state_dict(checkpoint['model_state_dict'])

model.eval()  # Set the model to evaluation mode

# Define the class labels (if available)
class_labels = ['Apple__Apple_scab', 'Apple_Black_rot', 'Apple_Cedar_apple_rust', 'Apple_healthy', 'Background_without_leaves', 'Cherry_Powdery_mildew', 'Corn_Cercospora_leaf_spot Gray_leaf_spot', 'Corn_Common_rust', 'Blueberry_healthy', 'Cherry_healthy', 'Grape_healthy', 'Grape_Leaf_blight(Isariopsis_Leaf_Spot)', 'Orange__Haunglongbing(Citrus_greening)', 'Peach__Bacterial_spot', 'Corn_healthy', 'Corn_Northern_Leaf_Blight', 'Grape_Black_rot', 'Grape_Esca(Black_Measles)', 'Potato__Late_blight', 'Potato_healthy', 'Raspberry_healthy', 'Peach_healthy', 'Pepper,_bell_healthy', 'Potato_Early_blight', 'Pepper,_bell_Bacterial_spot', 'Soybean_healthy', 'Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_healthy', 'Strawberry_healthy', 'Squash_Powdery_mildew', 'Strawberry_Leaf_scorch', 'Tomato_Tomato_mosaic_virus', 'Tomato_Spider_mites Two-spotted_spider_mite', 'Tomato_Target_Spot', 'Tomato_Tomato_Yellow_Leaf_Curl_Virus', 'Tomato_Late_blight', 'Tomato_Leaf_Mold', 'Tomato__Septoria_leaf_spot']

# Function to perform inference on the input image
def predict_disease(image):
    # Preprocess the image
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    image = transform(image).unsqueeze(0)

    # Perform inference
    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.softmax(outputs, dim=1)

        # Get the top predicted class and its probability
        _, predicted_idx = torch.max(probabilities, 1)
        predicted_label = class_labels[predicted_idx.item()]
        prediction_percentage = probabilities[0, predicted_idx].item() * 100

    return predicted_label, prediction_percentage

# Create the Streamlit app
def main():
    # Set the CSS style to make the background green
    green_bg = """
    <style>
    body {
        background-color: #00ff00;
    }
    </style>
    """
    st.markdown(green_bg, unsafe_allow_html=True)

    st.title("Plant Leaf Disease Detection")


    # Add an option to upload an image or use the camera
    image_option = st.radio("Select Image Source:", ("Upload Image", "Use Camera"))

    if image_option == "Upload Image":
        # Option to upload an image
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            # Display the uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)

            # Perform prediction on the uploaded image
            predicted_label = predict_disease(image)
            predicted_label, prediction_percentage = predict_disease(image)
            st.write(f"Predicted Disease: {predicted_label}")
            st.write(f"Prediction Percentage: {prediction_percentage:.2f}%")

            # Additional information for "Apple__Apple_scab"
            if predicted_label == "Apple__Apple_scab":
                st.markdown("### Apple Scab")
                st.markdown("Apple scab is a plant disease that affects apple trees. It is caused by the fungus "
                            "*Venturia inaequalis*. The disease primarily targets leaves and fruit, leading to "
                            "significant economic losses in apple production.")

                st.markdown("### Causes of Apple Scab")
                st.markdown("The primary causes of apple scab include:")
                st.markdown("- Warm and wet weather conditions favor the growth and spread of the fungus.")
                st.markdown("- Presence of infected apple debris from previous seasons.")
                st.markdown("- Lack of proper sanitation and hygiene practices in orchards.")
                st.markdown("- Planting susceptible apple tree varieties.")

                st.markdown("### What to Do")
                st.markdown("To manage and prevent apple scab, the following measures can be taken:")
                st.markdown("- Prune infected plant parts and dispose of them properly.")
                st.markdown("- Apply fungicides following recommended schedules.")
                st.markdown("- Improve orchard sanitation by removing fallen leaves and fruit.")
                st.markdown("- Plant resistant apple tree varieties.")
                st.markdown("- Monitor weather conditions and adjust spraying accordingly.")

    elif image_option == "Use Camera":
        # Option to use the camera for real-time prediction
        st.write("Using Camera...")

        # Use OpenCV to capture video from the camera
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            st.error("Error: Unable to access the camera.")
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                st.error("Error: Unable to capture frame.")
                break

            # Convert the frame to PIL image format
            frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            # Perform prediction on the captured frame
            predicted_label = predict_disease(frame_pil)

            # Display the frame with the prediction
            st.image(frame, caption=f"Predicted Disease: {predicted_label}", use_column_width=True)

            # Stop capturing when the 'Stop' button is pressed
            if st.button("Stop"):
                cap.release()
                break

    st.write("Thank you for using the Plant Leaf Disease Detection app!")

if __name__ == "__main__":
    main()

