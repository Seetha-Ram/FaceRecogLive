import streamlit as st
import cv2
import tempfile
import os
import torch
from torchvision import transforms
from facenet_pytorch import MTCNN, InceptionResnetV1
import numpy as np
from PIL import Image
import time

# Load models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Function to detect faces and calculate embeddings
def detect_faces(image):
    # Convert image to PIL format
    image_pil = Image.fromarray(image)

    # Detect faces in the current frame
    boxes, _ = mtcnn.detect(image_pil)

    if boxes is not None and len(boxes) > 0:
        box = boxes[0]
        face = image_pil.crop((box[0], box[1], box[2], box[3])).resize((160, 160))
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        face_tensor = transform(face).unsqueeze(0).to(device)
        embedding = resnet(face_tensor).detach().cpu().numpy()[0]
        return embedding
    else:
        return None

# Main Streamlit app
def main():
    st.title("Face Recognition Demo")

    # Upload an image
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_image is not None:
        # Read the uploaded image
        image = np.array(Image.open(uploaded_image))

        # Detect faces in the uploaded image
        embedding_image = detect_faces(image)

        if embedding_image is not None:
            st.image(image, caption="Uploaded Image", use_column_width=True)

            st.write("Now, capturing 1-minute live stream from camera...")

            # Open camera stream
            cap = cv2.VideoCapture(0)

            # Get current time
            start_time = time.time()

            while time.time() - start_time < 60:  # Capture for 1 minute
                ret, frame = cap.read()
                if not ret:
                    break

                # Detect faces in the current frame
                embedding_video = detect_faces(frame)

                # Calculate the cosine similarity between the embeddings
                if embedding_video is not None:
                    similarity = np.dot(embedding_image, embedding_video) / (
                                np.linalg.norm(embedding_image) * np.linalg.norm(embedding_video))
                    similarity_percentage = round(similarity * 100, 2)

                    # Display result
                    if similarity_percentage >= 70:
                        st.write(f"Matching: Yes with {similarity_percentage}% similarity")
                    else:
                        st.write(f"Matching: No with {similarity_percentage}% similarity")

                # Display the current frame
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                st.image(frame, channels="RGB", use_column_width=True)

            # Release video capture
            cap.release()
        else:
            st.write("No faces detected in the uploaded image.")

if __name__ == "__main__":
    main()
