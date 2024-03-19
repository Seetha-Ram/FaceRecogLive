import streamlit as st
import cv2
import tempfile
import os
import torch
from torchvision import transforms
from facenet_pytorch import MTCNN, InceptionResnetV1
import numpy as np

def main():
    st.title("Face Recognition Demo")

    # Load models
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mtcnn = MTCNN(device=device)
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_image is not None:
        # Read the uploaded image
        image = cv2.imdecode(np.frombuffer(uploaded_image.read(), np.uint8), 1)

        # Detect faces in the uploaded image
        boxes, _ = mtcnn.detect(image)

        if boxes is not None and len(boxes) > 0:
            box = boxes[0]
            face = image[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((160, 160)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])
            face_tensor = transform(face).unsqueeze(0).to(device)
            embedding_image = resnet(face_tensor).detach().cpu().numpy()[0]

            st.image(image, caption="Uploaded Image", use_column_width=True)

            st.write("Now, showing live stream from camera...")
            
            # Open camera stream
            cap = cv2.VideoCapture(0)
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Detect faces in the current frame
                boxes, _ = mtcnn.detect(frame)

                if boxes is not None and len(boxes) > 0:
                    box = boxes[0]
                    face = frame[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
                    face_tensor = transform(face).unsqueeze(0).to(device)
                    embedding_video = resnet(face_tensor).detach().cpu().numpy()[0]

                    # Calculate the cosine similarity between the embeddings
                    similarity = np.dot(embedding_image, embedding_video) / (
                                np.linalg.norm(embedding_image) * np.linalg.norm(embedding_video))
                    similarity_percentage = round(similarity * 100, 2)

                    # Display result
                    if similarity_percentage >= 70:
                        st.write(f"Matching: Yes with {similarity_percentage}% similarity")
                    else:
                        st.write(f"Matching: No with {similarity_percentage}% similarity")

                # Display the current frame
                cv2.imshow('Face Recognition', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # Release video capture
            cap.release()
            cv2.destroyAllWindows()
        else:
            st.write("No faces detected in the image.")

if __name__ == "__main__":
    main()
