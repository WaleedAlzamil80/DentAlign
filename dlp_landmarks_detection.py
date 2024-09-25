from __future__ import division
import dlib
import cv2
import numpy as np

# Function to resize the image while maintaining aspect ratio
def resize(img, width=None, height=None, interpolation=cv2.INTER_AREA):
    h, w = img.shape[:2]
    if width is None and height is None:
        return img
    elif width is None:
        ratio = height / h
        width = int(w * ratio)
    else:
        ratio = width / w
        height = int(h * ratio)
    resized = cv2.resize(img, (width, height), interpolation)
    return resized, ratio

# Convert the dlib shape object to a NumPy array
def shape_to_np(shape, dtype="int"):
    coords = np.zeros((68, 2), dtype=dtype)
    for i in range(68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords

# Path to the shape predictor model
predictor_path = "/home/waleed/Documents/3DLearning/DDS/68_points/shape_predictor_68_face_landmarks.dat"

# Initialize dlib's face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

# --- Part 1: Real-time detection using a webcam ---
def real_time_landmarks():
    camera = cv2.VideoCapture(0)

    while True:
        ret, frame = camera.read()
        if not ret:
            print("Failed to capture frame from camera.")
            break

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_resized, ratio = resize(frame_gray, width=120)

        # Detect faces in the resized frame
        faces = detector(frame_resized, 1)
        if len(faces) > 0:
            for face in faces:
                shape = predictor(frame_resized, face)
                shape_np = shape_to_np(shape)

                # Draw landmarks on the original frame
                for (x, y) in shape_np:
                    cv2.circle(frame, (int(x / ratio), int(y / ratio)), 3, (255, 255, 255), -1)

        cv2.imshow("Real-time Facial Landmarks", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.release()
    cv2.destroyAllWindows()

# --- Part 2: Facial landmarks on a static image ---
def detect_landmarks_on_image(image_path):
    # Load the image and convert to grayscale
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale image
    faces = detector(gray)

    # Loop over each detected face
    for face in faces:
        # Get the landmarks for the face
        landmarks = predictor(gray, face)

        # Draw the landmarks on the image
        for n in range(68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            cv2.circle(img, (x, y), 2, (0, 255, 0), -1)

    # Display the output image with landmarks
    cv2.imshow('Landmarks', img)
    cv2.waitKey(5000)
    cv2.destroyAllWindows()

# Run the real-time landmark detection
# Uncomment to use real-time detection:
# real_time_landmarks()

# Run the static image landmark detection
# Example usage for an image file:
image_path = "/home/waleed/Documents/3DLearning/DDS/DentAlign/dataset/LS3D-W-balanced-20-03-2017/new_dataset/7153.jpg"
detect_landmarks_on_image(image_path)