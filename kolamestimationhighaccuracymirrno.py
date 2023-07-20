import cv2
import mediapipe as mp
import pandas as pd

# Function to calculate the Intersection over Union (IoU)
def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    intersection_area = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    boxA_area = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxB_area = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    iou = intersection_area / float(boxA_area + boxB_area - intersection_area)
    return iou

# Main function for tracking index finger and saving the coordinates to CSV
def track_index_finger():
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands

    # Initialize MediaPipe hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    # Video capture object using the computer's webcam (change to 0 if you have only one camera)
    cap = cv2.VideoCapture(0)

    # CSV file to save the coordinates
    csv_filename = 'finger_coordinates.csv'

    # List to store coordinates
    finger_coordinates = []

    while cap.isOpened():
        ret, image = cap.read()
        if not ret:
            break

        # Convert the image from BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process the image with MediaPipe hands
        results = hands.process(image_rgb)

        # Clear previous finger coordinates
        finger_coordinates.clear()

        if results.multi_hand_landmarks:
            # Get the bounding box of the hand (for better finger tracking accuracy)
            hand_landmarks = results.multi_hand_landmarks[0]
            x_min, y_min, x_max, y_max = 1, 1, 0, 0
            for landmark in hand_landmarks.landmark:
                x, y = int(landmark.x * image.shape[1]), int(landmark.y * image.shape[0])
                x_min, y_min = min(x_min, x), min(y_min, y)
                x_max, y_max = max(x_max, x), max(y_max, y)

            # Calculate the hand bounding box
            hand_box = (x_min, y_min, x_max, y_max)

            # Get the landmark points for the index finger (Landmark #8)
            index_finger_landmarks = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            x, y = int(index_finger_landmarks.x * image.shape[1]), int(index_finger_landmarks.y * image.shape[0])
            finger_coordinates.append((x, y))

            # Draw the hand bounding box
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

        # Draw landmarks on the image
        mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Display the image
        cv2.imshow('MediaPipe Hands', image)

        # Write the finger coordinates to the CSV file
        if finger_coordinates:
            df = pd.DataFrame(finger_coordinates, columns=['x', 'y'])
            df.to_csv(csv_filename, mode='a', index=False, header=False)

        # Exit the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    track_index_finger()

