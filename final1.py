import cv2
import mediapipe as mp
import time
import os
import math

# Suppress TensorFlow Lite logs (optional)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Initialize mediapipe pose solution
mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose = mpPose.Pose()

# Load the video
cap = cv2.VideoCapture(r"D:\\Datasets\\RIGHT_9.mp4")  # Specify the path to your video file

# Check if the video was opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Define the red line position (constant Y)
red_line_y = 150
frame_count = 0

# Create a folder to save frames
output_folder = 'output_frames'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

def calculate_angle(x1, y1, x2, y2, x3, y3):
    """Calculate the angle between two lines: static line and elbow-wrist line."""
    angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
    return abs(angle) if angle >= 0 else 360 + angle

pTime = 0

while True:
    ret, frame = cap.read()
    
    # Check if frame is read correctly
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Convert the frame to RGB for MediaPipe
    imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)

    # List of landmarks (right elbow and wrist)
    if results.pose_landmarks:
        right_elbow = mpPose.PoseLandmark.RIGHT_ELBOW
        right_wrist = mpPose.PoseLandmark.RIGHT_WRIST

        # Extract positions of the right elbow and wrist
        landmarks = {}
        for idx in [right_elbow, right_wrist]:
            lm = results.pose_landmarks.landmark[idx]
            h, w, c = frame.shape
            cx, cy = int(lm.x * w), int(lm.y * h)
            landmarks[idx] = (cx, cy)
            # Draw a circle on the landmark
            cv2.circle(frame, (cx, cy), 5, (255, 0, 0), cv2.FILLED)

        # Get the positions for the elbow and wrist
        if right_elbow in landmarks and right_wrist in landmarks:
            elbow_x, elbow_y = landmarks[right_elbow]
            wrist_x, wrist_y = landmarks[right_wrist]

            # Draw line between elbow and wrist on the frame
            cv2.line(frame, (elbow_x, elbow_y), (wrist_x, wrist_y), (255, 255, 255), 3)  # White line between elbow and wrist

            # Check if the elbow meets the red line
            if abs(elbow_y - red_line_y) < 5:  # Allow for slight deviations
                # Calculate the angle between the static red line and the elbow-wrist line
                angle = calculate_angle(190, red_line_y, elbow_x, elbow_y, wrist_x, wrist_y)
                
                # Display the angle on the frame
                cv2.putText(frame, f'Angle: {int(angle)}', (elbow_x - 50, elbow_y - 20), 
                            cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
                
                # Draw static red line on the frame
                cv2.line(frame, (650, red_line_y), (190, red_line_y), (0, 0, 255), 2)

                # Save the frame with the calculated angle and the lines
                frame_count += 1
                frame_filename = os.path.join(output_folder, f'frame_{frame_count}.jpg')
                cv2.imwrite(frame_filename, frame)

            # Stop the calculation when the wrist goes below the red line
            if wrist_y > red_line_y:
                print("Wrist went below the red line. Exiting ...")
                break

    # Draw the static red line on the frame
    cv2.line(frame, (650, red_line_y), (190, red_line_y), (0, 0, 255), 2)

    # Calculate FPS
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    # Display FPS on the image
    cv2.putText(frame, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    # Display the frame
    cv2.imshow('Frame', frame)
    
    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        break

# Release everything when job is finished
cap.release()
cv2.destroyAllWindows()
