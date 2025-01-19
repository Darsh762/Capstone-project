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
video_path = (r"C:\\Users\\c104\\Desktop\\v1\\RIGHT_2.mp4")  # Specify the path to your video file
cap = cv2.VideoCapture(video_path)

# Check if the video was opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Create a folder to save frames
output_folder = 'output_frames'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

def calculate_angle(x1, y1, x2, y2, x3, y3):
    """Calculate the angle between two lines."""
    angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
    return abs(angle) if angle >= 0 else 360 + angle

# Define the connections for right shoulder, elbow, wrist, and hip
connections = [
    (mpPose.PoseLandmark.RIGHT_SHOULDER, mpPose.PoseLandmark.RIGHT_ELBOW),
    (mpPose.PoseLandmark.RIGHT_ELBOW, mpPose.PoseLandmark.RIGHT_WRIST),
    (mpPose.PoseLandmark.RIGHT_SHOULDER, mpPose.PoseLandmark.RIGHT_HIP)
]

pTime = 0
frame_count = 0
collecting_frames = False  # Flag to indicate whether to start collecting frames

while True:
    ret, frame = cap.read()

    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)

    frame_count += 1  # Increment frame count at each iteration

    # Check if the frame count is within the specified range
    if 170 <= frame_count <= 210 and results.pose_landmarks:
        # Get positions of right shoulder, elbow, wrist, and hip
        landmarks = {}
        for idx in [mpPose.PoseLandmark.RIGHT_SHOULDER, mpPose.PoseLandmark.RIGHT_ELBOW, 
                    mpPose.PoseLandmark.RIGHT_WRIST, mpPose.PoseLandmark.RIGHT_HIP]:
            lm = results.pose_landmarks.landmark[idx]
            h, w, c = frame.shape
            cx, cy = int(lm.x * w), int(lm.y * h)
            landmarks[idx] = (cx, cy)
            cv2.circle(frame, (cx, cy), 5, (255, 0, 0), cv2.FILLED)

        # Draw connections
        for start, end in connections:
            if start in landmarks and end in landmarks:
                cv2.line(frame, landmarks[start], landmarks[end], (0, 255, 0), 2)

        if mpPose.PoseLandmark.RIGHT_ELBOW in landmarks and mpPose.PoseLandmark.RIGHT_HIP in landmarks:
            elbow_x, elbow_y = landmarks[mpPose.PoseLandmark.RIGHT_ELBOW]
            shoulder_x, shoulder_y = landmarks[mpPose.PoseLandmark.RIGHT_SHOULDER]
            hip_x, hip_y = landmarks[mpPose.PoseLandmark.RIGHT_HIP]

            # Draw the reference red line from shoulder to hip
            cv2.line(frame, (shoulder_x, shoulder_y), (hip_x, hip_y), (0, 0, 255), 2)

            # Calculate the angle between shoulder-elbow line and shoulder-hip line
            angle = calculate_angle(
                shoulder_x, shoulder_y,
                elbow_x, elbow_y,
                hip_x, hip_y
            )
            cv2.putText(frame, f'Angle: {int(angle)}', (elbow_x - 50, elbow_y - 20), 
                        cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

            # Start collecting frames
            collecting_frames = True

            # Collect frames
            if collecting_frames:
                cv2.putText(frame, f'Frame: {frame_count}', (30, 70), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 0), 2)

                # Save the frame
                frame_filename = os.path.join(output_folder, f'frame_{frame_count}.jpg')
                cv2.imwrite(frame_filename, frame)

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
