import cv2
import os

# Set output directory and create if it doesn't exist
output_dir = 'calibration_images'
os.makedirs(output_dir, exist_ok=True)

# Define chessboard size (number of internal corners per a chessboard row and column)
chessboard_size = (9, 6)
image_count = 0
max_images = 20

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Webcam not accessible.")
    exit()

print("Press 's' to save an image when the checkerboard is clearly visible.")
print("Press 'q' to quit.")

while image_count < max_images:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale and find chessboard corners
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCorners(gray, chessboard_size)

    if found:
        cv2.drawChessboardCorners(frame, chessboard_size, corners, found)
        cv2.putText(frame, f"Ready to save ({image_count}/{max_images})", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "Chessboard not detected", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('Calibration Capture', frame)
    key = cv2.waitKey(1)

    if key == ord('s') and found:
        filename = os.path.join(output_dir, f"calib_{image_count}.jpg")
        cv2.imwrite(filename, frame)
        print(f"Saved {filename}")
        image_count += 1

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
