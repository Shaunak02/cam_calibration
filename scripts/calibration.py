import cv2
import numpy as np
import glob
import os

# Chessboard dimensions (number of inner corners per row and column)
chessboard_size = (9, 6)

# Prepare 3D points in real-world space (z=0 since it's a flat board)
objp = np.zeros((chessboard_size[0]*chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)

# Arrays to store object points and image points
objpoints = []  # 3D points
imgpoints = []  # 2D image points

# Load images
images = glob.glob('opencv_calibration_images/*.jpg')

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

    if ret:
        objpoints.append(objp)
        imgpoints.append(corners)

        # Optional: Draw and display the corners
        cv2.drawChessboardCorners(img, chessboard_size, corners, ret)
        cv2.imshow('Corners', img)
        cv2.waitKey(1000)

cv2.destroyAllWindows()

# Calibrate the camera
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None)

# Print results
print("Camera Matrix:\n", camera_matrix)
print("Distortion Coefficients:\n", dist_coeffs)

# Save for future use
np.savez('calibration_data.npz', camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)

# Undistort a sample image
img = cv2.imread(images[0])
h, w = img.shape[:2]
new_camera_mtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w,h), 1, (w,h))
undistorted = cv2.undistort(img, camera_matrix, dist_coeffs, None, new_camera_mtx)

# Show side-by-side
cv2.imshow('Original', img)
cv2.imshow('Undistorted', undistorted)
cv2.waitKey(0)
cv2.destroyAllWindows()
