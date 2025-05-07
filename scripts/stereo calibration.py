import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt # type: ignore
import open3d as o3d # type: ignore

# Termination criteria for cornerSubPix
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points (0,0,0) ... (6,5,0) for a 7x6 chessboard
objp = np.zeros((6*7, 3), np.float32)
objp[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)

# Arrays to store object points and image points from all images
objpoints = []      # 3D points in real world space
imgpoints_left = [] # 2D points in image plane of left camera
imgpoints_right = [] # 2D points in image plane of right camera 

# Load image file names
images_left = sorted(glob.glob('left/left*.jpg'))
images_right = sorted(glob.glob('right/right*.jpg'))

assert len(images_left) == len(images_right), "Mismatch in number of left and right images!"

for img_left_path, img_right_path in zip(images_left, images_right):
    imgL = cv2.imread(img_left_path)
    imgR = cv2.imread(img_right_path)
    grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    retL, cornersL = cv2.findChessboardCorners(grayL, (7, 6), None)
    retR, cornersR = cv2.findChessboardCorners(grayR, (7, 6), None)

    if retL and retR:
        objpoints.append(objp)

        corners2L = cv2.cornerSubPix(grayL, cornersL, (11, 11), (-1, -1), criteria)
        corners2R = cv2.cornerSubPix(grayR, cornersR, (11, 11), (-1, -1), criteria)

        imgpoints_left.append(corners2L)
        imgpoints_right.append(corners2R)

# Get image shape
img_shape = grayL.shape[::-1]

# Calibrate individual cameras
retL, mtxL, distL, rvecsL, tvecsL = cv2.calibrateCamera(objpoints, imgpoints_left, img_shape, None, None)
retR, mtxR, distR, rvecsR, tvecsR = cv2.calibrateCamera(objpoints, imgpoints_right, img_shape, None, None)

# Stereo calibration
flags = 0
flags |= cv2.CALIB_FIX_INTRINSIC

criteria_stereo = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5)

retStereo, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
    objpoints, imgpoints_left, imgpoints_right,
    mtxL, distL, mtxR, distR,
    img_shape, criteria=criteria_stereo, flags=flags
)

# print("\nStereo Calibration RMS error:", retStereo)
# print("Rotation matrix (R):\n", R)
# print("Translation vector (T):\n", T)
# print("Essential matrix (E):\n", E)
# print("Fundamental matrix (F):\n", F)



### IMAGE RECTIFICATION


#Load images
imgL = cv2.imread('left/left01.jpg')
imgR = cv2.imread('right/right01.jpg')
h, w = imgL.shape[:2]

# Stereo rectification
R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
    mtxL, distL, mtxR, distR, (w, h), R, T, alpha=1
)                                                       #alpha=0: Zooms in and removes black borders. Use alpha=1 to retain full image.

np.save("outputs/Q_matrix.npy", Q)


#Compute undistortion and rectification maps
mapLx, mapLy = cv2.initUndistortRectifyMap(mtxL, distL, R1, P1, (w,h), cv2.CV_32FC1)
mapRx, mapRy = cv2.initUndistortRectifyMap(mtxR, distR, R2, P2, (w,h), cv2.CV_32FC1)

#apply remapping
rectifiedL = cv2.remap(imgL, mapLx, mapLy, cv2.INTER_LINEAR)
rectifiedR = cv2.remap(imgR, mapRx, mapRy, cv2.INTER_LINEAR)

cv2.imwrite("rectified_left01.jpg", rectifiedL)
cv2.imwrite("rectified_right01.jpg", rectifiedR)


#draw horizontal lines for visualization
def draw_lines(img):
    for y in range(0, img.shape[0], 50):
        cv2.line(img, (0, y), (img.shape[1], y), (0, 255, 0), 1)
    return img



# cv2.imshow("Rectified Left", draw_lines(rectifiedL.copy()))
# cv2.imshow("Rectified Right", draw_lines(rectifiedR.copy()))
# cv2.waitKey(0)
# cv2.destroyAllWindows()



### DEPTH ESTIMATION USING StereoBM (block matching)


#Load rectified images
imgL = cv2.imread('rectified_left01.jpg', cv2.IMREAD_GRAYSCALE)
imgR = cv2.imread('rectified_right01.jpg', cv2.IMREAD_GRAYSCALE)

assert imgL is not None and imgR is not None, "Images not loaded correctly."

# Create StereoBM object and compute disparity map
stereo = cv2.StereoBM_create(numDisparities=128, blockSize=9)
disparity = stereo.compute(imgL, imgR)

# Normalize the values to a 0–255 range for display
disparity_normalized = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
disparity_normalized = np.uint8(disparity_normalized)

# Show or save result
plt.imshow(disparity_normalized, 'gray')
plt.title('Disparity Map')
plt.axis('off')
plt.show()

print("Disparity range:", disparity.min(), disparity.max())


#Convert disparity to actual depth values
# Reproject image to 3D using Q matrix from stereoRectify
Q = np.load("outputs/Q_matrix.npy")
points_3D = cv2.reprojectImageTo3D(disparity, Q)
colors = cv2.imread("rectified_left01.jpg")  # Load color for texturing

# Mask to remove points with no depth
mask = disparity > disparity.min()
output_points = points_3D[mask]
output_colors = colors[mask]

# Convert to Open3D format
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(output_points)
pcd.colors = o3d.utility.Vector3dVector(output_colors.astype(np.float64) / 255.0)

# print(f"Number of points in point cloud: {np.asarray(pcd.points).shape[0]}")

points = np.asarray(pcd.points)
# print("First 5 points:\n", points[:5])
# print("Max Z (depth):", np.max(points[:, 2]))
# print("Min Z (depth):", np.min(points[:, 2]))

# Create mask for finite Z values
finite_mask = np.isfinite(points[:, 2]) & (points[:, 2] > 0)

# Filter out invalid points
filtered_points = points[finite_mask]

if pcd.has_colors():
    colors = np.asarray(pcd.colors)
    filtered_colors = colors[finite_mask]
else:
    filtered_colors = None

clean_pc = o3d.geometry.PointCloud()
clean_pc.points = o3d.utility.Vector3dVector(filtered_points)

if filtered_colors is not None:
    clean_pc.colors = o3d.utility.Vector3dVector(filtered_colors)

# Save and view
o3d.io.write_point_cloud("outputs/filtered_pointcloud.ply", clean_pc)
o3d.visualization.draw_geometries([clean_pc])