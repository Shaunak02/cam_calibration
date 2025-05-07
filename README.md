# 🎯 Camera Calibration with OpenCV

This project performs **single-camera** and **stereo-camera calibration** using OpenCV. It estimates intrinsic and extrinsic parameters, corrects distortion, and rectifies stereo images for further 3D processing.

---

## 📁 Project Structure
.
├── scripts/ # Python scripts for calibration
├── images/ # Calibration image sets
├── output/ # Undistorted images, parameters
├── notebooks/ # Optional Jupyter notebooks for experiments
├── .gitignore
├── README.md
└── requirements.txt

---

## 🔧 Features

### ✅ Single Camera Calibration
- Uses checkerboard images
- Estimates:
  - Intrinsic parameters (focal length, principal point, distortion)
  - Undistorts sample images

### ✅ Stereo Camera Calibration
- Uses stereo image pairs (`leftXX.jpg`, `rightXX.jpg`)
- Computes:
  - Intrinsics for both cameras
  - Rotation (`R`) and translation (`T`) between them
  - Rectified image pairs
- Saves:
  - Calibration data (`.npz`, `.npy`)
  - Optional disparity maps

---

## 🚀 How to Run

1. Activate your virtual environment:
   ```bash
   source cam_calib_env/bin/activate  # or cam_calib_env\Scripts\activate on Windows

2. Install requirements:
   ```bash
   pip install -r requirements.txt

3. Run scripts:

    scripts/calibration.py    
    scripts/stereo_calibration.py



