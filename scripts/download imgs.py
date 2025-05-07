import os
import urllib.request

# Create folders
os.makedirs("left", exist_ok=True)
os.makedirs("right", exist_ok=True)

# Base URL for the OpenCV GitHub repository
base_url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/"

# Image indices to download
image_indices = [f"{i:02d}" for i in range(1, 15)]  # 01 to 14

# Download left and right images
for idx in image_indices:
    for side in ["left", "right"]:
        filename = f"{side}{idx}.jpg"
        url = base_url + filename
        out_path = os.path.join(side, filename)
        try:
            print(f"Downloading {filename} ...")
            urllib.request.urlretrieve(url, out_path)
        except Exception as e:
            print(f"Failed to download {filename}: {e}")
