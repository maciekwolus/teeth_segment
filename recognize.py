import cv2
import os
import shutil
import numpy as np
import json

# Test function to display an image using OpenCV.
def show_image(image_path):
    image = cv2.imread(image_path)
    tgt_height = 300
    height, width = image.shape[:2]
    scale_factor = tgt_height / height
    tgt_width = int(width * scale_factor)
    resized_image = cv2.resize(image, (tgt_width, tgt_height))
    cv2.imshow("Image Viewer", resized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Function to copy files
def copy_files(source_path, file_list, destination_dir):
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)  # Create the directory if it does not exist
    for file_name in file_list:
        src_path = os.path.join(source_path, file_name)
        dest_path = os.path.join(destination_dir, file_name)
        if not os.path.exists(dest_path):  # Check if file already exists
            shutil.copy(src_path, dest_path)

# Function to clear a directory
def clear_directory(directory_path):
    if os.path.exists(directory_path):
        for file in os.listdir(directory_path):
            file_path = os.path.join(directory_path, file)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)  # Remove the file
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)  # Remove the folder
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")

# Function to split photos between folders
def split_data(source_path, train_path, valid_path):
    # Clear existing data in train and valid directories
    clear_directory(train_path)
    clear_directory(valid_path)

    # Getting list of files
    image_files = [f for f in os.listdir(source_path) if f.endswith('.jpg')]

    # Splitting ratio
    train_ratio = 0.85
    valid_ratio = 0.15

    # Counts of images after split
    total_images = len(image_files)
    train_count = int(total_images * train_ratio)
    valid_count = total_images - train_count

    # Split image paths
    train_files = image_files[:train_count]
    valid_files = image_files[train_count:train_count + valid_count]

    # Copy images
    copy_files(source_path, train_files, train_path)
    copy_files(source_path, valid_files, valid_path)

    # Show parameters
    print(f"Total images: {total_images}")
    print(f"Training images: {len(train_files)}")
    print(f"Validation images: {len(valid_files)}")

# Function to convert polygons from JSON to jpg
def json_to_mask(json_folder, output_folder, image_shape):
    # Create the directory if it does not exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)  

    # Clear existing data in train and valid directories
    clear_directory(output_folder)
    
    # List of all files
    json_files = [f for f in os.listdir(json_folder) if f.endswith('.json')]

    # Iterate through JSONs
    for json_file in json_files:
        json_path = os.path.join(json_folder, json_file)

        # Open JSONs
        with open(json_path, 'r') as file:
            data = json.load(file)

        # Extract polygons
        # Format like that - {'objects': [{'points': {'exterior': [[x1, y1], [x2, y2], ...]}}, ...]}
        coordinates = []
        for obj in data.get('objects', []):
            if 'points' in obj and 'exterior' in obj['points']:
                coordinates.append(np.array(obj['points']['exterior'], dtype=np.int32))

        # White starting mask
        mask = np.ones(image_shape, dtype=np.uint8) * 255

        # Draw polygons on mask
        for polygon in coordinates:
            cv2.fillPoly(mask, [polygon], color=(0)) # Fulfill inside of polygon with black
            cv2.polylines(mask, [polygon], isClosed=True, color=(255), thickness=3)  # Draw white contur

        # Reverse colors
        mask = cv2.bitwise_not(mask)

        # Save masks
        mask_filename = os.path.splitext(os.path.splitext(json_file)[0])[0]
        mask_path = os.path.join(output_folder, mask_filename + ".jpg")  # Adding .jpg
        cv2.imwrite(mask_path, mask)



def main():
    # Call test function to display the image
    # show_image(r"C:/mgr/data/kakashi.png")
    
    # Defining data folders
    data_root_path = r'C:/mgr/data'
    source_path = data_root_path + r'/Teeth Segmentation PNG/d2/img'
    train_path = data_root_path + r'/TRAIN_IMAGES'
    valid_path = data_root_path + r'/VALID_IMAGES'
    
    # Defining mask data
    masks_path = data_root_path + r'/MASKS'
    json_path = data_root_path + r'/Teeth Segmentation PNG/d2/ann'
    image_shape = (1024, 2041)

    # Split data into train and valid data
    split_data(source_path, train_path, valid_path)

    # Create masks
    json_to_mask(json_path, masks_path, image_shape)

if __name__ == "__main__":
    main()