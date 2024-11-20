import cv2
import os
import shutil

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

# Function to split photos between folders
def split_data():
    # Defining folders
    source_path = r'C:/mgr/data/Teeth Segmentation PNG/d2/img'
    train_path = r'C:/mgr/data/TRAIN_IMAGES'
    valid_path = r'C:/mgr/data/2D_XRAY/VALID_IMAGES'

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

def main():
    # Call test function to display the image
    # show_image(r"C:/mgr/data/kakashi.png")
    
    data_root_path = r'C:/mgr/data' # idk if needed

    # Split data into train and valid data
    split_data()

if __name__ == "__main__":
    main()
