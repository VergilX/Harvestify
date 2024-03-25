import os
import cv2

def resize_images_in_folder(input_folder, output_folder, target_size=(256, 256)):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    files = os.listdir(input_folder)
    for file_name in files:
        picture_folder_path = os.path.join(input_folder, file_name)
        plant_types_folder = os.listdir(picture_folder_path)

        for plant_file_name in plant_types_folder:
            plant_file_path = os.path.join(picture_folder_path, plant_file_name)
            output_path = os.path.join(output_folder, file_name, plant_file_name)

            # Check if the subdirectory exists, if not create it
            output_subfolder = os.path.join(output_folder, file_name)
            if not os.path.exists(output_subfolder):
                os.makedirs(output_subfolder)

            image = cv2.imread(plant_file_path)
            if image is not None:
                resized_image = cv2.resize(image, target_size)
                cv2.imwrite(output_path, resized_image)
                print(f"Resized and saved '{plant_file_name}' in '{output_subfolder}'")

# Specify the input and output folders
input_folder = 'train'
output_folder = 'resize_train'

# Call the function to resize images
resize_images_in_folder(input_folder, output_folder)
import os
import cv2

def resize_images_in_folder(input_folder, output_folder, target_size=(256, 256)):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    files = os.listdir(input_folder)
    for file_name in files:
        picture_folder_path = os.path.join(input_folder, file_name)
        plant_types_folder = os.listdir(picture_folder_path)

        for plant_file_name in plant_types_folder:
            plant_file_path = os.path.join(picture_folder_path, plant_file_name)
            output_path = os.path.join(output_folder, file_name, plant_file_name)

            # Check if the subdirectory exists, if not create it
            output_subfolder = os.path.join(output_folder, file_name)
            if not os.path.exists(output_subfolder):
                os.makedirs(output_subfolder)

            image = cv2.imread(plant_file_path)
            if image is not None:
                resized_image = cv2.resize(image, target_size)
                cv2.imwrite(output_path, resized_image)
                print(f"Resized and saved '{plant_file_name}' in '{output_subfolder}'")

# Specify the input and output folders
input_folder = 'valid'
output_folder = 'resize_valid'

# Call the function to resize images
resize_images_in_folder(input_folder, output_folder)
