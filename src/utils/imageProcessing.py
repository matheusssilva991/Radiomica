from pydicom import dcmread
from cv2 import imread, imwrite, COLOR_BGR2GRAY, INTER_AREA, cvtColor, resize

def extract_image_dicom(image_path, save=False, path=None, image_type=None):
    try:
        # Read the dicom file
        dicom_file = dcmread(image_path)
        
        # Extract the image from dicom file
        image = dicom_file.pixel_array

        if save:
            image_type = image_type or "png"
            path = path or image_path.split("/")[-1].replace("dcm", image_type)
            imwrite(path, image)

        return image
    except FileNotFoundError:
        raise Exception("Arquivo não encontrado")
    


def resize_image(image, dim, save=False, path=None, image_type=None):
    try:
        if isinstance(image, str):
            
            image = imread(image)
            image = cvtColor(image, COLOR_BGR2GRAY)
        
        # resize image
        resized = resize(image, dim, interpolation = INTER_AREA)
        
        if save:
            image_type = image_type or "png"
            path = path or f"./resized_image.{image_type}"
            imwrite(path, resized)

        return resized
    except FileNotFoundError:
        raise Exception("Arquivo não encontrado")
