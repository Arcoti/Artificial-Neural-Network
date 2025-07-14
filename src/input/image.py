from PIL import Image
import numpy as np

def convert(image_path: str):
    # Load the image
    image = Image.open(image_path)

    # Resize the image
    resized_image = image.resize((28, 28))

    # Convert to Grayscale
    gray_image = resized_image.convert('L')

    # Convert to numpy array
    pixel_array = np.array(gray_image)

    # Convert to light background
    pixel_array = 255 - pixel_array

    # Normalize
    pixel_array = pixel_array.astype(float) / 255

    # Flatten
    pixel_array = pixel_array.reshape((1, pixel_array.shape[0] * pixel_array.shape[1]))

    return pixel_array
