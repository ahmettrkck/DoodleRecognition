import numpy
import random
from PIL import Image, ImageDraw, ImageOps

##########################################################################

def convert_drawing_to_PIL(drawing, width, height):
    """
    Converts drawing to PIL image.
    ARGS:
        drawing - from 'drawing' column
        width
        height
    RETURNS:
        pil_image
    """
    pil_image = Image.new('RGB', (width, height), 'white')  # initialize empty image

    draw = ImageDraw.Draw(pil_image)

    for x, y in drawing:
        for i in range(1, len(x)):
            draw.line((x[i-1], y[i-1], x[i], y[i]), fill=0)  # draw strokes

    return pil_image


def convert_drawing_to_np_raw(drawing, width, height):
    """
    ARGS:
        drawing
        width
        height
    RETURNS:
        image - drawing as numpy array (28X28)
    """
   
    image = numpy.zeros((28, 28))  # empty array

    pil_image = convert_drawing_to_PIL(drawing, width, height)  # create PIL image
    pil_image.thumbnail((28, 28), Image.ANTIALIAS) # resize
    pil_image = pil_image.convert('RGB')
    pixels = pil_image.load()

    
    for i in range(0, 28):
        for j in range(0, 28):
            image[i, j] = 1 - pixels[j, i][0] / 255 # fill array with pixel values

    return image


def convert_np_to_PIL(image, width, height):
    """
    Converts numpy to PIL image.
    ARGS:
        image
    RETURNS:
        pil_image
    """
    image_reshape = image.reshape(width, height)

    pil_image = Image.new('RGB', (width, height), 'white')
    pixels = pil_image.load()

    for i in range(0, width):
        for j in range(0, height):
            if image_reshape[i, j] > 0:
                pixels[j, i] = (255 - int(image_reshape[i, j] * 255), 255 -
                                int(image_reshape[i, j] * 255), 255 - int(image_reshape[i, j] * 255))

    return pil_image


def rotate(input_image, angle=60, size=(28, 28)):
    """
    Rotates PIL Image
    ARGS:
        input_image
        angle
        size - size of output image
    RETURNS:
        rotated_image
    """
    input_image = input_image.convert('RGBA')
    rotate = input_image.rotate(angle)
    rotated_image = Image.new("RGBA", size, "white")
    rotated_image.paste(rotate, (0, 0), rotate)

    return rotated_image

def flip(input_image):
    """
    Flips PIL Image
    ARGS:
        input_image
    RETURNS:
        flipped_image
    """
    return input_image.transpose(Image.FLIP_LEFT_RIGHT)

def convert_PIL_to_np(pil_image, width=28, height=28):
    """
    Converts PIL Image to numpy array
    ARGS:
        pil_image
    RETURNS:
        image
    """
    pil_image = pil_image.convert('RGB')

    image = numpy.zeros((width, height))
    pixels = pil_image.load()

    for i in range(0, width):
        for j in range(0, height):
            image[i, j] = 1 - pixels[j, i][0] / 255

    return image


def enrich_images(X_train, y_train, input_size):
    """
    Expand original dataset with modified images.
    ARGS:
        X_train - original training data
        y_train - original dataset labels
        input_size
    RETURNS:
        enriched_X_train
        enriched_y_train
    """
    enriched_X_train = X_train.copy()
    enriched_y_train = y_train.copy().reshape(y_train.shape[0], 1)

    for i in range(0, X_train.shape[0]):
        image = X_train[i]
        pil_image = convert_np_to_PIL(image, 28, 28)
   
        angle = random.randint(5, 25) 

        rotated = convert_PIL_to_np(rotate(pil_image, angle))
        flipped = convert_PIL_to_np(flip(pil_image))
    
        # extend dataset
        enriched_X_train = numpy.append(     
            enriched_X_train, rotated.reshape(1,  input_size), axis=0)
        enriched_X_train = numpy.append(
            enriched_X_train, flipped.reshape(1,  input_size), axis=0)
        enriched_y_train = numpy.append(
            enriched_y_train, y_train[i].reshape(1, 1), axis=0)
        enriched_y_train = numpy.append(
            enriched_y_train, y_train[i].reshape(1, 1), axis=0)

    return enriched_X_train, enriched_y_train


def crop(image):
    """
    ARGS:
        image
    RETURNS:
        output_image
    """

    width, height = image.size  # get image size

    image_pixels = image.load()

    row_strokes = []
    column_strokes = []

    for i in range(0, width):
        for j in range(0, height):
            # record image coordinates
            if (image_pixels[i, j][0] > 0):
                column_strokes.append(i)
                row_strokes.append(j)

    # find image box and crop
    if (len(row_strokes)) > 0:
        minimum_row = numpy.array(row_strokes).min()
        maximum_row = numpy.array(row_strokes).max()
        minimum_column = numpy.array(column_strokes).min()
        maximum_column = numpy.array(column_strokes).max()

        border = (minimum_column, minimum_row, width - maximum_column, height - maximum_row)
        image = ImageOps.crop(image, border)

    new_width, new_height = image.size

    # paste cropped image to centre of square image
    output_image = Image.new("RGBA", (max(new_width, new_height), max(
        new_width, new_height)), "white")
    offset = ((max(new_width, new_height) - new_width) //
              2, (max(new_width, new_height) - new_height) // 2)
    
    output_image.paste(image, offset, image)

    output_image.thumbnail((28, 28), Image.ANTIALIAS)

    return output_image


def normalise_array(input_array):
    """
    Performs linear normalisation of array (pre-written)

    ARGS:
        input_array - original
    RETURNS:
        normalised_array - normalised
    """
    normalised_array = input_array.astype('float')
    
    for i in range(3):
        minimum_value = normalised_array[..., i].min()
        maximum_value = normalised_array[..., i].max()
        if minimum_value != maximum_value:
            normalised_array[..., i] -= minimum_value
            normalised_array[..., i] *= (255.0/(maximum_value-minimum_value))
    return normalised_array


def normalise_image(image):
    """
    Normalises image (pre-written)
    ARGS:
        image
    RETURNS:
        normalised_image
    """
    array = numpy.array(image)

    return Image.fromarray(normalise_array(array).astype('uint8'), 'RGBA')


def alpha_comp_images(image_1, image_2):
    """
    Generate alpha composite of input RBGA images (pre-written)

    ARGS:
        image_1
        image_2
    RETURNS:
        alpha_comp_image

    """
    image_1 = numpy.asarray(image_1)
    image_2 = numpy.asarray(image_2)

    alpha_comp_image = numpy.empty(image_1.shape, dtype='float')
    alpha = numpy.index_exp[:, :, 3:]
    rgb = numpy.index_exp[:, :, :3]

    alpha_image_1 = image_1[alpha] / 255.0
    alpha_image_2 = image_2[alpha] / 255.0
    alpha_comp_image[alpha] = alpha_image_1 + alpha_image_2 * (1 - alpha_image_1)

    invalid_setting = numpy.seterr(invalid='ignore')
    alpha_comp_image[rgb] = (image_1[rgb] * alpha_image_1 + image_2[rgb] *
                   alpha_image_2 * (1 - alpha_image_1)) / alpha_comp_image[alpha]
    numpy.seterr(**invalid_setting)

    alpha_comp_image[alpha] *= 255
    numpy.clip(alpha_comp_image, 0, 255)
    alpha_comp_image = alpha_comp_image.astype('uint8')
    alpha_comp_image = Image.fromarray(alpha_comp_image, 'RGBA')

    return alpha_comp_image


def colour_alpha_comp(input_image, colour=(255, 255, 255)):
    """
    Convert RGBA to RGB and make single colour
    
    ARGS:
        input_image - RGBA Image
        colour
    RETURNS:
        processed image

    """
    return alpha_comp_images(input_image, Image.new('RGBA', size=input_image.size, color=colour + (255,)))


def rgba_to_rgb(rgba_image):
    """
    ARGS:
        rgba_image
    RETURNS:
        rgb_image
    """
    rgb_image = colour_alpha_comp(rgba_image)
    rgb_image.convert('RGB')

    return rgb_image
