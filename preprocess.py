import cv2
import config as cf

def crop(image):
    return image[70:-20, :, :]

def resize(image):
    return cv2.resize(image, (cf.IMAGE_WIDTH, cf.IMAGE_HEIGHT), cv2.INTER_AREA)

def rgb2yuv(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2YUV)

def preprocess(image):
    image = crop(image)
    image = resize(image)
    image = rgb2yuv(image)
    return image
