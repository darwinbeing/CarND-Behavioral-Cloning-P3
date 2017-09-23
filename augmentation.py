import cv2
import numpy as np
import config as cf

def random_flip_left_right(image, steering_angle):
    if np.random.rand() < 0.5:
        image = cv2.flip(image, 1)
        steering_angle = -steering_angle
    return image, steering_angle

def random_translate(image, steering_angle, wrg, hrg):
    tx = wrg * (np.random.rand() - 0.5)
    ty = hrg * (np.random.rand() - 0.5)
    steering_angle += tx * 0.002
    tm = np.array([[1, 0, tx], [0, 1, ty]])
    height, width = image.shape[:2]
    image = cv2.warpAffine(image, tm, (width, height))
    return image, steering_angle

# reference: https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9
def random_shadow(image):
    x1, y1 = cf.IMAGE_WIDTH * np.random.rand(), 0
    x2, y2 = cf.IMAGE_WIDTH * np.random.rand(), cf.IMAGE_HEIGHT
    xm, ym = np.mgrid[0:cf.IMAGE_HEIGHT, 0:cf.IMAGE_WIDTH]

    mask = np.zeros_like(image[:, :, 1])
    mask[np.where((ym - y1) * (x2 - x1) - (y2 - y1) * (xm - x1) > 0)] = 1

    cond = mask == np.random.randint(2)
    ratio = np.random.uniform(low=0.2, high=0.5)

    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    hls[:, :, 1][cond] = hls[:, :, 1][cond] * ratio
    return cv2.cvtColor(hls, cv2.COLOR_HLS2RGB)

# def random_shadow(image):
#     top_y = 320 * np.random.uniform()
#     top_x = 0
#     bot_x = 160
#     bot_y = 320 * np.random.uniform()
#     image_hls = cv2.cvtColor(image,cv2.COLOR_RGB2HLS)
#     shadow_mask = 0 * image_hls[:,:,1]
#     X_m = np.mgrid[0:image.shape[0], 0:image.shape[1]][0]
#     Y_m = np.mgrid[0:image.shape[0], 0:image.shape[1]][1]
#     shadow_mask[((X_m - top_x) * (bot_y-top_y) - (bot_x - top_x) * (Y_m - top_y) >=0)] = 1
#     #random_bright = .25+.7 * np.random.uniform()
#     if np.random.randint(2) == 1:
#         random_bright = .5
#         cond1 = shadow_mask == 1
#         cond0 = shadow_mask == 0
#         if np.random.randint(2) == 1:
#             image_hls[:,:,1][cond1] = image_hls[:,:,1][cond1] * random_bright
#         else:
#             image_hls[:,:,1][cond0] = image_hls[:,:,1][cond0] * random_bright
#     image = cv2.cvtColor(image_hls,cv2.COLOR_HLS2RGB)
#     return image

def random_brightness(image):
    image1 = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    image1 = np.array(image1, dtype = np.float64)
    random_bright = .5 + np.random.uniform()
    image1[:,:,2] = image1[:,:,2] * random_bright
    image1[:,:,2][image1[:,:,2] > 255] = 255
    image1 = np.array(image1, dtype = np.uint8)
    image1 = cv2.cvtColor(image1, cv2.COLOR_HSV2RGB)
    return image1

def augument(image, steering_angle, wrg=100, hrg=10):
    image, steering_angle = random_flip_left_right(image, steering_angle)
    image, steering_angle = random_translate(image, steering_angle, wrg, hrg)
    image = random_shadow(image)
    image = random_brightness(image)
    return image, steering_angle
