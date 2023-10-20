# import time
# import cProfile, pstats
# from pstats import SortKey
# import cv2
# import numpy as np
from PIL import Image

from codes.yolo3_eval import YOLO


def run():
    yolo = YOLO()
    mode = "predict"

    if mode == "predict":
        img_path = 'img/street.jpg'
        # img = input('Input image filename:', img_path)
        try:
            image = Image.open(img_path)
        except:
            print('Open Error! Try again!')
        else:
            r_image = yolo.predict(image)
            r_image.show()


if __name__ == "__main__":
    run()
