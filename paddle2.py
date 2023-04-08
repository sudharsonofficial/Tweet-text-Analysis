from paddleocr import PaddleOCR,draw_ocr
from matplotlib import pyplot as plt
import cv2
from PIL import Image
import numpy as np
import os

class pad():
    ocr_m=PaddleOCR(lang='en')

    def x_t(self,img_p):
        # img_p=r"C:\Users\Sudha\Desktop\Sentiment analysis\twitter6.png"
        # img_p
        # img_p=np.asarray(img_p)
        # print("The type is :",type(img_p))
        # data=Image.fromarray(img_p)
        # im = cv2.imshow(img_p)
        # print("********************",img_p)
        result=self.ocr_m.ocr(img_p)

        for i in result:
            a=(i[1][1][0])
            x=(i[2][1][0])
            y=(i[3][1][0])
        txt=a+" "+x+" "+y
        return txt