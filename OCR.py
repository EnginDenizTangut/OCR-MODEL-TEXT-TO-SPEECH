
import cv2
import pytesseract
import numpy as np
from pytesseract import Output
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
import pyttsx3
import gradio as gr

img = cv2.imread("testocr.png")
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #gri tonlama
img = cv2.medianBlur(img,3) #denoised kucuk leke temizleme
thresh = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,31,10) #heterojen aydinlatma durumlarinin onune gecmek icin
kernel= np.ones((2,2),np.uint8)
img = cv2.morphologyEx(thresh,cv2.MORPH_CLOSE,kernel)
img = cv2.GaussianBlur(img, (5, 5), 0)
_, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
edges = cv2.Canny(img, 100, 200)

"""
cv2.THRESH_BINARY turns the image into black and white based on a threshold
"""

"""MorphologyEx
| Operation         | Description                                                           |
| ----------------- | --------------------------------------------------------------------- |
| `cv2.MORPH_CLOSE` | Dilation followed by erosion; fills small holes inside characters     |
| `cv2.MORPH_OPEN`  | Erosion followed by dilation; removes small noise spots               |
"""

"""ADAPTIVE THRESHOLD PARAMETERS
| Parameter         | Description                                                                 |
| ----------------- | --------------------------------------------------------------------------- |
| `src`             | Input grayscale image                                                       |
| `maxValue`        | Pixel value to assign when condition is met (usually 255 for white)         |
| `adaptiveMethod`  | `cv2.ADAPTIVE_THRESH_MEAN_C` or `cv2.ADAPTIVE_THRESH_GAUSSIAN_C`            |
| `thresholdType`   | `cv2.THRESH_BINARY` or `cv2.THRESH_BINARY_INV`                              |
| `blockSize`       | Size of neighborhood area to calculate threshold (must be odd, e.g., 11–31) |
| `C`               | Constant subtracted from the mean (tunes light/dark regions)                |
"""

"""
| Step              | Explanation                                                                    |
| ----------------- | ------------------------------------------------------------------------------ |
| `medianBlur`      | Very effective against salt-and-pepper noise, common in scanned documents      |
| `adaptiveThreshold` | Uses local thresholding to deal with inconsistent lighting                  |
| `morphologyEx`    | Can close small gaps inside characters; be cautious not to merge letters       |
| `GaussianBlur`    | Softens edges to reduce OCR misreads, but too much can blur out characters     |
"""
custom_config = r'--oem 3 --psm 6'
text = pytesseract.image_to_string(img,lang='eng',config=custom_config)

print("OCR SONUCLARI:")
print(text)
cv2.imshow('Binary Image', img)

ocr_data = pytesseract.image_to_data(img,output_type=Output.DICT)

n_boxes = len(ocr_data['text']) #kac kelime var
for i in range(n_boxes):
    if int(ocr_data['conf'][i]) > 60:
        (x,y,w,h) = (ocr_data["left"][i],ocr_data['top'][i],
                     ocr_data['width'][i],ocr_data['height'][i])
        
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.putText(img, ocr_data['text'][i], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, (0, 255, 0), 1)

cv2.imshow('Detected Text with Boxes', img)
engine = pyttsx3.init()
engine.say(text)
engine.runAndWait()
cv2.waitKey(0)
cv2.destroyAllWindows()


#GRADIO PART