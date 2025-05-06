# 🧠 OCR Text Reader with TTS and Gradio UI

This project is a full OCR (Optical Character Recognition) pipeline using **Tesseract OCR** for text extraction, **OpenCV** for image preprocessing, **pyttsx3** for offline text-to-speech, and an optional **Gradio** interface for user interaction.

---

## 📦 Requirements

Install dependencies with:

```bash
pip install opencv-python pytesseract pyttsx3 numpy gradio
```

Make sure Tesseract OCR is installed on your machine.

🛠️ Example Tesseract path for Windows:

```bash
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
```

🚀 How It Works
🔧 1. Image Preprocessing
The image is first converted to grayscale and denoised:

```bash
img = cv2.imread("testocr.png")
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.medianBlur(img, 3)  # Removes salt-and-pepper noise
```

Adaptive thresholding is used to handle uneven lighting:

```bash
thresh = cv2.adaptiveThreshold(
    img, 255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY, 31, 10
)
```

Morphological operations help fill gaps inside letters:

```bash
kernel = np.ones((2, 2), np.uint8)
img = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
```

Further smoothing and binarization:

```bash
img = cv2.GaussianBlur(img, (5, 5), 0)
_, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
```

📄 2. Text Extraction with Tesseract

Extract string from image:
```bash
custom_config = r'--oem 3 --psm 6'
text = pytesseract.image_to_string(img, lang='eng', config=custom_config)
print("OCR RESULTS:\n", text)
```

Draw bounding boxes around high-confidence text:
```bash
from pytesseract import Output
ocr_data = pytesseract.image_to_data(img, output_type=Output.DICT)

for i in range(len(ocr_data['text'])):
    if int(ocr_data['conf'][i]) > 60:
        x, y, w, h = (ocr_data["left"][i], ocr_data['top'][i],
                      ocr_data['width'][i], ocr_data['height'][i])
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img, ocr_data['text'][i], (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
```

🔊 3. Text-to-Speech (Offline)

Speak the extracted text using pyttsx3:
```bash
import pyttsx3
engine = pyttsx3.init()
engine.say(text)
engine.runAndWait()
```
