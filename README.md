# Image Processing TPs with OpenCV & NumPy

A structured collection of practical image processing tasks (Travaux Pratiques) implemented in Python using OpenCV and NumPy. Each TP explores a key concept in computer vision, including filtering, morphological operations, face detection, and Kalman filtering. Designed for educational and experimental purposes.

## Content Overview

---

### TP1 – Image Histogram and Binarization  
**Title:** *Histogram Representation (Lines & Columns)*  
Reads a grayscale image, applies binarization via thresholding, and computes horizontal and vertical histograms. Displays them alongside the source image.

---

### TP2 – Spatial Filtering  
**Title:** *Mean and Median Filters*  
Implements manual 3x3 mean and median filters to smooth grayscale images and reduce noise.

---

### TP3 – Thresholding with Trackbars  
**Title:** *Interactive Thresholding (Five Types)*  
Applies five OpenCV thresholding techniques with real-time control using trackbars for threshold value and type selection.

---

### TP4 – Edge Detection by Gradient  
**Title:** *Gradient-based Contour Extraction*  
Computes image gradients (X and Y), combines them using Euclidean norm, and allows edge extraction based on a user-defined threshold.

---

### TP5 – Convolution Filters & Sharpening  
**Title:** *Gaussian Blur, Laplacian, and Gradient Filters*  
Applies predefined and custom filters using `cv2.filter2D`, including Laplacian sharpening and Gaussian smoothing (manual and auto-generated).

---

### TP6 – Morphological Operations  
**Title:** *Erosion, Dilation, and Morphological Gradient*  
Demonstrates erosion, dilation, and morphological gradient effects on binary images. Trackbars allow dynamic structuring element size.

---

### TP7 – Color Space Transformations  
**Title:** *RGB Channel Splitting and HSV Conversion*  
Splits RGB channels, computes a grayscale image manually, and converts the image to HSV and float representations for color analysis.

---

### TP8 – Video Capture and Saving  
**Title:** *Real-Time Camera Feed & Video Recording*  
Captures video from webcam or phone camera (IP webcam), displays frames in real-time, and saves the output with timestamped FPS display.

---

### TP9 – Color Object Tracking  
**Title:** *Contour Detection via HSV Color Range*  
Detects color-specific objects using HSV thresholding, contour extraction, and minimum enclosing circles. Used for real-time tracking.

---

### TP10 – Kalman Filter Implementation  
**Title:** *Object Tracking with Kalman Filter*  
Implements a 2D Kalman filter to track moving points, predicting and correcting positions using state estimation.

---

### TP11 – Face Detection with Tracking  
**Title:** *Face Detection & Prediction*  
Uses Haar cascade classifiers to detect human faces and integrates Kalman filtering to predict their next position in real time.

---

### TP12 – Cell Phone Detection and Tracking
**Title:** *YOLO + DeepSORT for Real-Time Cell Phone Tracking*
Detection of cell phones using YOLOv5n and tracking with DeepSORT, live webcam display, and annotated center tracking.

---

## 👩‍💻 Author

- **Ferchichi Manel**

---

##  Technologies

- Python 3.x  
- OpenCV 4.x  
- NumPy  
- Matplotlib (for histogram plotting)  
- Haar Cascades (face detection XML)


---

## OpenCV + NumPy Cheatsheet


A quick reference for the most common operations you'll use during real-time image/video processing with OpenCV and NumPy.

---

### 1. Image I/O & Properties

| Task                 | Code / Explanation                                   |
| -------------------- | ---------------------------------------------------- |
| Read a color image   | `img = cv2.imread(image_path)`                       |
| Read grayscale image | `img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)` |
| Get image dimensions | `h, w, c = img.shape`                                |
| Check image type     | `img.dtype`                                          |
| Convert to float     | `img = img.astype(np.float32)`                       |

---

### 2. Image Initialization

| Task                        | Code                                  |
| --------------------------- | ------------------------------------- |
| Create empty (zeros) image  | `img = np.zeros((h, w, c), np.uint8)` |
| Clone shape & type          | `img2 = np.zeros_like(img)`           |
| Create image filled with 1s | `img = np.ones((h, w), np.float32)`   |

---

### 3. Image Manipulation

| Task                 | Code                                                |
| -------------------- | --------------------------------------------------- |
| Invert image         | `img_inv = 255 - img`                               |
| Normalize to \[0, 1] | `img = img / 255.0`                                 |
| Custom normalization | `img[y,x] = ((img[y,x] - min) / (max - min)) * 255` |
| Resize image         | `resized = cv2.resize(img, (width, height))`        |
| Concatenate images   | `cv2.hconcat([img1, img2])`                         |

---

### 4. Filtering & Convolution

| Task             | Code                                                     |
| ---------------- | -------------------------------------------------------- |
| Apply kernel     | `cv2.filter2D(img, -1, kernel)`                          |
| Normalize result | `cv2.normalize(imgRes, imgRes, 0, 255, cv2.NORM_MINMAX)` |

---

### 5. Histogram & Plotting (Matplotlib)

| Task                | Code                                              |
| ------------------- | ------------------------------------------------- |
| Init histogram      | `histo = np.zeros((256, 1), np.uint16)`           |
| Calculate histogram | `cv2.calcHist([img], [0], None, [256], [0,256])`  |
| Plot                | `plt.plot(histo); plt.xlim([0, 255]); plt.show()` |

---

### 6. Trackbars (Interactive Sliders)

| Task            | Code                                                          |
| --------------- | ------------------------------------------------------------- |
| Create trackbar | `cv2.createTrackbar("thresh", "window", 0, 255, callback_fn)` |

---

### 7. Morphological Operations

| Task          | Code                                                       |
| ------------- | ---------------------------------------------------------- |
| Thresholding  | `cv2.threshold(img, 128, 255, cv2.THRESH_BINARY, img)`     |
| Erode         | `cv2.erode(img, kernel)`                                   |
| Dilate        | `cv2.dilate(img, kernel)`                                  |
| MorphologyEx  | `cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)`        |
| Create kernel | `cv2.getStructuringElement(cv2.MORPH_CROSS, (size, size))` |

---

### 8. Gradient & Edge Detection

| Task              | Code                                                      |
| ----------------- | --------------------------------------------------------- |
| Compute gradients | `grad_x = img[:, :-1] - img[:, 1:]`                       |
| Pad gradient      | `grad_x = np.pad(grad_x, ((0,0),(0,1)), mode='constant')` |
| Magnitude         | `grad = np.sqrt(grad_x**2 + grad_y**2)`                   |

---

### 9. Video & Camera Input

| Task           | Code                                       |
| -------------- | ------------------------------------------ |
| Webcam         | `cv2.VideoCapture(0)`                      |
| File           | `cv2.VideoCapture('video.avi')`            |
| IP cam         | `cv2.VideoCapture("http://IP:PORT/video")` |
| Get frame size | `w = int(cap.get(3)), h = int(cap.get(4))` |
| Read frame     | `ret, frame = cap.read()`                  |
| Flip frame     | `cv2.flip(frame, 1, frame)`                |

---

### 10. Video Output (Recording)

| Task           | Code                                                    |
| -------------- | ------------------------------------------------------- |
| Codec          | `fourcc = cv2.VideoWriter_fourcc('X','V','I','D')`      |
| Create writer  | `out = cv2.VideoWriter('file.avi', fourcc, 30, (w, h))` |
| Write frame    | `out.write(frame)`                                      |
| Release writer | `out.release()`                                         |

---

### 11. Image Color Spaces

| Task               | Code                                                                            |
| ------------------ | ------------------------------------------------------------------------------- |
| Split BGR          | `img_b[:,:,0], img_g[:,:,1], img_r[:,:,2] = img[:,:,0], img[:,:,1], img[:,:,2]` |
| Manual RGB to gray | `gray = (b + g + r) / 3`                                                        |
| BGR to HSV         | `cv2.cvtColor(img, cv2.COLOR_BGR2HLS)`                                          |
| BGR to Grayscale   | `cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)`                                         |

---

### 12. Utility / Control Flow

| Task            | Code                                     |
| --------------- | ---------------------------------------- |
| Wait for key    | `cv2.waitKey(0)`                         |
| Destroy windows | `cv2.destroyAllWindows()`                |
| Quit condition  | `if cv2.waitKey(20) & 0xFF == ord('q'):` |
| Random pixel    | `x, y = randrange(w), randrange(h)`      |

---

### 13. NumPy Data Types

| Type      | Range                      |
| --------- | -------------------------- |
| `uint8`   | \[0, 255]                  |
| `int8`    | \[-128, 127]               |
| `uint16`  | \[0, 65535]                |
| `int16`   | \[-32768, 32767]           |
| `uint32`  | \[0, 4294967295]           |
| `int32`   | \[-2147483648, 2147483647] |
| `float32` | \~ \[1.2e-38, 3.4e+38]     |

---



