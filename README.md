# Image Processing Basics with OpenCV

Basic pixel manipulation is the groundwork before feeding images to AI models. This notebook walks through the full pipeline with OpenCV and NumPy, including the mathematics of image convolution.

## What's inside

| Step | What it does | Key functions |
|---|---|---|
| 1. Read | Load an image as a BGR `uint8` array | `cv2.imread` |
| 2. Grayscale | Weighted channel sum, `Y = 0.299R + 0.587G + 0.114B` | `cv2.cvtColor` |
| 3. Convolution | Hand-written 2D convolution (blur, sharpen, Sobel), verified against OpenCV | `cv2.filter2D` |
| 4. Edge detection | Sobel gradient magnitude and Canny with threshold comparison | `cv2.Sobel`, `cv2.Canny` |
| 5. Bounding boxes | Morphological close, contours, rectangles | `cv2.findContours`, `cv2.boundingRect` |
| 6. AI preprocessing | RGB, resize, ImageNet normalization, CHW, batch dim | `cv2.resize` |

## Results

![Grayscale](images/02_grayscale.png)
![Convolution kernels](images/03_convolution.png)
![Edge detection](images/04_edges.png)
![Bounding boxes](images/05_bounding_boxes.png)

## Run it

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
pip install -r requirements.txt
jupyter notebook opencv_image_processing_basics.ipynb
```

The notebook uses a generated demo image by default. To use your own, set `IMAGE_PATH = "your_photo.jpg"` in the first cell of Section 1.

## Key takeaways

- OpenCV uses **BGR** channel order, not RGB.
- Convolution multiplies a small kernel with each pixel neighborhood and sums the result. The manual implementation here matches `cv2.filter2D`.
- Sobel kernels sum to zero, so flat regions give zero response and only intensity changes show up.
- CNN layers perform this same operation, with kernels that are learned instead of hand-designed.

## Tech

Python, OpenCV, NumPy, Matplotlib
