# VisionArtificialAndImageProcessing

A image 2x2[px] representation in RGB is:
```
#img = numpy.zeros([2,2,3], dtype='uint8')
img = [[[0. 0. 0.]
        [0. 0. 0.]]
       [[0. 0. 0.]
        [0. 0. 0.]]]
```
Spliting in 3 matrix r,g,b, using opencv:
```
r,g,b = cv2.split(img)
```
Where matrix r,g & b have 2x2 size, like this:
```
r=g=b = [[0. 0.]
         [0. 0.]]
```

# PNG images

This type of images have 4 channels: r,g,b & alpha

```
img_4ch = cv2.merge((r, g, b, alpha))
cv2.imwrite("img_4ch.png", img_4ch)
```

Adding the alpha mask to image

```
img_4ch = np.dstack([bgr, alpha])
cv2.imwrite("img_4ch.png", img_4ch)
```

# Useful commands

```
diff_im = cv2.subtract(im, im2)
diff_im = cv2.absdiff(im, im2)
```
#### Creating a mask using 'img_rgb' like reference.

```
# dtype is 'uint8' in images
mask = np.ones( img_rgb.shape[:2], dtype=img_rgb.dtype ) * 0
```

#### Creating a white black image like other image reference.

```
img_black = np.ones_like( img_reference ) * 255
```

## Working with contours

### Functions
- cv2.findContours:
	- cv2.RETR_LIST: Returns all contours. Other methods exist that return only external contours
	- cv2.RETR_EXTERNAL: Return only external most contours of each shape in image. Example, if this was specified below, the oval shape in the yellow rectangle would not be returned.
	- cv2.CHAIN_ARPPOX_SIMPLE: Compresses horizontal, vertical and diagonal segments of contours and leaves only their end points.returns a tuple of values, each tuple contains points along a contour.
	- imutils.grab_contours The problem with the returning tuple is that it is in a different format for OpenCV 2.4, OpenCV 3, OpenCV 3.4, OpenCV 4.0.0-pre, OpenCV 4.0.0-alpha, and OpenCV 4.0.0 (official). This function solves this
- cv2.drawContours:
	- Draw returned contours on the clone of original image (or even a blank canvas), and provide thickness of the layer. (-1 for fill shape)

The contours are define by array with the same size of the image original.

```python
from skimage import data, segmentation
from skimage.segmentation import mark_boundaries
from skimage.future import graph
from google.colab.patches import cv2_imshow

img_tmp = imagen_in_rgb

# Aplying SLIC algorithm to get the matrix of labels with 4 segments
n_segments = 4
labels = segmentation.slic(img_tmp, compactness=30, n_segments=n_segments)
img_boundaries_slic = mark_boundaries(img_tmp, labels,color=(255,0,0),background_label=3)
```

#### Image

<div align="center">
<img src="https://github.com/hoat23/VisionArtificialAndImageProcessing/blob/master/img/img_04_20x20.jpg" width="200" align="center"/>
</div>

#### Matriz contours

<div align="center">
<img src="https://github.com/hoat23/VisionArtificialAndImageProcessing/blob/master/img/img_05_labels.png" width="200" align="center"/>
</div>

## Contours detection

Filter the values 2 from the labels matriz and applying contours detection:

```python
# Countours Detection
filter_value = 2; threshold_level = 0; mode = cv2.RETR_EXTERNAL # _LIST _EXTERNAL _CCOMP _TREE

mask_8bit = np.uint8( np.where(labels == filter_value, 1 , 0) )
print(mask_8bit)
_, binarized = cv2.threshold(mask_8bit, threshold_level, 255, cv2.THRESH_BINARY)
contours, hierarchy = cv2.findContours(binarized, mode, cv2.CHAIN_APPROX_SIMPLE)

# Drawing contours
countourIdx=255; color = (0,255,0); thickness = 3
img_show = cv2.drawContours(img_orig, contours, -1, (0, 255, 0), 1) 
plt.imshow(img_show)
plt.show()
```

<div align="center">
<img src="https://github.com/hoat23/VisionArtificialAndImageProcessing/blob/master/img/img_06_contours.png" width="460" align="center"/>
</div>

More info: https://docs.opencv.org/3.4/dd/d49/tutorial_py_contour_features.html

# 2D convolution

## Image Smoothing

Like one-dimensional signals, images can also be filtered with various types of filters, such as low pass filters (FPB), high pass filters (FPA), band pass filters, etc. While an FPB helps to eliminate noise in the image or blur the image, an FPA helps to find the edges in an image.
The cv2.filter2D () function, available in OpenCV, allows to apply a convolution between a given kernel and an image. An example of a kernel is an averaging filter, like the 5x5 FPB shown below:

<img src="https://www.codecogs.com/eqnedit.php?latex=K=\frac{1}{25}\begin{bmatrix}&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;\\&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;\\&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;\\&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;\\&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;\end{bmatrix}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?K=\frac{1}{25}\begin{bmatrix}&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;\\&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;\\&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;\\&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;\\&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;\end{bmatrix}" title="Matrix" alt="centered image"/>


# Local Scale-Invariant Features

- Step 1: Compute SIFT descriptors using your favorite SIFT library.
- Step 2: L1-normalize each SIFT vector.
- Step 3: Take the square root of each element in the SIFT vector. Then the vectors are L2 normalized

## Code

### Clase RootSIFT (rootsift.py)
```python
# import the necessary packages
import numpy as np
import cv2
class RootSIFT:
	def __init__(self):
		# initialize the SIFT feature extractor
		self.extractor = cv2.DescriptorExtractor_create("SIFT")
	def compute(self, image, kps, eps=1e-7):
		# compute SIFT descriptors
		(kps, descs) = self.extractor.compute(image, kps)
		# if there are no keypoints or descriptors, return an empty tuple
		if len(kps) == 0:
			return ([], None)
		# apply the Hellinger kernel by first L1-normalizing and taking the
		# square-root
		descs /= (descs.sum(axis=1, keepdims=True) + eps)
		descs = np.sqrt(descs)
		#descs /= (np.linalg.norm(descs, axis=1, ord=2) + eps)
		# return a tuple of the keypoints and descriptors
		return (kps, descs)
```

### Runing
```python
# import the necessary packages
from rootsift import RootSIFT
import cv2
# load the image we are going to extract descriptors from and convert
# it to grayscale
image = cv2.imread("example.png")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# detect Difference of Gaussian keypoints in the image
detector = cv2.FeatureDetector_create("SIFT")
kps = detector.detect(gray)
# extract normal SIFT descriptors
extractor = cv2.DescriptorExtractor_create("SIFT")
(kps, descs) = extractor.compute(gray, kps)
print "SIFT: kps=%d, descriptors=%s " % (len(kps), descs.shape)
# extract RootSIFT descriptors
rs = RootSIFT()
(kps, descs) = rs.compute(gray, kps)
print "RootSIFT: kps=%d, descriptors=%s " % (len(kps), descs.shape)
```
Reference: 
- https://www.cs.ubc.ca/~lowe/papers/iccv99.pdf
- https://www.robots.ox.ac.uk/~vgg/publications/2012/Arandjelovic12/arandjelovic12.pdf
- https://www.pyimagesearch.com/2015/04/13/implementing-rootsift-in-python-and-opencv/

# Useful Formules 

### L1 Normalize 

<img src="https://latex.codecogs.com/svg.latex?\Large&space;S=\sum_{i=1}^{n}%20{\mid}y_i-f(x_i){\mid}" title="L1 normalize" alt="centered image"/>

### L2 Normalize

<img src="https://latex.codecogs.com/svg.latex?\Large&space;S=\sum_{i=1}^{n}%20(y_i-f(x_i))^2" title="L2 normalize" alt="centered image"/>


# Utilities 

Loading code in memory from github url

```python
from urllib.request import urlopen

def load_code_from_url(url_path):
  code_str = urlopen(url_path).read()
  code_str = code_str.decode('utf-8')
  exec(code_str)
  return code_str

# This is util when are using Jupyter notebook.
url_code_github = "https://raw.githubusercontent.com/hoat23/VisionArtificialAndImageProcessing/master/bin/utils_imgprocessing.py"
exec( load_code_from_url(url_code_github) )
```

More info: https://www.w3resource.com/python/built-in-function/compile.php
