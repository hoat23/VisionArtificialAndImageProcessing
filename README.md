# VISION ARTIFICIAL AND IMAGE PROCESSING


# Lighting conditions

In general, your lighting conditions should have three primary goals. Let’s review them below.

## High Contrast

Maximize the contrast between the Regions of Interest in your image (objects you want to detect, extract, describe classify, manipulate, etc. should have sufficiently high contrast from the rest of the image so they are easily detectable).
Generalizable

Your lighting conditions should be consistent enough that they work well from one object to the next. If our goal is to identify various United States coins in an image, our lighting conditions should be generalizable enough to facilitate in the coin identification, whether we are examining a penny, nickel, dime, or quarter.
Stable

Having stable, consistent, and repeatable lighting conditions is the holy grail of computer vision application development. However, it’s often hard (if not impossible) to guarantee — this is especially true if we are developing computer vision algorithms that are intended to work in outdoor lighting conditions. As the time of day changes, clouds roll in over the sun, and rain starts to pour, our lighting conditions will obviously change.

## COLOR SPACES
- RGB
- HSV
  - Hue: Which “pure” color we are examining. For example, all shadows and tones of the color “red” will have the same Hue.
  - Saturation: How “white” the color is. A fully saturated color would be pure, as in “pure red.” And a color with zero saturation would be pure white.
  - Value: The Value allows us to control the lightness of our color. A Value of zero would indicate pure black, whereas increasing the value would produce lighter colors.
  - Useful when building applications where we are tracking color of some object in an image.
  - Easier to define valid color range using HSV than it is RGB
- L*a*b:
  - RGB is non-intuitive when defining exact shades of a color or specifyig a particular range of colors
  - HSV color space is more intuitive but does not do the best job in representing how humans see and interpret colors in images.
  - L*a*b goal is to mimic the methodology in which humans see and interpret color.
  - L-channel: Lightness of pixel. This value goes up and doesn the vertical axis, white to black, with neutral grays at the center of the axis.
  - a-channel: Originates from the center of the L-channel and defines pure green on one end of spectrum and pure red on the other.
  - b-channel: Also originates from the center of the L-channel but is perpendicular to the a-channel. The b-channel defines pure blue at one end of spectrum and pure yellow at the other

# Image representation 

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
## Basic Image Descriptors

Extract mean and stdev from each channel
```python
(means, stds) = cv2.meanStdDev(img)
print("num_channels:", img.ndim)
print("size        :", img.shape[:2])
print("means       :", means.flatten())
print("stds        :", stds.flatten())
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

More info: 
- https://docs.opencv.org/3.4/dd/d49/tutorial_py_contour_features.html
- https://www.youtube.com/watch?v=gBn7Ej5iIpI
# 2D Convolution

Like one-dimensional signals, images can also be filtered with various types of filters, such as low pass filters (FPB), high pass filters (FPA), band pass filters, etc. While an FPB helps to eliminate noise in the image or blur the image, an FPA helps to find the edges in an image.
The cv2.filter2D () function, available in OpenCV, allows to apply a convolution between a given kernel and an image. An example of a kernel is an averaging filter, like the 5x5 FPB shown below:

<div align="center">
<img src="https://latex.codecogs.com/svg.latex?\Large&space;K=\frac{1}{25}\begin{bmatrix}&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;\\&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;\\&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;\\&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;\\&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;\end{bmatrix}" title="Matrix" align="center"/>
</div>

Filtering a given image with the above kernel works as follows: a 5 × 5 window is centered on each pixel of the image. The pixels contained in this window are added and divided by 25, and the resulting value is assigned to the pixel.
This is equivalent to calculating the average value of the falling pixels in the 5 × 5 window. The operation is repeated on all the pixels of the image, giving rise to the filtered image. The following code generates the K kernel and applies it to an image:

```python
# Kernel
kernel = np.ones((5,5),np.float32)/25

# Filter the image using the kernel
dst = cv2.filter2D(image,-1,kernel)
```

## Averange
This filter takes the average of all the pixels under the kernel area and replaces the middle element by this average. An alternative way to do this is by using the cv2.blur () or cv2.boxFilter () functions. When using these functions we have to specify the width and height of the kernel. A 3 × 3 normalized box filter would look like this:

<div align="center">
<img src="https://latex.codecogs.com/svg.latex?\Large&space;K=\frac{1}{9}\begin{bmatrix}&space;1&space;&&space;1&space;&&space;1&space;\\&space;1&space;&&space;1&space;&&space;1&space;\\&space;1&space;&&space;1&space;&&space;1&space;\end{bmatrix}" title="Matrix" align="center"/>
</div>

If you don't want to use a normalized box filter, use cv2.boxFilter () and pass the normalize = False argument to the function.

```python
blur = cv2.blur(img,(3,3))
```

## Gaussian Filter
In this approach, instead of a box filter consisting of equal coefficients, a Gaussian kernel is used. This is done with the function, cv2.GaussianBlur (). The width and height of the kernel must be passed as input parameters, which must be positive and odd. In addition, the standard deviation must be specified in the X and Y directions,
sigmaX and sigmaY, respectively. This type of filtering is very effective in removing Gaussian noise from the image.

If only sigmaX is specified, sigmaY is taken as equal to sigmaX. If both are passed as zeros, they are calculated from the kernel size.

```python
blur = cv2.GaussianBlur(img,(5,5),0)
```

<div align="center">
<img src="https://latex.codecogs.com/svg.latex?\Large&space;K=H(u,v)=\frac{1}{16}\begin{bmatrix}&space;1&space;&&space;2&space;&&space;1&space;\\&space;2&space;&&space;4&space;&&space;2&space;\\&space;1&space;&&space;2&space;&&space;1&space;\end{bmatrix}" title="Gaussian Kernel" align="center"/>
</div>

Note: The kernel width must be 6 sigma

<div align="center">
<img src="https://latex.codecogs.com/svg.latex?\Large&space;h(u,v)=\frac{1}{2\pi\sigma^{2}}e^{-\frac{u^{2}+v^{2}}{\sigma^{2}}}" title="Gaussian Kernel" align="center"/>
</div>

## Median Filter
This filter calculates the median of all the pixels under the kernel window and the center pixel is replaced with this median value. This is very effective in eliminating noise known as salt and pepper noise. OpenCV has the cv2.medianBlur () function to apply this type of filter to an image. 

As in the Gaussian filter, he kernel size in the median filter has to be a positive odd integer.

```python
median = cv2.medianBlur(img,5)
```

The median is a robust estimator against outliers.

<div align="center">
<img src="https://github.com/hoat23/VisionArtificialAndImageProcessing/blob/master/img/img_08_deleting_outlier_with_median.png" width="250" align="center"/>
</div>

A example of aplying median filter:

<div align="center">
<img src="https://github.com/hoat23/VisionArtificialAndImageProcessing/blob/master/img/img_09_applying_median_filter.png" width="400" align="center"/>
</div>

Reference: 
- https://unipython.com/suavizando-imagenes-con-opencv/
- http://www.dccia.ua.es/dccia/inf/asignaturas/Vision/vision-tema2.pdf

# Image Pre-Processing

# Image Normalization
The normalization is a process that changes the range of pixel intensity values. 

Normalization transforms an n-dimensional grayscale image 
<div align="center">
<img src="https://latex.codecogs.com/svg.latex?\Large&space;I:\{{\mathbb{X}}\subseteq{\mathbb{R}}^{n}\}\rightarrow\{{\text{Min}},..,{\text{Max}}\}" width="20"/> 
</div>

With intensity values in the range (Min,Max), into a new image:
<div align="center">
<img src="https://latex.codecogs.com/svg.latex?\Large&space;I_{N}:\{{\mathbb{X}}\subseteq%20{\mathbb%20{R}}^{n}\}\rightarrow\{{\text{newMin}},..,{\text{newMax}}\}" width="20"/>
</div>

With intensity values in the range (newMin,newMax).

The linear normalization of a grayscale digital image is performed according to the formula:
<div align="center">
<img src="https://latex.codecogs.com/svg.latex?\Large&space;I_{N}=({{\text{newMax}}-{\text{newMin}}})%20({\frac{{I-{\text{Min}}}%20}{{\text{Max}}-{\text{Min}}}})+{\text{newMin}}"/>
</div>

# Histogram Equalization

<div align="center">
<img src="https://github.com/hoat23/VisionArtificialAndImageProcessing/blob/master/img/histogram_normalization_img01.png" width="400" align="center"/>
</div>

```
img_original = [ [52,	55,	61,	59,	79,	61,	76,	61]
		 [62,	59,	55,	104,	94,	85,	59,	71]
		 [63,	65,	66,	113,	144,	104,	63,	72]
		 [64,	70,	70,	126,	154,	109,	71,	69]
		 [67,	73,	68,	106,	122,	88,	68,	68]
		 [68,	79,	60,	70,	77,	66,	58,	75]
		 [69,	85,	64,	58,	55,	61,	65,	83]
		 [70,	87,	69,	68,	65,	73,	78,	90]]

img_equalized = [[0,	12,	53,	32,	190,	53,	174,	53 ]
		 [57,	32,	12,	227,	219,	202,	32,	154]
		 [65,	85,	93,	239,	251,	227,	65,	158]
		 [73,	146,	146,	247,	255,	235,	154,	130]
		 [97,	166,	117,	231,	243,	210,	117,	117]
		 [117,	190,	36,	146,	178,	93,	20,	170]
		 [130,	202,	73,	20,	12,	53,	85,	194]
		 [146,	206,	130,	117,	85,	166,	182,	215]]
```
# Features Extraction

## Local Scale-Invariant Features

A robust interest detector SIFT is applied which is tweaked with center of mass algorithm which localizes the spliced object and only nearest points are used concentrically with respect to coordinates of center of mass of given image.

- Step 1: Compute SIFT descriptors using your favorite SIFT library.
- Step 2: L1-normalize each SIFT vector.
- Step 3: Take the square root of each element in the SIFT vector. Then the vectors are L2 normalized

### Code

#### Clase RootSIFT (rootsift.py)
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

#### Runing
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
## Zernike moments
zernike will give measure about how the mass is distributed all over image.

## Local binary pattern
Local binary pattern will give measure of how many pixels represent a particular code.

## Haralick Features
Haralick Features which is a combination of feature vector which provides 14 useful statistical features.

- x(1)  Angular Second Moment (Energy).
- x(2)  Contrast.
- x(3)  Correlation.
- x(4)  Variance.
- x(5)  Inverse Difference Moment (Homogeneity).
- x(6)  Sum Average.
- x(7)  Sum Variance.
- x(8)  Sum Entropy.
- x(9)  Entropy.
- x(10) Difference Variance.
- x(11) Difference Entropy.
- x(12) Information Measure of Correlation I.
- x(13) Information Measure of Correlation II.
- x(14) Maximal Correlation Coefficien.

# Clasification  Methodology

- Effective morphology based image filtering techniques are used to reduce the noise and get prominent edge map.
- Final feature vector by applying PCA which reduces dimention to a fixed component and final feature vector is feeded to SVM classifier for training model.
- N-fold cross validation is used to get minimally overfitted and accurate model.

## Reference: 

- https://www.cs.ubc.ca/~lowe/papers/iccv99.pdf
- https://www.robots.ox.ac.uk/~vgg/publications/2012/Arandjelovic12/arandjelovic12.pdf
- https://www.pyimagesearch.com/2015/04/13/implementing-rootsift-in-python-and-opencv/
- https://www.pcigeomatics.com/geomatica-help/references/pciFunction_r/python/P_tex.html

# Usefull Formules 

### L1 Normalize 
<div align="center">
<img src="https://latex.codecogs.com/svg.latex?\Large&space;S=\sum_{i=1}^{n}%20{\mid}y_i-f(x_i){\mid}" title="L1 normalize" align="center"/>
</div>

### L2 Normalize
<div align="center">
<img src="https://latex.codecogs.com/svg.latex?\Large&space;S=\sum_{i=1}^{n}%20(y_i-f(x_i))^2" title="L2 normalize" align="center"/>
</div>

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

# More info
- https://www.w3resource.com/python/built-in-function/compile.php
- http://haralick.org/journals/
- https://www.pcigeomatics.com/geomatica-help/references/pciFunction_r/python/P_tex.html
- https://github.com/wkentaro/labelme/tree/master/examples/tutorial
