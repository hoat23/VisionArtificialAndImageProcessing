# VisionArtificialAndImageProcessing

A image 2x2[px] representation in RGB is:
```
#img = numpy.zeros([2,2,3])
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

Creating a white mask like other image reference.

```
img_black = np.ones_like( img_reference ) * 255
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

# Local Scale-Invariant Features

Reference: 
- https://www.cs.ubc.ca/~lowe/papers/iccv99.pdf
- https://www.robots.ox.ac.uk/~vgg/publications/2012/Arandjelovic12/arandjelovic12.pdf



# Utilities 

Loading code in memory from github url

```
from urllib.request import urlopen

def load_code_from_url(url_path):
  code_str = urlopen(url_path).read()
  code_str = code_str.decode('utf-8')
  exec(code_str)
  return code_str
```

More info: https://www.w3resource.com/python/built-in-function/compile.php
