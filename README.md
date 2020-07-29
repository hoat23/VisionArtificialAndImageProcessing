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
