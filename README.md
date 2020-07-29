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

# Utilities 

Loading code in memory from github url

```
def load_code_from_url(url_path):
  code_str = urlopen(url_path).read()
  code_str = code_str.decode('utf-8')
  exec(code_str)
  return code_str
```

More info: https://www.w3resource.com/python/built-in-function/compile.php
