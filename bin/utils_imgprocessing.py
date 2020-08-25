# coding: utf-8
# Developer: Deiner Zapata Silva.
# Date: 13/07/2020
# Last update: 25/08/2020
# Description: Basic Algorithms by image processing
#######################################################################################
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from urllib.request import urlopen
#######################################################################################
##########################   Global Variables  ########################################
directory_work = None
#######################################################################################
def set_directory_work(directory):
  global directory_work
  directory_work = directory
  print("set_directory_work | [{}]".format(directory_work))
  return

def load_image_from_url(url, readFlag=cv2.IMREAD_COLOR):
    # download the image, convert it to a NumPy array, and then read
    # it into OpenCV format
    resp = urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, readFlag)
    return image

def read_image(namefile, directory='/images/CUOX_XML'):
  global directory_work
  img_json = {
      "name": namefile,
      "directory": directory,
  }

  try:
    os.chdir(directory_work+directory)
    #path_image = directory+"/"+namefile
    img = cv2.imread(namefile) #cv2.COLOR_BGR2RGB
    height = img.shape[0]
    width = img.shape[1]
    channels = img.shape[2]
    tmp = {
        "height": height,
        "width": width,
        "channels": channels,
        "size": int(width*height),
        "img": img
    }
    img_json.update(tmp)
    flag_error = False
  except:
    print("read_image | Error reading {}".format(namefile))
    flag_error = True
  finally:
    img_json.update({'error': flag_error})
    os.chdir(directory_work)
  return img_json

def histogram(img, color=('r','g','b'),scale=1):
  list_histr = []
  for i,col in enumerate(color):
      histr = cv2.calcHist([img],[i],None,[256],[0,256])/scale
      list_histr.append(histr)
  return list_histr
#######################################################################################
def plt_render(render=True):
    if render:
       plt.show()
def create_subplot(fig, rows, columns, i, img, title=None,render=False):
    fig.add_subplot(rows, columns, i)
    plot_img(img, title=title,render=render)

def plot_histogram(list_histr,color=('r','g','b'),title=None,render=True):
    for i,col in enumerate(color):
      plt.plot(list_histr[i],color = col)
      plt.xlim([0,256]) #plt.ylim([0,1])
    plt_render(render=render)

def plot_img(img, title=None, cmap=None, divison=50.0, usefigure=False, render=True):
  if img.ndim == 3:
     img = img[:,:,::-1]
  if usefigure:
     width = img.shape[1] / division
     height = img.shape[0] / division
     f = plt.figure(figsize=(width, height))
  plt.title(title)
  plt_render(render=render) #cmap="gray"

def plot_list_img(list_img, rows=1, cols=2, axis='off', render=True):
  size = rows * cols 
  if size < len(list_img):
    rows = int(len(list_img)/cols)
  
  for num, img in enumerate(list_img):
    title = "Img{0:02d}".format(num)
    plt.subplot(rows,cols,num+1)
    plt.axis(axis)
    plot_img(img, title=title, render=render)
#######################################################################################
	
"""
def get_list_img(directory):
  global directory_work
  os.chdir(directory_work+directory)
  list_img = !ls -1 *.jpg
  os.chdir(directory_work)
  return list_img
"""
def get_mask_by_color(img, color_filter):
  hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
  #np.array([0,120,70]) np.array([10,255,255])
  #"lower": np.array([170,120,70]),"upper": np.array([180,255,255])
  range = {
    "red": {
      "lower": np.array([0,120,70]),
      "upper": np.array([10,255,255])
    },
    "blue":{
      "lower": np.array([60, 100, 100]),
      "upper": np.array([70, 100, 100])     
    },
    "yellow": {
      "lower": np.array([20, 100, 100]),
      "upper": np.array([30, 255, 255])
    }
  }
  # preparing the mask to overlay
  upper = range[color_filter]['upper']
  lower = range[color_filter]['lower'] 
  return cv2.inRange(hsv, lower, upper )

def adjust_gamma(img, gamma=1.0):
	# build a lookup table mapping the pixel values [0, 255] to
	# their adjusted gamma values
  gamma = gamma if gamma > 0 else 0.1
  invGamma = 1.0 / gamma
  table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
  # apply gamma correction using the lookup table
  return cv2.LUT(img, table)

def test_multigamma_value(img_original , lower=0.0, upper=3.5, step=0.5):
  # loop over various values of gamma
  for gamma in np.arange(lower, upper, step):
    # ignore when gamma is 1 (there will be no change to the image)
    if gamma == 1:
      continue
    # apply gamma correction and show the images
    adjusted = adjust_gamma(img_original, gamma=gamma)
    cv2.putText(adjusted, "g={}".format(gamma), (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
    #cv2_imshow(np.hstack([adjusted]))
  return

def converting_color(img_src, code=cv2.COLOR_BGR2RGB):
  # cv2.COLOR_BGR2RGB  | cv2.COLOR_RGB2BGR
  # im_rgb = im_bgr[:, :, [2, 1, 0]]
  # im_rgb = im_bgr[:, :, ::-1]
  img_out = cv2.cvtColor(img_src, code)
  return img_out

#######################################################################################
def filter_using_matrix(filter_value, matrix_orig, labels, fill_value = np.nan):
    mask_filter = np.where(labels == filter_value, labels, fill_value)
    matrix_filter = np.where( mask_filter == filter_value , matrix_orig, mask_filter)
    return matrix_filter

def filter_label(img, labels, filter_value, fill_value = np.nan):
    b,g,r = cv2.split(img)
    b_ = filter_using_matrix(filter_value, b, labels, fill_value = fill_value)
    g_ = filter_using_matrix(filter_value, g, labels, fill_value = fill_value)
    r_ = filter_using_matrix(filter_value, r, labels, fill_value = fill_value)
    img = cv2.merge((b_,g_,r_))
    return img

def split_labels(img, labels, saveimg=False, fill_value = np.nan):
    list_labels = np.unique(labels)
    list_segments = []
    for filter_value in list_labels:
        tmp_img = filter_label(img, labels, filter_value, fill_value = fill_value)
        if saveimg:
            cv2.imwrite("label_{0:02d}.jpg".format(filter_value), tmp_img)
        list_segments.append(tmp_img)
    return list_segments
#######################################################################################
def grab_contours(cnts):
    # if the length the contours tuple returned by cv2.findContours
    # is '2' then we are using either OpenCV v2.4, v4-beta, or
    # v4-official
    if len(cnts) == 2:
        cnts = cnts[0]

    # if the length of the contours tuple is '3' then we are using
    # either OpenCV v3, v4-pre, or v4-alpha
    elif len(cnts) == 3:
        cnts = cnts[1]

    # otherwise OpenCV has changed their cv2.findContours return
    # signature yet again and I have no idea WTH is going on
    else:
        raise Exception(("Contours tuple must have length 2 or 3, "
            "otherwise OpenCV changed their cv2.findContours return "
            "signature yet again. Refer to OpenCV's documentation "
            "in that case"))

    # return the actual contours array
    return cnts

def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)

    # return the edged image
    return edged

def skeletonize(image, size, structuring=cv2.MORPH_RECT):
    # determine the area (i.e. total number of pixels in the image),
    # initialize the output skeletonized image, and construct the
    # morphological structuring element
    area = image.shape[0] * image.shape[1]
    skeleton = np.zeros(image.shape, dtype="uint8")
    elem = cv2.getStructuringElement(structuring, size)

    # keep looping until the erosions remove all pixels from the
    # image
    while True:
        # erode and dilate the image using the structuring element
        eroded = cv2.erode(image, elem)
        temp = cv2.dilate(eroded, elem)

        # subtract the temporary image from the original, eroded
        # image, then take the bitwise 'or' between the skeleton
        # and the temporary image
        temp = cv2.subtract(image, temp)
        skeleton = cv2.bitwise_or(skeleton, temp)
        image = eroded.copy()

        # if there are no more 'white' pixels in the image, then
        # break from the loop
        if area == area - cv2.countNonZero(image):
            break

    # return the skeletonized image
    return skeleton
