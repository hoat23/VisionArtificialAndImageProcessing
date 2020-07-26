# coding: utf-8
# Developer: Deiner Zapata Silva.
# Date: 13/07/2020
# Last update: 13/07/2020
# Description: Basic Algorithms by image processing
#######################################################################################
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
#######################################################################################
directory_work = None

def set_directory_work(directory):
  global directory_work
  directory_work = directory
  print("set_directory_work | [{}]".format(directory_work))
  return

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

def plot_histogram(list_histr,color=('r','g','b')):
    for i,col in enumerate(color):
      plt.plot(list_histr[i],color = col)
      plt.xlim([0,256])
      #plt.ylim([0,1])
    plt.show()
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

##############################################################################################################################
def filter_using_matrix(filter_value, matrix_orig, labels, fill_value = np.nan):
    mask_filter = np.where(labels == filter_value, labels, fill_value)
    matrix_filter = np.where( mask_filter == filter_value , matrix_orig, mask_filter)
    return matrix_filter

def filter_label(img, labels, filter_value):
    b_ = filter_using_matrix(filter_value, b, segments1)
    g_ = filter_using_matrix(filter_value, g, segments1)
    r_ = filter_using_matrix(filter_value, r, segments1)
    img = cv2.merge((b_,g_,r_))
    return img

def split_labels(img, labels):
    labels = segments1
    list_labels = np.unique(labels)
    for filter_value in list_labels:
        tmp_img = filter_label(img, labels, filter_value)
        cv2.imwrite("label_{0:02d}.jpg".format(filter_value), tmp_img)
##############################################################################################################################
