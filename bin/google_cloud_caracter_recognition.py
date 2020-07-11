# coding: utf-8
# Developer: Deiner Zapata Silva.
# Date: 10/07/2020
# Description: using GoogleCloudPlatform by caracter recognition.
#########################################################################################
#pip install -U --user mss        # mss-5.1.0 # https://python-mss.readthedocs.io/examples.html
#pip install google-cloud-vision  # google-cloud-0.34.0
#pip install pyasn1 --upgrade     #pyasn1<0.5.0,>=0.4.6, but you'll have pyasn1 0.4.4 which is incompatible.
#pip install setuptools --upgrade #setuptools>=40.3.0, but you'll have setuptools 39.1.0 which is incompatible.
#########################################################################################
import os
import io
from google.cloud import vision
from mss.windows import MSS as mss
#########################################################################################
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r'hoat23-1549832222837-435b3bf94637.json'
client = vision.ImageAnnotatorClient()
#########################################################################################

def detect_text(path):
    """Detects text in the file."""
    with io.open(path, 'rb') as image_file:
        content = image_file.read()

    image = vision.types.Image(content=content)

    response = client.text_detection(image=image)
    texts = response.text_annotations
    print('Texts:')

    for text in texts:
        print('\n"{}"'.format(text.description))

        vertices = (['({},{})'.format(vertex.x, vertex.y)
                    for vertex in text.bounding_poly.vertices])

        print('bounds: {}'.format(','.join(vertices)))

    if response.error.message:
        raise Exception(
            '{}\nFor more info on error messages, check: '
            'https://cloud.google.com/apis/design/errors'.format(
                response.error.message))

if __name__ == "__main__":
    print("INFO |google_cloud_vision |")
    with mss() as sct:
        filename = sct.shot(mon=-1, output='fullscreen.png')
        detect_text(filename)
    #detect_text()
    pass

