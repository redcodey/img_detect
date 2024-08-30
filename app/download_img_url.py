import cv2
from urllib.request import urlopen
import numpy as np
import detect

try:
    image_url = url  = 'https://webhub.acacy.com.vn/storage/images/view?guid=241E578DF36B96C4ACFE48E6DA0133E5'
    resp = urlopen(image_url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR) # The image object

    # Optional: For testing & viewing the image
    cv2.imshow('image',image)
    
    key = cv2.waitKey(0)
except Exception as ex:
    print('ex: ', ex)