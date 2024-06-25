'''
Generate an ArUco tag using OpenCV
Change the TAG ID to get different tags 
'''


import cv2 as cv

TAG_ID   = 0    # ID of the tag you want to generate
TAG_SIZE = 200  # Size of the tag image

aruco_dict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_6X6_250)
tag_image  = cv.aruco.generateImageMarker(aruco_dict, TAG_ID, TAG_SIZE)

cv.imshow("ArUco Tag", tag_image)
cv.waitKey(0)
cv.destroyAllWindows()

cv.imwrite(f"aruco_tag_{TAG_ID}.png", tag_image)