import cv2 as cv
import numpy as np

def autocrop(img, K=3): 
    """
        Args:
            img : input image
            K : number of clusters to group pixels
    """

    # convert to grayscale
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # resize to speedup
    img = cv.resize(img, None, fx=0.2, fy=0.2)
    # blur the image five times
    img = cv.GaussianBlur(img, (5,5), 0)
    
    # group pixels
    pixels = np.float32(img.reshape(-1)) # reshape the array so that we get each 3-channel pixel, then convert to float
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0) # define criteria
    _, groups, centroids = cv.kmeans(pixels, K, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)
    centroids = np.uint8(centroids) # convert back the centroids to 8 bit data type
    reassign_pixels = centroids[groups.flatten()] # reassign pixels to centroid values
    grp_img = reassign_pixels.reshape((img.shape))
    
    # extract only hi density areas
    hi_only = np.where(grp_img == min(centroids), 255, 0).astype(np.uint8)
    kernel = np.ones((3,3),np.uint8)
    contours, _ = cv.findContours(hi_only, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    bboxes = []
    for c in contours:
        x, y, w, h = cv.boundingRect(c)
        bboxes.append([x, y, x+w, y+h])
    bboxes = np.asarray(bboxes).T
    # get the enclosing bounding box (rescale)
    x, y, w, h = min(bboxes[0])*5, min(bboxes[1])*5, max(bboxes[2])*5, max(bboxes[3])*5
    
    return  x, y, w-x, h-y # x, y, w, h