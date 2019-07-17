import cv2
import numpy as np

def lineSegmentation(img, sigmaY):
    ''' line segmentation '''
    img = 255 - img
    Py = np.sum(img, axis=1)

    y = np.arange(img.shape[0])
    expTerm = np.exp(-y**2 / (2*sigmaY**2))
    yTerm = 1 / (np.sqrt(2*np.pi) * sigmaY)
    Gy = yTerm * expTerm

    Py_derivative = np.convolve(Py, Gy)
    thres = np.max(Py_derivative) // 2
    # find local maximum
    res = (np.diff(np.sign(np.diff(Py_derivative))) < 0).nonzero()[0] + 1

    lines = []
    for idx in res:
        if Py_derivative[idx] >= thres:
            lines.append(idx)
    return lines


def wordSegmentation(img, kernelSize, sigma, theta, minArea=0):
    ''' word segmentation '''
    sigma_X = sigma
    sigma_Y = sigma * theta
    # use gaussian blur and applies threshold
    imgFiltered = cv2.GaussianBlur(img, (kernelSize, kernelSize), sigmaX=sigma_X, sigmaY=sigma_Y)
    _, imgThres = cv2.threshold(imgFiltered, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    imgThres = 255 - imgThres
    # find connected components
    components, _ = cv2.findContours(imgThres, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    lines = lineSegmentation(img, sigma)

    items = []
    for c in components:
        # skip small word candidates
        if cv2.contourArea(c) < minArea:
            continue
        # append bounding box and image of word to items list
        currBox = cv2.boundingRect(c)
        (x, y, w, h) = currBox
        currImg = img[y:y+h, x:x+w]
        items.append([currBox, currImg])
    
    result = []
    for line in lines:
        temp = []
        for currBox, currImg in items:
            if currBox[1] < line:
                temp.append([currBox, currImg])
        for element in temp:
            items.remove(element)
        # list of words, sorted by x-coordinate
        result.append(sorted(temp, key=lambda entry: entry[0][0]))
    return result

def prepareImg(img, height):
    ''' convert given image to grayscale image (if needed) and resize to desired height '''
    assert img.ndim in (2, 3)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h = img.shape[0]
    factor = height / h
    return cv2.resize(img, dsize=None, fx=factor, fy=factor)