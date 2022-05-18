import numpy as np
import cv2

def detect(img, cascade):
    """Detect face/orientation

    Args:
        img (numpy array): gray
        cascade (object): cascade algorithm

    Returns:
        rects: bounding boxes when orientation (xmin, ymin, xmax, ymax)
    """
    rects,_,_ = cascade.detectMultiScale3(img, scaleFactor=1.3, minNeighbors=4, minSize=(30, 30),
                                    flags=cv2.CASCADE_SCALE_IMAGE, outputRejectLevels = True)
    if len(rects) == 0:
        return ()
    rects[:,2:] += rects[:,:2]
    return rects


def convert_rightbox(img, box_right):
    """Generate new box in right side of face
       NOTE: When flipping image, xmin = w-xmax(flipped), xmax = w-xmin(flipped)
    Args:
        img (numpy array): gray
        box_right (numpy array): bounding box when orientation (actual left)

    Returns:
        numpy array: bounding box when orientation right side (xmin, ymin, xmax, ymax)
    """
    res = np.array([])
    h, w = img.shape
    print('box_right: ', box_right)
    for box_ in box_right:
        box = np.copy(box_)
        box[0] = w-box_[2]
        box[2] = w-box_[0]
        res = np.expand_dims(box,axis=0)
    return res

def get_areas(boxes):
    """Calculate all areas of bounding boxes

    Args:
        boxes (numpy array): bounding boxes

    Returns:
        list: corresponding areas
    """
    areas = []
    for box in boxes:
        x0,y0,x1,y1 = box
        area = (y1-y0)*(x1-x0)
        areas.append(area)
    return areas