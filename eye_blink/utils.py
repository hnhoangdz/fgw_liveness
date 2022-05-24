from scipy.spatial import distance as dist

def eye_aspect_ratio(eye):
    """Calculate eye aspect ratio

    Args:
        eye (list): six coordinates of eye

    Returns:
        float: eye aspect ratio (0.15 - 0.4)
    """
    A = dist.euclidean(eye[1],eye[5])
    B = dist.euclidean(eye[2],eye[4])
    C = dist.euclidean(eye[0],eye[3])
    ear = (A + B) / (2.0 * C)
    return ear