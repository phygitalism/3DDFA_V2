"""Description of landmarks
https://ibug.doc.ic.ac.uk/resources/300-W/

Each dictionary contains name of part face and indices of landmarks. 
Bounds are half interval: [start_index, end_index)

All indices are zero-based.
"""

LEFT_EYEBROW = {
    "name": "left_eyebrow",
    "index_bounds": (17, 22)
}

RIGHT_EYEBROW = {
    "name": "right_eyebrow",
    "index_bounds": (22, 27)
}

LEFT_EYE = {
    "name": "left_eye",
    "index_bounds": (36, 42)
}

RIGHT_EYE = {
    "name": "right_eye",
    "index_bounds": (42, 48)
}

NOSE = {
    "name":  "nose",
    "index_bounds": (27, 36)
}

TOP_MOUTH = {
    "name": "top_mouth",
    "index_bounds": (48, 55)
}

BOTTOM_MOUTH = {
    "name": "bottom_mouth",
    "index_bounds": (55, 60)
}

TOP_LEAP = {
    "name": "top_leap",
    "index_bounds": (60, 65)
}

BOTTOM_LEAP = {
    "name": "bottom_leap",
    "index_bounds": (65, 68)
}

FACE_BOUND = {
    "name": "face_bound",
    "index_bounds": (0, 17)
}


