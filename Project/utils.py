### Necessary Packages
import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage.measure import label, regionprops
from scipy.optimize import least_squares
from scipy.linalg import rq
import math


#################################################################
# color_ranges = {
#         'red':    [([0, 150, 100], [10, 255, 255]), ([160, 150, 100], [179, 255, 255])],
#         'green':  [([40, 80, 50], [80, 255, 255])],  # More specific green range
#         'blue':   [([90, 50, 50], [130, 255, 255])],
#         'magenta':[([140, 50, 50], [170, 255, 255])],
#         'yellow': [([20, 100, 100], [30, 255, 255])]
#     }
HSV_RANGES = {
    'blue': (np.array([90, 50, 50]), np.array([130, 255, 255])),
    'green': (np.array([40, 50, 50]), np.array([80, 255, 255])),
    'red1': (np.array([0, 70, 50]), np.array([10, 255, 255])),   
    'red2': (np.array([170, 70, 50]), np.array([180, 255, 255]))
}

# ROBOT LOCALIZATION
###############################################
def locate_robot(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Purple (magenta) HSV range
    lower_purple = np.array([125, 50, 50])
    upper_purple = np.array([155, 255, 255])
    mask_purple = cv2.inRange(hsv, lower_purple, upper_purple)

    # Yellow HSV range
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([35, 255, 255])
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # Locate purple
    labeled_purple = label(mask_purple)
    props_purple = regionprops(labeled_purple)
    purple_center = np.array([[np.nan], [np.nan]])

    if len(props_purple) > 0:
        max_purple = max(props_purple, key=lambda x: x.area)
        y_p, x_p = max_purple.centroid
        purple_center = np.array([[x_p], [y_p]])

    # Locate yellow
    labeled_yellow = label(mask_yellow)
    props_yellow = regionprops(labeled_yellow)
    yellow_center = np.array([[np.nan], [np.nan]])

    if len(props_yellow) > 0:
        max_yellow = max(props_yellow, key=lambda x: x.area)
        y_y, x_y = max_yellow.centroid
        yellow_center = np.array([[x_y], [y_y]])

    return purple_center, yellow_center

#################################################################

def locate_cubes(img, lower_hsv, upper_hsv):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
    
    # Fill holes in the mask (imfill operation)
    # We use a morphological closing operation, which dilates and then erodes to fill small holes
    kernel = np.ones((3,2), np.uint8)  # You can adjust the kernel size
    mask_filled = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Label connected components
    labeled = label(mask_filled)
    props = regionprops(labeled)

    # Check if there are any regions detected
    if len(props) == 0:
        return None  

    # Choose the largest blob based on area
    largest_blob = max(props, key=lambda x: x.area)
    y, x = largest_blob.centroid
    return (x, y)

def locate_all_cubes(img):
    # Locate largest blue cube
    blue_location = locate_cubes(img, *HSV_RANGES['blue'])
    
    # Locate largest green cube
    green_location = locate_cubes(img, *HSV_RANGES['green'])
    
    # Locate largest red cube (two ranges combined)
    red_location = locate_cubes(img, *HSV_RANGES['red1']) or locate_cubes(img, *HSV_RANGES['red2'])

    return blue_location, green_location, red_location
############################################################################


# Locating target location
def locate_target_location(img, lower_hsv, upper_hsv):

    # Check if image is loaded properly
    if img is None:
        print(f"Error: Image {img} not loaded properly.")
        return None
    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Convert HSV boundaries to numpy arrays
    lower_hsv = np.array(lower_hsv, dtype=np.uint8)
    upper_hsv = np.array(upper_hsv, dtype=np.uint8)

    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)

    # Fill holes in the mask (imfill operation)
    kernel = np.ones((3,4), np.uint8)
    mask_filled = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Label connected components
    labeled = label(mask_filled)
    props = regionprops(labeled)

    if len(props) == 0:
        return None

    sorted_props = sorted(props, key=lambda x: x.area, reverse=True)

    if len(sorted_props) < 2:
        print("Not enough blobs to find the second largest.")
        return None

    second_largest_blob = sorted_props[1]
    y, x = second_largest_blob.centroid

    return (x, y)

def locate_donuts(img):
    # Locate second largest blue donut
    blue_donut = locate_target_location(img, *HSV_RANGES['blue'])
    
    # Locate second largest green cube
    green_donut = locate_target_location(img, *HSV_RANGES['green'])
    
    # Locate second largest red cube (two ranges combined)
    red_donut = locate_target_location(img, *HSV_RANGES['red1']) or locate_target_location(img, *HSV_RANGES['red2'])

    return blue_donut, green_donut, red_donut

def command(robot_arm2d,robot2d, cube2d, tar2d):
    robot2d = robot2d[:2]
    robot_arm2d = robot_arm2d[:2]
    cube2d = cube2d[:2]
    tar2d = tar2d[:2]
    offset = 2
    v0 = robot_arm2d - robot2d
    angle1_rad = math.atan2(v0[1], v0[0])
    off_deg = math.degrees(angle1_rad)-offset
    # Vector robot → g_cube
    v1 = cube2d - robot2d
    angle1_rad = math.atan2(v1[1], v1[0])
    angle1_deg = math.degrees(angle1_rad)
    distance1 = np.linalg.norm(v1)

    # Vector g_cube → g_target
    v2 = tar2d - cube2d
    angle2_rad = math.atan2(v2[1], v2[0])
    angle2_deg = math.degrees(angle2_rad)
    distance2 = np.linalg.norm(v2)-11

    # Relative turn angle from v1 to v2
    angle_turn_deg = angle2_deg - angle1_deg 
    # Normalize angle to [-180, 180]
    angle_turn_deg = (angle_turn_deg + 180) % 360 - 180

    firtdeg = angle1_deg-off_deg
    # Results
    print(f"turn({firtdeg:.2f}); go({distance1:.2f}); grab(); turn({angle_turn_deg:.2f}); go({distance2:.2f}); let_go()")


def swap(cube, donut, color="blue"):
    if color == "blue":
        cube_0 = (donut[0], cube[1], cube[2])
        donut_0 = (cube[0], donut[1], donut[2])

    elif color == "green":
        cube_0 = (cube[0], donut[1], cube[2])
        donut_0 = (donut[0], cube[1], donut[2])

    else:
        cube_0 = (cube[0], cube[1], donut[2])
        donut_0 = (donut[0], donut[1], cube[2])

    cube = cube_0
    donut = donut_0
    return cube, donut
if __name__ == "__main__":
    pass