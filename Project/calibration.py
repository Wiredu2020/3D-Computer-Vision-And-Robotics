### Necessary Packages
import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import least_squares
from scipy.linalg import rq


def auto_detect_corners(img, pattern_size=(6, 8)):
    """
    Try cv2.findChessboardCorners to get Nx2 image points automatically.
    Adjust pattern_size to match your printed template.
    """
    criteria = (cv2.TermCriteria_EPS + cv2.TermCriteria_MAX_ITER, 30, 0.001)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, pattern_size)
    if not ret:
        raise RuntimeError("Chessboard corners not detected. Try a different pattern_size or image.")
    else:
        # Refine the corner positions
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    #corners = corners.squeeze().T  # 2×N
    return corners2.squeeze().T

def make_world_grid(pattern_size, square_size):
    """ pattern_size = (cols, rows) of inner corners,
      square_size in your chosen unit (e.g. mm). """
    cols, rows = pattern_size
    objp = np.zeros((cols*rows, 3), dtype=np.float32)
    # x = column index * square_size, y = row index * square_size
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_size
    return objp.T

def select_points_single_image(img, scale_factor=0.25):
    """Function to select points from a single image.
    
    Click to select points. Press Esc or close the window when done.
    """
    selected_points = []

    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # Convert to original scale
            original_x = int(x / scale_factor)
            original_y = int(y / scale_factor)
            selected_points.append((original_x, original_y))

            # Draw the point
            cv2.circle(param, (x, y), 3, (0, 255, 0), -1)
            cv2.imshow("Image", param)

    # Check if grayscale or color
    if len(img.shape) == 2 or img.shape[2] == 1:
        img_display = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        img_display = img.copy()

    height, width = img_display.shape[:2]
    new_height, new_width = int(height * scale_factor), int(width * scale_factor)
    resized_img = cv2.resize(img_display, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

    cv2.imshow("Image", resized_img)
    cv2.setMouseCallback("Image", click_event, resized_img)

    while True:
        key = cv2.waitKey(20) & 0xFF
        if key == 27:
            break
        if cv2.getWindowProperty("Image", cv2.WND_PROP_VISIBLE) < 1:
            break
    cv2.destroyAllWindows()

    plt.figure(figsize=(10, 6))
    plt.imshow(cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.tight_layout()
    plt.show()
    print(selected_points)
    return selected_points


def reproj_residuals(P, point3d, point2d):
    P = P.reshape(3,4)
    #  perform calibration and apply it to the 3D points
    points3d_homog = np.hstack((point3d, np.ones(( point3d.shape[0],1))))
    projected_points = P @ points3d_homog.T
    projected_points = projected_points[:2] / projected_points[2]# Normalize
    pro_2d = projected_points.T
    return  np.linalg.norm(point2d - pro_2d, axis=1)

def DLT(points2d, points3d, Norm = False):
    # Number of points
    if Norm == False:
        points2d = points2d.T
        points3d = points3d.T
    N = points3d.shape[0]

    # Construct the A matrix
    A = []
    for i in range(N):
        X, Y, Z = points3d[i, :]  
        u, v = points2d[i, :]

        A.append([X, Y, Z, 1, 0, 0, 0, 0, -u*X, -u*Y, -u*Z, -u])
        A.append([0, 0, 0, 0, X, Y, Z, 1, -v*X, -v*Y, -v*Z, -v])

    A = np.array(A)

    # Solve for M using SVD
    U, S, Vt = np.linalg.svd(A)
    # Last row of Vt gives solution
    M = Vt[-1, :].reshape(3, 4)  

    
    return M

def my_cal(points2d,points3d):
    p_2d = np.hstack((points2d, np.ones((points2d.shape[0], 1)))) 
    p_3d = np.hstack((points3d, np.ones((points3d.shape[0], 1))))  

    mean = np.mean(points2d, axis=0)  
    std = np.std(points2d, axis=0) 
    mean_3d = np.mean(points3d, axis=0)  
    std_3d = np.std(points3d, axis=0)  

    T_2d = np.array([
        [np.sqrt(2) / std[0], 0, -np.sqrt(2) / std[0] * mean[0]],
        [0, np.sqrt(2) / std[1], -np.sqrt(2) / std[1] * mean[1]],
        [0, 0, 1]
    ])
    T_3d = np.array([
        [np.sqrt(3) / std_3d[0], 0, 0, -np.sqrt(3) / std_3d[0] * mean_3d[0]],
        [0, np.sqrt(3) / std_3d[1], 0, -np.sqrt(3) / std_3d[1] * mean_3d[1]],
        [0, 0, np.sqrt(3) / (std_3d[2]+0.0001), -np.sqrt(3) / (std_3d[2]+0.0001) * (mean_3d[2]+0.0001)],
        [0, 0, 0, 1]
    ])

    normalized_2d_pts =  (T_2d @ p_2d.T).T[:, :2]  
    normalized_3d_pts = (T_3d @ p_3d.T).T[:, :3] 
    M = DLT(normalized_2d_pts, normalized_3d_pts, Norm= True)
    DM = np.linalg.inv(T_2d) @ M @ T_3d 
    # initial guess x0 = P0.flatten()
    DM = DM .flatten()
    res = least_squares(reproj_residuals, DM, args=(points3d, points2d),method='lm')   # Levenberg–Marquardt or Gauss–Newton
    P_opt = res.x.reshape(3,4)

    return  P_opt

def decompose_projection(M):
    # TODO implement the decomposition
    K, R = rq(M[:, :3]) 


    # rq decomposition can throw a weird result, this make sure that the result is valid for our purposes
    R = R * np.sign(K[-1,-1])
    K = K * np.sign(K[-1,-1])
    t = np.linalg.inv(K)@M[:, 3]
    
    #C = np.linalg.solve(M[:, :3], -M[:, 3]) 
    return K, R,t



def bp(u,v,K,R,t,z=0):
    x_h = np.linalg.inv(K)@np.array([u,v,1])
    R_inv = R.T
    num = z + (R_inv@t)[2]
    denom = (R_inv @ x_h)[2]

    alpha = num/denom

    Xc = alpha*x_h
    XW = R_inv@ (Xc-t)
    return XW

def project_2d23d(POSITIONS,cubes,donuts,a,b,c):
    robot3d = bp(POSITIONS[0][0][0],POSITIONS[0][1][0],a,b,c,z=9)
    robot_arm = bp(POSITIONS[1][0][0],POSITIONS[1][1][0],a,b,c,z=8.5)
    b_cube3d = bp(cubes[0][0],cubes[0][1],a,b,c,z=4)
    g_cube3d = bp(cubes[1][0],cubes[1][1],a,b,c,z=4)
    r_cube3d = bp(cubes[2][0],cubes[2][1],a,b,c,z=4)
    b_tar3d = bp(donuts[0][0],donuts[0][1],a,b,c,z=0)
    g_tar3d = bp(donuts[1][0],donuts[1][1],a,b,c,z=0)
    r_tar3d = bp(donuts[2][0],donuts[2][1],a,b,c,z=0)

    projection ={
        "robot" : robot3d,
        "arm" : robot_arm,
        "blue_cube" : b_cube3d,
        "red_cube" : r_cube3d,
        "green_cube" : g_cube3d,
        "blue_target" : b_tar3d ,
        "green_target" : g_tar3d,
        "red_target" : r_tar3d 
    }
    return projection
if __name__ == "__main__":
    pass