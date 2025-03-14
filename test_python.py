import torch

#x = torch.tensor([[1,2, -1, 8], [3,4, -1, 12]]).detach().cpu().numpy()
#y = [(e, x[1][index]) for index, e in enumerate(x[0]) if e !=-1 and x[1][index] !=-1]

#print("y = ", y)

"""

def ndc2pixel(v, S):
    return ((v + 1.0) * S - 1.0) * 0.5

def pixel2ndc(pixel, S):
    return (((pixel/0.5)+1.0)/S)-1.0

W = 640
H = 480
full_proj_matrix = torch.tensor([[ 0.9204, -1.3891,  0.5375,  0.5375],
        [ 1.0459,  1.6325,  0.2143,  0.2143],
        [-0.8813,  0.4865,  0.8157,  0.8156],
        [ 2.0387, -2.3517,  4.3125,  4.3221]])

origP = torch.tensor([10,210,62,1]).float()
p_hom = torch.matmul(origP, full_proj_matrix).tolist()
print(p_hom)
p_w = 1/p_hom[3]
p_proj = [p_hom[0]*p_w, p_hom[1]*p_w, p_hom[2]*p_w]
pixel = [ndc2pixel(p_hom[0]*p_w, 640), ndc2pixel(p_hom[1]*p_w, 480)]
print(pixel)

p_proj_back = [pixel2ndc(pixel[0], 640), pixel2ndc(pixel[1], 480), p_proj[2]]
p_hom_back = [p_proj_back[0]/p_w,p_proj_back[1]/p_w,p_proj_back[2]/p_w, 1/p_w]

p_orij_back = torch.matmul(torch.tensor(p_hom_back), torch.linalg.inv(full_proj_matrix))
print(p_orij_back) 

"""
"""
import numpy as np
import cv2

def pixel2ndc(pixel, S):
    return (((pixel/0.5)+1.0)/S)-1.0

def getOrigPoint(point_in_render_img, W, H, ProjMatrix):
    pixel_x, pixel_y, proj_z, p_w = point_in_render_img[1], point_in_render_img[0], point_in_render_img[2], point_in_render_img[3]
    p_proj_x, p_proj_y = pixel2ndc(pixel_x, W), pixel2ndc(pixel_y, H)
    p_hom_x, p_hom_y, p_hom_z = p_proj_x/p_w, p_proj_y/p_w, proj_z/p_w
    p_hom_w = 1/p_w
    p_hom = np.array([p_hom_x, p_hom_y, p_hom_z,p_hom_w])
    origP = np.matmul(p_hom, np.linalg.inv(ProjMatrix), dtype=np.float32)
    origP = origP[:3]
    return origP

def getAllOrigPoints(points_in_render_img, W,H,ProjMatrix):
    match_3d = []
    for piri in points_in_render_img:
        origP = getOrigPoint(piri, W,H,ProjMatrix)
        match_3d.append(origP)
    return match_3d

def calculate_pose_errors(R_gt, t_gt, R_est, t_est):
    # Calculate rotation error
    rotError = np.matmul(R_est.T, R_gt)
    rotError = cv2.Rodrigues(rotError)[0]
    rotError = np.linalg.norm(rotError) * 180 / np.pi

    # Calculate translation error
    transError = np.linalg.norm(t_gt - t_est.squeeze(1)) * 100  # Convert to cm
    
    return rotError, transError
"""
"""
def calculate_pose_errors(R_gt, t_gt, R_est, t_est):
    # Calculate rotation error
    rotError = np.matmul(R_est.T, R_gt)
    rotError = cv2.Rodrigues(rotError)[0]
    rotError = np.linalg.norm(rotError) * 180 / np.pi

    # Calculate translation error
    transError = np.linalg.norm(t_gt - t_est.squeeze(1)) * 100  # Convert to cm
    
    return rotError, transError

import numpy as np
import cv2
#query_point_valid = [[515.,53.], [305.,20.], [320., 77.], [374.,26.], [282., 62.], [365.,49.], [295., 272.], [280.,25.], [457.,267.]]
#query_haha = []
#for q in query_point_valid:
#    query_haha.append([q[1], q[0]])
query_point_valid =  [[492.0, 266.0], [254.0, 204.0], [400.0, 284.0], [456.0, 255.0], [277.0, 400.0], [307.0, 140.0], [275.0, 154.0], [298.0, 
272.0], [491.0, 238.0], [308.0, 155.0], [281.0, 132.0], [458.0, 320.0], [440.0, 473.0], [280.0, 142.0], [340.0, 266.0], [297.0, 142.0], [328.0, 128.0], [282.0, 437.0], [195.0, 293.0], [404.0, 256.0], [487.0, 273.0], [466.0, 237.0], [359.0, 285.0], [127.0, 261.0], [401.0, 434.0], [387.0, 353.0], [272.0, 465.0], [290.0, 257.0], [449.0, 272.0], [292.0, 139.0], [239.0, 213.0], [397.0, 272.0], [165.0, 264.0], [389.0, 159.0], [375.0, 158.0], [425.0, 464.0], [455.0, 272.0], [291.0, 126.0], [401.0, 163.0], [256.0, 183.0], [342.0, 144.0]]





proj_point_valid = [[5.2064564e+01, 5.1916528e+02], [1.9785721e+01, 3.0707730e+02  ], [7.6446777e+01, 3.2093994e+02], [2.5158417e+01, 3.7386636e+02], [5.7595512e+01, 2.8230994e+02], [4.4861385e+01, 3.6531406e+02], [2.6925873e+02, 2.9371579e+02], [2.4051544e+01, 2.8117508e+02], [2.6963049e+02, 4.6225717e+02]]


K_query =  np.array([[522.04683096,   0 ,        320 ],
 [  0,         522.04683096,  240        ],
 [  0,           0,           1        ]]) 

#match_3d = getAllOrigPoints(proj_points_valid, 640,480, fullprojmatrix)
#match_3d = [[-6.395389, 2.8958693, 6.798067], [-4.6355796, -1.200465, 5.0189414], [-4.1403184, -0.3968175, 6.4790726], [-5.6746674, 0.25262165, 8.035915], [-4.018721, -1.2590245, 5.8101525], [-4.709421, -0.2041668, 4.6004915], [-0.41768503, 1.1654953, 6.5677066], [-5.2582793, -1.5145651, 12.147212], [-2.565925, 2.3412042, 2.4039721]]
match_3d = [[4.7405806, 3.293595, 5.292528], [0.14041625, -0.8764599, 6.2472625], [2.5728705, 2.561705, 6.0991397], [3.941768, 2.564368, 5.433526], [-1.9379532, 2.069453, 3.382568], [2.2542543, -1.5568339, 6.6779537], [1.4951009, -1.626111, 7.0295353], [0.5370154, 1.1285802, 7.080181], [5.1695604, 2.682706, 5.541408], [2.1208017, -1.288401, 6.824552], [1.7380794, -2.0086613, 6.8038354], [4.2677107, 5.0167103, 7.067818], [-0.51579285, 3.6515868, 1.4401146], [1.6693182, -1.8603626, 6.875218], [1.6072003, 1.5820603, 7.0881243], [1.9722067, -1.6757356, 6.8012123], [2.7416024, -1.5668403, 6.51291], [-2.1657543, 2.3262138, 2.7033641], [-2.0465147, 0.22342749, 6.0665107], [2.987932, 2.0863383, 6.1960487], [5.1393766, 3.7586155, 6.1587625], [4.7108464, 2.3342998, 5.74658], [1.9702436, 2.3305876, 7.231776], [-2.9850714, -1.0324104, 8.121578], [-0.60646445, 3.1313324, 2.0148704], [0.88200027, 3.3313496, 4.9291806], [-2.629976, 3.099563, 3.476569], [0.67915016, 0.81928015, 7.4867215], [3.445426, 2.6665437, 5.2988734], [1.923647, -1.7792728, 6.732753], [-0.21965013, -0.85873747, 6.229202], [2.6646366, 2.3706431, 6.3187394], [-2.1991663, -0.64891493, 7.0238514], [3.765422, -0.27533454, 6.1613364], [3.5953836, -0.52907395, 6.4917874], [-0.3077639, 4.4321084, 2.5996437], [3.5990467, 2.743534, 5.145132], [2.0143435, -2.0216033, 6.647013], [3.6632192, -0.1310633, 5.5297008], [0.78595465, -1.2857418, 
7.5331135], [2.8801951, -1.1569709, 6.4099984]]
    
_, R, t, _ = cv2.solvePnPRansac(np.array(match_3d), np.array(query_point_valid), 
                                                  K_query, 
                                                  distCoeffs=None, 
                                                  flags=cv2.SOLVEPNP_ITERATIVE, 
                                                  iterationsCount=20000
                                                  )
R, _ = cv2.Rodrigues(R) 
    
print("R = ", R  , "  t= ", t)

gt_R =  np.array([[ 0.76319761, -0.60973661,  0.21389409],
 [ 0.53632637,  0.78237661,  0.31660837],
 [-0.36039345, -0.12691771,  0.92412578]]) 
gt_t = np.array([ 1.15673305, -0.88748399, 4.06010318])

rotError, transError = calculate_pose_errors(gt_R, gt_t.T, R.T, t)
  # Print the errors
print(f"Rotation Error: {rotError} deg")
print(f"Translation Error: {transError} cm")


"""
import numpy as np
import torch.nn as nn
import torch


tsor = torch.tensor([[1.0,2.0,3.0], [4.,5.,6.]])
print(torch.mean(tsor, dim=0))

