import glob

import cv2
import matplotlib.pyplot as plt
import numpy as np

# Définir les dimensions de échiquier
CHECKERBOARD = (8, 6)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Liste pour stocker les points d'objet 3D pour chaque image de damier dans l'espace du monde réel
objpoints = []
# liste pour stocker les points d'image 2D pour chaque image de damier dans le plan image
imgpoints = []

# Définir les coordonnées du monde reel pour les points 3D
objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

# Données Images
data = glob.glob('data/GO*.jpg')

# Parcourir la liste et cherchez les coins de l'échiquier
for idx, fname in enumerate(data):
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detecter automatiquement les coins -> findChessbordCorners
    # Si le nombre de coins souhaité est trouvé dans l'image, ret = vrai
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD,
                                             cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK +
                                             cv2.CALIB_CB_NORMALIZE_IMAGE)

    if ret == True:  # Si le nombre de coins souhaité est détecté

        objpoints.append(objp)
        imgpoints.append(corners)

        # Dessiner et afficher les coins
        # img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
        img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners, ret)

    cv2.imshow('img', img)
    cv2.waitKey(0)
cv2.destroyAllWindows()

h, w = img.shape[:2]

# Calibrer la caméra
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

print("Camera matrix : {}\n".format(mtx))
symbole = "<>"
print(50 * symbole)
print("Coefficient de distorsion : {}\n".format(dist))
# print(dist)
print(50 * symbole)
print("Valeurs de rotation : \n {}".format(rvecs))
# print(rvecs)
print(50 * symbole)
print("Valeurs de translation: {}\n".format(tvecs))

# Image de test
img_test = cv2.imread('test_image.jpg')
# Taille image
img_size = (img_test.shape[1], img_test.shape[0])


def cal_undistort(img_test, objpoints, imgpoints):
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)

    # Corriger la distorsion
    undist = cv2.undistort(img_test, mtx, dist, None, mtx)
    return undist


undistorted = cal_undistort(img_test, objpoints, imgpoints)

# Resultats
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()
ax1.imshow(img_test)
ax1.set_title('Image Original', fontsize=15)
ax2.imshow(undistorted)
ax2.set_title('Image non déformée', fontsize=15)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
plt.show()
