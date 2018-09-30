#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import sys
import argparse
import time
import pdb

FOCAL_DISTANCE = 3740 #px
BASELINE = 160 #mm
THRESHOLD = 50
MORPHEUS_LEFT_CAMERA = np.array([[6704.926882, 0.000103,    738.251932],
                                 [0.,          6705.241311, 457.560286],
                                 [0.,          0.,          1.]])
MORPHEUS_LEFT_ROTATION = np.array([[0.70717199,  0.70613396, -0.03581348],
                                   [0.28815232, -0.33409066, -0.89741388],
                                   [-0.64565936,  0.62430623, -0.43973369]])
MORPHEUS_LEFT_TRANSLATION = np.array([-532.285900, 207.183600, 2977.408000])
MORPHEUS_RIGHT_CAMERA = np.array([[6682.125964, 0.000101,    875.207200],
                                 [0.,          6681.475962, 357.700292],
                                 [0.,          0.,          1.]])
MORPHEUS_RIGHT_ROTATION = np.array([[0.48946344,  0.87099159, -0.04241701],
                                   [0.33782142, -0.23423702, -0.91159734],
                                   [-0.80392924,  0.43186419, -0.40889007]])
MORPHEUS_RIGHT_TRANSLATION = np.array([-614.549000, 193.240700, 3242.754000])


def check_positive_odd(value):
    ivalue = int(value)
    if ivalue % 2 == 0 or ivalue <= 0:
         raise argparse.ArgumentTypeError("%s is an even value or not positive" % value)
    return ivalue


parser = argparse.ArgumentParser(
    description="Visão Stereo")
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument("--r1", action="store_true",
                    help="Requisito 1")
group.add_argument("--r2", action="store_true",
                    help="Requisito 2")
group.add_argument("--r3", action="store_true",
                    help="Requisito 3")
parser.add_argument("--bonus", action="store_true",
                    help="Mapa de profundidade pela matriz fundamental")
parser.add_argument("--img_l", nargs="?", default=None, metavar="path/to/file",
                    help="Caminho para a imagem da esquerda")
parser.add_argument("--img_r", nargs="?", default=None, metavar="path/to/file",
                    help="Caminho para a imagem da direita")


def openImage(file):
    if file is None:
        filename = "data/aloeL.png"
    else:
        filename = file
    img = cv2.imread(filename)
    if img is None:
        sys.exit("Não foi possível abrir imagem")
    return img


def findPixelMatch(leftImage, rightImage, windowSize, numDisparities=128):
    #para o pixel x y imgL achar o correspondente na imgR 
    #    Se não hover correspondencia retorna none
    #    Se houver retorna a coordenada da correspondencia
    #function
    stereo = cv2.StereoBM_create(numDisparities, blockSize=windowSize)
    lGray = cv2.cvtColor(leftImage, cv2.COLOR_BGR2GRAY)
    rGray = cv2.cvtColor(rightImage, cv2.COLOR_BGR2GRAY)
    return stereo.compute(lGray,rGray)


def getWorldCoords(leftImage, corMatrix):
    x_coords = np.arange(0, leftImage.shape[1]).reshape(leftImage.shape[1], 1).T
    y_coords = np.arange(0, leftImage.shape[0]).reshape(leftImage.shape[0], 1)
    x = BASELINE * (2 * x_coords - corMatrix) / (2 * corMatrix)
    y = BASELINE * (2 * y_coords) / (2 * corMatrix)
    z = BASELINE * FOCAL_DISTANCE  /   corMatrix
    return x, y, z


def normalize(matrix):
    return cv2.normalize(matrix, dst=np.zeros(matrix.shape),
                         alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,
                         dtype=cv2.CV_8U)

def saveImage(name, image):
    print("Salvando {}...".format(name))
    cv2.imwrite(name, image)

def createDepthMap(imageL, imageR):
    if imageL is None or imageR is None:
        imageL = "../data/aloeL.png"
        print("Criando mapa de profundidade para aloe. Caso deseje a do bebe, use a opção -imageL and -imageR com as imagens correspondentes.")
        leftImage = openImage("../data/aloeL.png")
        rightImage = openImage("../data/aloeR.png")
    else:
        leftImage = openImage(imageL)
        rightImage = openImage(imageR)
    dispMatrix, depthMatrix, _, _ = tuneParameters("r1", leftImage, rightImage)
    saveImage(imageL.rstrip("L.png") + "_disp.png", dispMatrix)
    saveImage(imageL.rstrip("L.png") + "_depth.png", depthMatrix)


def getRotation(r1, r2):
    return np.matmul(r2, r1.T)


def getTranslation(r, t1, t2):
    return t2 - np.matmul(r, t1)


def depthFromHomography():
    leftImage = openImage("../data/MorpheusL.jpg")
    rightImage = openImage("../data/MorpheusR.jpg")
    rotation = getRotation(MORPHEUS_LEFT_ROTATION, MORPHEUS_RIGHT_ROTATION)
    translation = getTranslation(rotation, MORPHEUS_LEFT_TRANSLATION, MORPHEUS_RIGHT_TRANSLATION)

    r1, r2, p1, p2, q, _, _ = cv2.stereoRectify(MORPHEUS_LEFT_CAMERA, None, MORPHEUS_RIGHT_CAMERA, None,
                                                leftImage.shape[0:2], rotation, translation, flags=0)

    map1x, map1y = cv2.initUndistortRectifyMap(MORPHEUS_LEFT_CAMERA, None, r1, p1, leftImage.shape[0:2], cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(MORPHEUS_RIGHT_CAMERA, None, r2, p2, leftImage.shape[0:2], cv2.CV_32FC1)
    leftReprojection = cv2.remap(leftImage, map1x, map1y, cv2.INTER_LINEAR)
    rightReprojection = cv2.remap(rightImage, map2x, map2y, cv2.INTER_LINEAR)

    cv2.namedWindow('Retificação', cv2.WINDOW_NORMAL)

    while(True):
        cv2.imshow("Retificação", np.hstack([leftReprojection, rightReprojection]))
        if cv2.waitKey(1) & 0xFF == 27:
            break
    cv2.destroyAllWindows()
    cv2.waitKey(1)

    dispMatrix, depthMatrix, _, _ = tuneParameters("r2", leftReprojection, rightReprojection, q)
    saveImage("../data/morpheus_disp.jpg", dispMatrix)
    saveImage("../data/morpheus_depth.jpg", depthMatrix)


def depthFromFundamentalMatrix():

    #find possible matches
    leftImage = openImage("../data/MorpheusL.jpg")
    rightImage = openImage("../data/MorpheusR.jpg")

    sift = cv2.xfeatures2d.SIFT_create()

    kpl, descriptorLeft = sift.detectAndCompute(leftImage,None)
    kpr, descriptorRight = sift.detectAndCompute(rightImage,None)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params,search_params)
    matches = flann.knnMatch(descriptorLeft,descriptorRight,k=2)

    good = []
    pts1 = []
    pts2 = []

    for i,(m,n) in enumerate(matches):
        if m.distance < 0.8*n.distance:
            good.append(m)
            pts2.append(kpr[m.trainIdx].pt)
            pts1.append(kpl[m.queryIdx].pt)
    #possible matches: pts1 and pts2

    pts1 = np.array(pts1)
    pts2 = np.array(pts2)
    F, mask = cv2.findFundamentalMat(pts1,pts2,cv2.FM_LMEDS)

    # We select only inlier points
    pts1 = pts1[mask.ravel()==1]
    pts2 = pts2[mask.ravel()==1]

    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)

    p1fNew = pts1.reshape((pts1.shape[0] * 2, 1))
    p2fNew = pts2.reshape((pts2.shape[0] * 2, 1))
    retBool ,rectmat1, rectmat2 = cv2.stereoRectifyUncalibrated(p1fNew,p2fNew,F,(2048,2048))
    dst1 = cv2.warpPerspective(leftImage,rectmat1,(2048,2048))
    dst2 = cv2.warpPerspective(rightImage,rectmat2,(2048,2048))

    cv2.namedWindow('Teste 1', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Teste 2', cv2.WINDOW_NORMAL)

    while(True):
        cv2.imshow("Teste 1", dst1)
        cv2.imshow("Teste 2", dst2)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    

def measureBox(bonus):
    global first, pos1, pos2
    first = True
    pos1 = pos2 = None

    if bonus:
        pass
    else:
        leftImage = openImage("../data/MorpheusL.jpg")
        rightImage = openImage("../data/MorpheusR.jpg")
        rotation = getRotation(MORPHEUS_LEFT_ROTATION, MORPHEUS_RIGHT_ROTATION)
        translation = getTranslation(rotation, MORPHEUS_LEFT_TRANSLATION.T, MORPHEUS_RIGHT_TRANSLATION.T)

        r1, r2, p1, p2, q, _, _ = cv2.stereoRectify(MORPHEUS_LEFT_CAMERA, None, MORPHEUS_RIGHT_CAMERA, None,
                                                    leftImage.shape[0:2], rotation, translation, flags=0)

        map1x, map1y = cv2.initUndistortRectifyMap(MORPHEUS_LEFT_CAMERA, None, r1, p1, leftImage.shape[0:2], cv2.CV_32FC1)
        map2x, map2y = cv2.initUndistortRectifyMap(MORPHEUS_RIGHT_CAMERA, None, r2, p2, leftImage.shape[0:2], cv2.CV_32FC1)
        leftReprojection = cv2.remap(leftImage, map1x, map1y, cv2.INTER_LINEAR)
        rightReprojection = cv2.remap(rightImage, map2x, map2y, cv2.INTER_LINEAR)

        dispMatrix, depthMatrix, corMatrix, worldCoords = tuneParameters("r3", leftReprojection, rightReprojection, q)

        cv2.namedWindow('Imagem', cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('Imagem', getMeasure, [leftReprojection, worldCoords])

        while(True):
            cv2.imshow("Imagem", leftReprojection)
            if cv2.waitKey(1) & 0xFF == 27:
                break
        cv2.destroyAllWindows()
        cv2.waitKey(1)


def getMeasure(event, x, y, flags, params):
    global first, pos1, pos2
    leftReprojection, worldCoords = params
    x_w, y_w, z_w = worldCoords
    if event == cv2.EVENT_LBUTTONUP:
        if first:
            pos1 = [x, y]
        else:
            pos2 = [x, y]
        first = not first
        if pos1 is not None and pos2 is not None:
            cv2.namedWindow('Medida', cv2.WINDOW_NORMAL)
            img = leftReprojection
            cv2.line(img, tuple(pos1),
                     tuple(pos2), (0, 0, 255), 2)
            cv2.imshow("Medida", img)
            x1, y1 = pos1
            x2, y2 = pos2
            print("Pos1:{}, Pos2:{}".format((y_w[x2][y2], y_w[x1][y1], z_w[x1][y1]),
                                            (x_w[x2][y2], y_w[x2][y2], z_w[x2][y2])))
            height = np.abs(y_w[x1][y1] - y_w[x2][y2])
            width = np.abs(y_w[x2][y2] - x_w[x2][y2])
            depth = np.abs(z_w[x1][y1] - z_w[x2][y2])
            print("Altura: {}\nLargura: {}\nProfundidade={}\n".format(height, width, depth))
            pos1 = pos2 = None


def tuneParameters(req, leftImage, rightImage, q=None):
    print("Otimizar parâmetros...")
    def nothing(x):
        pass

    cv2.namedWindow('disp', cv2.WINDOW_NORMAL)

    cv2.createTrackbar('numOfDisparities', 'disp', 1, 20, nothing)
    cv2.createTrackbar('windowSize', 'disp', 5, 255, nothing)

    while(True):
        numOfDisparities = cv2.getTrackbarPos('numOfDisparities', 'disp') * 16
        windowSize = cv2.getTrackbarPos('windowSize', 'disp')
        if numOfDisparities < 16:
            numOfDisparities = 16
        if windowSize % 2 == 0:
            windowSize += 1
        if windowSize < 5:
            windowSize = 5
        corMatrix = findPixelMatch(leftImage, rightImage, windowSize, numOfDisparities)
        if req == "r1" or req == "r2":
            x, y, z = getWorldCoords(leftImage, corMatrix)
        else:
            worldCoords = cv2.reprojectImageTo3D(corMatrix, q, handleMissingValues=True)
            x, y, z = worldCoords[:,:,0], worldCoords[:,:,1], worldCoords[:,:,2]
        z[z == np.inf] = -np.inf
        z[z == -np.inf] = np.max(z)
        dispMatrix = normalize(corMatrix)
        depthMatrix = normalize(z)
        cv2.imshow("disp", dispMatrix)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
    cv2.destroyAllWindows()
    cv2.waitKey(1)

    return dispMatrix, depthMatrix, corMatrix, (x, y, z)


def main(r1, r2, r3, imageL=None,
         imageR=None, bonus=None):
    if r1:
        return createDepthMap(imageL, imageR)
    elif r2:
        if bonus:
            depthFromFundamentalMatrix()
        else:
            return depthFromHomography()
    elif r3:
        measureBox(bonus)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args.r1, args.r2, args.r3, args.img_l, args.img_r, args.bonus)
