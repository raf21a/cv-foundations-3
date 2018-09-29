#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import sys
import argparse
import time
import pdb

DEFAULT_WINDOW = 25
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
parser.add_argument("--windowSize", nargs="?", default=DEFAULT_WINDOW, type=check_positive_odd,
                    help="Tamanho da janela utilizazada para o calculo stereoMatching")


def openImage(file):
    if file is None:
        filename = "data/aloeL.png"
    else:
        filename = file
    img = cv2.imread(filename)
    if img is None:
        sys.exit("Não foi possível abrir imagem")
    return img


def findPixelMatch(leftImage, rightImage, windowSize):
    #para o pixel x y imgL achar o correspondente na imgR 
    #    Se não hover correspondencia retorna none
    #    Se houver retorna a coordenada da correspondencia
    #function
    stereo = cv2.StereoBM_create(numDisparities=128, blockSize=windowSize)
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

def createDepthMap(imageL, imageR, windowSize):
    if windowSize == DEFAULT_WINDOW:
        print("Usando janela de comparação de tamanho {}. Use a opção --windowSize para outro tamanho".format(DEFAULT_WINDOW))
    if imageL is None or imageR is None:
        imageL = "../data/aloeL.png"
        print("Criando mapa de profundidade para aloe. Caso deseje a do bebe, use a opção -imageL and -imageR com as imagens correspondentes.")
        leftImage = openImage("../data/aloeL.png")
        rightImage = openImage("../data/aloeR.png")
    else:
        leftImage = openImage(imageL)
        rightImage = openImage(imageR)
    corMatrix = findPixelMatch(leftImage, rightImage, windowSize)
    x, y, z = getWorldCoords(leftImage, corMatrix)
    z[z == np.inf] = -np.inf
    z[z == -np.inf] = np.max(z)
    dispMatrix = normalize(np.abs(corMatrix))
    depthMatrix = normalize(z)
    saveImage(imageL.rstrip("L.png") + "_disp.png", dispMatrix)
    saveImage(imageL.rstrip("L.png") + "_depth.png", depthMatrix)


def getRotation(r1, r2):
    return np.matmul(r1.T, r2)


def getTranslation(r1, t1, t2):
    return np.dot(r1.T, t2) - np.dot(r1.T, t1)


def depthFromHomography():
    leftImage = openImage("../data/MorpheusL.jpg")
    rightImage = openImage("../data/MorpheusR.jpg")
    rotation = getRotation(MORPHEUS_LEFT_ROTATION, MORPHEUS_RIGHT_ROTATION)
    translation = getTranslation(MORPHEUS_LEFT_ROTATION, MORPHEUS_LEFT_TRANSLATION.T, MORPHEUS_RIGHT_TRANSLATION.T)
    r1, r2, p1, p2, q = cv2.stereoRectify(MORPHEUS_LEFT_CAMERA, MORPHEUS_RIGHT_CAMERA, None, None, leftImage.shape[0:2], rotation, translation)

def main(r1, r2, r3, imageL=None,
         imageR=None, windowSize=DEFAULT_WINDOW, bonus=None):
    if r1:
        return createDepthMap(imageL, imageR, windowSize)
    elif r2:
        if bonus:
            pass
        else:
            return depthFromHomography()
    elif r3:
        print("r3")


if __name__ == "__main__":
    args = parser.parse_args()
    main(args.r1, args.r2, args.r3, args.img_l, args.img_r,
         args.windowSize, args.bonus)
