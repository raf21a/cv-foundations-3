import cv2
import numpy as np
import sys
import argparse
import time
import pdb

DEFAULT_WINDOW = 15
FOCAL_DISTANCE = 3740 #px
BASELINE = 160 #mm
THRESHOLD = 50


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

def saveImgMaps(img,typeMap):
    pass


def normalize(img):
    #function
    return img

def findPixelMatch(leftImage, rightImage, windowSize):
    #para o pixel x y imgL achar o correspondente na imgR 
    #    Se não hover correspondencia retorna none
    #    Se houver retorna a coordenada da correspondencia
    #function
    stereo = cv2.StereoBM_create(numDisparities=160, blockSize=windowSize)
    lGray = cv2.cvtColor(leftImage, cv2.COLOR_BGR2GRAY)
    rGray = cv2.cvtColor(rightImage, cv2.COLOR_BGR2GRAY)
    return stereo.compute(lGray,rGray)


def getWorldCoords(leftImage, corMatrix):
    x_coords = np.arange(0, leftImage.shape[1]).reshape(leftImage.shape[1], 1).T
    y_coords = np.arange(0, leftImage.shape[0]).reshape(leftImage.shape[0], 1)
    x = BASELINE * (2 * x_coords - corMatrix) / (2 * corMatrix)
    y = BASELINE * (2 * y_coords) / (2 * corMatrix)
    z = BASELINE * FOCAL_DISTANCE  / (2 * corMatrix)
    return np.stack([x, y, z], axis=-1) 

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
    worldCoords = getWorldCoords(leftImage, corMatrix)
    z = worldCoords[:, :, -1]
    z[z == np.inf] = -np.inf
    z[z == -np.inf] = np.max(z)
    worldCoords[worldCoords == np.inf] = np.max(worldCoords)
    dispMatrix = cv2.normalize(np.abs(corMatrix), dst=np.zeros(corMatrix.shape),
                               alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,
                               dtype=cv2.CV_8U)
    depthMatrix = cv2.normalize(z, dst=np.zeros(z.shape),
                               alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,
                               dtype=cv2.CV_8U)
    print("Salvando disp image em data...")
    cv2.imwrite(imageL.rstrip("L.png") + "_disp.png", dispMatrix)
    print("Salvando depth image em data...")
    cv2.imwrite(imageL.rstrip("L.png") + "_depth.png", depthMatrix)


def main(r1, r2, r3, imageL=None,
         imageR=None, windowSize=DEFAULT_WINDOW):
    if r1:
        return createDepthMap(imageL, imageR, windowSize)
    elif r2:
        print("r2")
    elif r3:
        print("r3")


if __name__ == "__main__":
    args = parser.parse_args()
    main(args.r1, args.r2, args.r3, args.img_l, args.img_r,
         args.windowSize)
