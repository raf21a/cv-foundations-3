import cv2
import numpy as np
import sys
import argparse
import time

DEFAULT_WINDOW = 5
FOCAL_DISTANCE = 3740 #px
BASELINE = 160 #mm

parser = argparse.ArgumentParser(
    description="Visão Stereo")
parser.add_argument("requisito", type=int, nargs=1, choices=range(1, 5),
                    help="número do requisito de avaliação")
parser.add_argument("-imageL", nargs="?", default=None, metavar="path/to/file",
                    help="Caminho para a imagem da esquerda")
parser.add_argument("-imageR", nargs="?", default=None, metavar="path/to/file",
                    help="Caminho para a imagem da direita")
parser.add_argument("-sizeWindow", nargs="?", default=DEFAULT_SQUARE, type=float,
                    help="Tamanho da janela utilizazada para o calculo stereoMacthing")

def openImage(file):
    if file is None:
        filename = "data/lena.png"
    else:
        filename = file
    img = cv2.imread(filename)
    if img is None:
        sys.exit("Não foi possível abrir imagem")
    return img

def saveImgMaps(img,typeMap):

def normalize(img):
    #function
    return img

def findPixelMatched(imageSplitL, imageSplitR, sizeWindow, x, y):
    #para o pixel x y imgL achar o correspondente na imgR 
    #    Se não hover correspondencia retorna none
    #    Se houver retorna a coordenada da correspondencia
    #function
    return x,y

def calcWorldCoords(xl, yl, xr, yr):
    X = (BASELINE/2)*((xl + xr)/(xl - xr))
    Y = (BASELINE/2)*((yl + yr)/(xl - xr))
    Z = (BASELINE*FOCAL_DISTANCE)/(xl - xr)
    return X, Y, Z

def createDethDistMaps(imageL, imageR, sizeWindow):
    #para todo pixel em imageL
    #    findPixelMatched
    #    calcWorldCoords com as coordenadas de correspondencia calcular as coordenadas no mundo
    #    calcula e adiciona a disparidade entre os pixels no numpy vetor disp_map
    #    adicionar ao vetor numpy world_coords
    #calcula o depth_map a partir de do world_coords
    #normaliza o depth_map
    #normaliza o dist_map
    #salva os mapas
    saveImgMaps("depth", imageL, depth_map)
    saveImgMaps("disp", imageL, disp_map)
    return 

def main(requisite, imageL=None,
         imageR=None, sizeWindow=DEFAULT_WINDOW):
    if requisite == 1:
        return createDethDistMaps(imageL, imageR, sizeWindow)
    elif requisite == 2:
        return None
    elif requisite == 3:
        return None
    elif requisite == 4:
        return None


if __name__ == "__main__":
    args = parser.parse_args()
    main(args.requisito[0], args.imageL, args.imageR,
         args.sizeWindow)
