import os, glob, re
import scipy.io
import numpy as np
import cv2


def get_fileSet_list(dir):
    return glob.glob(os.path.join(dir, "*"))


def get_ConfInfo(list):
    dir = [l for l in list if re.search("Conf.png", os.path.basename(l))]
    return dir[0]


def get_DepthInfo(list):
    dir = [l for l in list if re.search("Depth.png", os.path.basename(l))]
    return dir[0]


def get_imgOriInfo(list):
    dir = [l for l in list if re.search("Ori.png", os.path.basename(l))]
    return dir[0]


def get_GTInfo(list):
    dir = [l for l in list if re.search("Ori_gt.mat", os.path.basename(l))]
    return dir[0]


def loadGTMat(file):
    Vox = scipy.io.loadmat(file)['Ori']
    Vox = np.array(Vox)
    rV = Vox[:, :, 0:96]
    gV = Vox[:, :, 96:192]
    bV = Vox[:, :, 192:288]
    rV = np.expand_dims(rV, axis=3)
    gV = np.expand_dims(gV, axis=3)
    bV = np.expand_dims(bV, axis=3)
    Vox = np.concatenate((rV, gV, bV), axis=3)
    return Vox


def voxelMatTrans(M):
    G = M[0]
    print(G.shape)
    GX = G[:, :, :, 0]
    GX = np.resize(GX, [128, 128, 96])
    GY = G[:, :, :, 1]
    GY = np.resize(GY, [128, 128, 96])
    GZ = G[:, :, :, 2]
    GZ = np.resize(GZ, [128, 128, 96])
    rstG = np.concatenate((GX, GY, GZ), axis=2)
    print(rstG.shape)
    return rstG


def read_img(file):
    img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    img = img.astype(np.float32)
    return img

def read_oriMap(file):
    img = cv2.imread(file, cv2.IMREAD_COLOR)
    rimg = cv2.split(img)[2]
    rimg = rimg.astype(np.float32)
    rimg = np.expand_dims(rimg, axis=2)
    gimg = cv2.split(img)[1]
    gimg = gimg.astype(np.float32)
    gimg = np.expand_dims(gimg, axis=2)
    Info = np.concatenate((rimg, gimg), axis=2)
    return Info


def loadGTOutput(list):
    gtDir = get_GTInfo(list)
    vox = loadGTMat(gtDir)
    return vox


def loadImgInput(list):
    confDir = get_ConfInfo(list)
    confImg = read_img(confDir)
    confImg = np.expand_dims(confImg, axis=2)

    depthDir = get_DepthInfo(list)
    depthImg = read_img(depthDir)
    depthImg = np.expand_dims(depthImg, axis=2)
    Info = np.concatenate((confImg, depthImg), axis=2)

    oriDir = get_imgOriInfo(list)
    oriInfo = read_oriMap(oriDir)
    Info = np.concatenate((oriInfo, Info), axis=2)

    return Info*(1./255.)


def getTrainBatch(list, offset, batchSize):
    b_list = list[offset:offset+batchSize]
    ImgInfo = []
    GTV = []
    for f in b_list:
        flist = get_fileSet_list(f)
        ii = loadImgInput(flist)
        oo = loadGTOutput(flist)
        ImgInfo.append(ii)
        GTV.append(oo)
    ImgInfo = np.array(ImgInfo)
    ImgInfo = np.resize(ImgInfo, [batchSize, 1024, 1024, 4])
    GTV = np.array(GTV)
    GTV = np.resize(GTV, [batchSize, 128, 128, 96, 3])
    return ImgInfo, GTV





