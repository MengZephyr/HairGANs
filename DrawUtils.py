import matplotlib.pyplot as plt
import numpy as np

def drawVoxSlice(fId, V, slicID):
    fig = plt.figure(fId)
    slice = V[:, :, slicID, :]
    y = fig.add_subplot(1, 1, 1)
    y.imshow(slice)
    return fig

def ImgMergeVoxSlice(VV, bSize, sliceID):
    Img = VV[0]
    Img = Img[:, :, sliceID, :]
    Img = Img[:, :, ::-1]
    for i in range(1, bSize):
        t = VV[i]
        t = t[:, :, sliceID, :]
        t = t[:, :, ::-1]
        Img = np.concatenate((Img, t), axis=1)
    return Img*255.

def plotDLoss(fId, TrainArray, begI, testArray, testCount):
    yy = np.array(TrainArray)
    numIter = yy.shape[0]
    xx = np.arange(begI, begI+numIter)
    fig = plt.figure(fId)
    plt.plot(xx, yy, color='C0', linewidth=0.8)
    txx = np.array(testCount)
    tyy = np.array(testArray)
    plt.plot(txx, tyy, color='C1', linewidth=0.8)
    return fig




