import numpy as np
import os
from PIL import Image
from matplotlib.image import imread, imsave
from WaveFunctionCollapse3D import WaveFunctionCollapse3D
from pyvistaVisualizer import pyvistaVisualizer
from constantTable import NEUMANN_NEIGHBOR_3D

image = imread("./Debug/Resources/Mazelike.png")
image = np.asarray(image)
dimX, dimZ, colorDim = image.shape
dataArray = np.zeros((dimX, 1, dimZ))
colorDict = dict()
colorDictInv = dict()
for i in range(dimX):
    for k in range(dimZ):
        if tuple(image[i][k]) not in colorDict:
            colorDict[tuple(image[i][k])] = len(colorDict) + 1
            colorDictInv[colorDict[tuple(image[i][k])]] = tuple(image[i][k])
        dataArray[i][0][k] = colorDict[tuple(image[i][k])]

arrayX, arrayZ = dimX*4, dimZ*4

wfcModel = WaveFunctionCollapse3D(dataArray, (arrayX, 1, arrayZ), (3, 1, 3))

visualizer = pyvistaVisualizer(colorMode=1)
patternIndex = wfcModel.patternIndex.index
"""
for idx in patternIndex:
    os.makedirs(f"./Debug/Patterns/{idx}", exist_ok=True)
    
    pattern = wfcModel.patternList[idx].state
    shape = pattern.shape
    patternData = np.zeros((shape[0], shape[2], colorDim), dtype=float)
    for i in range(shape[0]):
        for k in range(shape[2]):
            if pattern[i][0][k] in colorDictInv:
                patternData[i][k] = np.array(colorDictInv[pattern[i][0][k]])
            else: patternData[i][k] = np.array([0.0 for _ in range(colorDim)])
    imsave(f"./Debug/Patterns/{idx}/{idx}-{wfcModel.patternList[idx].weight}.png", patternData)
    
    for n, coord in enumerate(NEUMANN_NEIGHBOR_3D):
        os.makedirs(f"./Debug/Patterns/{idx}/{n}", exist_ok=True)
        for u in patternIndex[idx][coord]:
            pattern = wfcModel.patternList[u].state
            shape = pattern.shape
            patternData = np.zeros((shape[0], shape[2], colorDim), dtype=float)
            for i in range(shape[0]):
                for k in range(shape[2]):
                    if pattern[i][0][k] in colorDictInv:
                        patternData[i][k] = np.array(colorDictInv[pattern[i][0][k]])
                    else: patternData[i][k] = np.array([0.0 for _ in range(colorDim)])
            imsave(f"./Debug/Patterns/{idx}/{n}/{u}-{wfcModel.patternList[u].weight}.png", patternData)
"""

result = wfcModel.run(debug=False)
resultImg = np.zeros((arrayX, arrayZ, colorDim), dtype=float)
for i in range(arrayX):
    for k in range(arrayZ):
        if result[i][0][k] in colorDictInv:
            resultImg[i][k] = np.array(colorDictInv[result[i][0][k]])
        else: resultImg[i][k] = np.array([0.0 for _ in range(colorDim)])

imsave("./Debug/Visualization/output.png", resultImg)
#image = Image.fromarray(resultImg, 'RGB')
#image.save("./Debug/Visualization/output.png")

#visualizer = pyvistaVisualizer(colorMode=1)
#visualizer.renderPlot(result, [], colorMode=1, colorSeed=0)