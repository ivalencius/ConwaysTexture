'''
Filename: c:\ConwaysTexture\texture.py
Path: c:\ConwaysTexture
Created Date: Monday, April 25th 2022, 12:22:17 pm
Author: Ilan Valencius

Copyright (c) 2022 Boston College
'''

import numpy as np
import matplotlib.pyplot as plt
import functools
import math

from gol import GameOfLife, downsample

# Inspiration https://engineering.purdue.edu/kak/Tutorials/TextureAndColor.pdf (section 3.3)
# https://www.mathworks.com/help/images/texture-analysis-using-the-gray-level-co-occurrence-matrix-glcm.html

# Reimplement using 'skimage.feature.graycomatrix' and 'skimage.feature.graycoprops' 
# from https://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.graycomatrix
def GLCM(image):
    displacement = [1,1]
    
    # CALCULATE THE GLCM MATRIX:
    print("The image: ") #(C2)
    glcm = np.zeros_like(image)
    rowmax, colmax = image.shape
    rowmax += -1
    colmax += -1
    '''glcm = [[0 for _ in range(GRAY_LEVELS)] for _ in range(GRAY_LEVELS)] #(C4)
    rowmax = IMAGE_SIZE - displacement[0] if displacement[0] else IMAGE_SIZE -1 #(C5)
    colmax = IMAGE_SIZE - displacement[1] if displacement[1] else IMAGE_SIZE -1 #(C6)'''
    for i in range(rowmax): #(C7)
        for j in range(colmax): #(C8)
            m, n = int(image[i][j]), int(image[i + displacement[0]][j + displacement[1]]) #(C9)
            glcm[m][n] += 1 #(C10)
            glcm[n][m] += 1 #(C11)
    print("\nGLCM: ") #(C12)
    
    # CALCULATE ATTRIBUTES OF THE GLCM MATRIX:
    entropy = energy = contrast = homogeneity = None #(D1)
    normalizer = functools.reduce(lambda x,y: x + sum(y), glcm, 0) #(D2)
    for m in range(len(glcm)): #(D3)
        for n in range(len(glcm[0])): #(D4)
            prob = (1.0 * glcm[m][n]) / normalizer #(D5)
            if (prob >= 0.0001) and (prob <= 0.999): #(D6)
                log_prob = math.log(prob,2) #(D7)
            if prob < 0.0001: #(D8)
                log_prob = 0 #(D9)
            if prob > 0.999: #(D10)
                log_prob = 0 #(D11)
            if entropy is None: #(D12)
                entropy = -1.0 * prob * log_prob #(D13)
                continue #(D14)
            entropy += -1.0 * prob * log_prob #(D15)
            if energy is None: #(D16)
                energy = prob ** 2 #(D17)
                continue #(D18)
            energy += prob ** 2 #(D19)
            if contrast is None: #(D20)
                contrast = ((m - n)**2 ) * prob #(D21)
                continue #(D22)
            contrast += ((m - n)**2 ) * prob #(D23)
            if homogeneity is None: #(D24)
                homogeneity = prob / ( ( 1 + abs(m - n) ) * 1.0 ) #(D25)
                continue #(D26)
            homogeneity += prob / ( ( 1 + abs(m - n) ) * 1.0 ) #(D27)
    if abs(entropy) < 0.0000001: entropy = 0.0 #(D28)
    print("\nTexture attributes: ") #(D29)
    print(" entropy: %f" % entropy) #(D30)
    print(" contrast: %f" % contrast) #(D31)
    print(" homogeneity: %f" % homogeneity) #(D32) 
    return glcm

if __name__ == "__main__":
    N = 100
    b = GameOfLife(N)
    b.set_board_rand(0.5)
    # for i in range(10):
    #     b.step()
    fig, ax = plt.subplots(1,2,figsize=(20,10))
    fig.suptitle('After 10 runs', fontsize=16)
    ax[0].imshow(downsample(b.get_board()), interpolation='nearest')
    ax[0].set_title('Raw Board '+str(b.get_board().shape))
    glcm = GLCM(downsample(b.get_board()))
    ax[1].imshow(glcm, interpolation='nearest')
    ax[1].set_title('GLCM of Board '+str(glcm.shape))
    plt.show()