import numpy as np
import pandas as pd
import spatialdata as sd
import spatialdata_io as sd_io
from spatialdata_io.readers.xenium import xenium_aligned_image
import spatialdata_plot
from spatialdata import bounding_box_query

#choose a region from the whole dataset
def boxquery(sdata, startX, startY, endX, endY):
    return bounding_box_query(sdata, min_coordinate=[startX, startY], max_coordinate=[endX, endY],axes=("x", "y"), target_coordinate_system="global")


#normalize the pathology image to [0, 1]
def normADpath(sptData, keyLab = 'Abeta_if', backgroundQuantile = 0.95, signalTopQuantile = 0.999):
    adPath = sptData[keyLab][0,:,:].data
    base = np.quantile(adPath,backgroundQuantile)
    topV = np.quantile(adPath,signalTopQuantile)
    adPath[adPath<base] = base
    adPath[adPath>topV] = topV
    valM = adPath.max().compute()/2
    adPath = (adPath- base)/(topV - base)
    return adPath


#prepare the multi-dimention spatial gene expression array, select the gene sets for the trainning
def prepareArrayX(sData, scaleX = 1.544418,scaleY = 1.544418 ):
    genes = sData['transcripts']['feature_name'].unique()
    num_genes = len(genes)
    gene_to_index = {gene: i for i, gene in enumerate(genes)}

    #Define the dimensions of the gene expression array size
    width = int(sData['transcripts']['x'].max().compute()*scaleX) - int(sData['transcripts']['x'].min().compute()*scaleX) +1
    height = int(sData['transcripts']['y'].max().compute()*scaleY) - int(sData['transcripts']['y'].min().compute()*scaleY) +1

    minvx = int(sData['transcripts']['x'].min().compute()*scaleX)
    minvy = int(sData['transcripts']['y'].min().compute()*scaleY)
    
    #create an empty array to hold the data
    array = np.zeros((num_genes, height, width), dtype = np.float32)

    #Fill the array with expression value
    for index, row in sData['transcripts'].iterrows():
        x, y, gene = int(row['x']*scaleX)-minvx, int(row['y']*scaleY)-minvy, row['feature_name']
        gene_index = gene_to_index[gene]
        array[gene_index, x, height - y -1] +=1
    return array, num_genes



def prepareDataset(dataPath, image_file, alignMatrix, patchsize):

    # load the raw single molecule spatial dataset from xenium
    sdata = sd_io.xenium(dataPath)
    image = xenium_aligned_image(image_file, alignMatrix)
    sdata['if'] = image
    
    (xArray, numGene) = prepareArrayX(sdata)
    yImage = normADpath(sdata)

    #Define the size of test and train dataset
    xTrain = np.zeros((sampleSize, num_genes, patch_size, patch_size), dtype= np.float32)
    yTrain = np.zeros((sampleSize, patch_size, patch_size), dtype= np.float32)

    xTest = np.zeros((sampleSize, num_genes, patch_size, patch_size), dtype= np.float32)
    yTest = np.zeros((sampleSize, patch_size, patch_size), dtype= np.float32)

    #loop to prepare trian and test dataset
    trainCount = 0
    testCount = 0
    totalCount = 0
    for i in range(int(yImage.shape[0]/patch_size)):
        for j in range(int(yImage.shape[0]/patch_size)):
            selImage = yImage[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size]

            #choose the one with positive AD signal
            if(totalCount%2==0 and trainCount<sampleSize):
                xTrain[trainCount] = xArray[:,i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size]
                yTrain[trainCount] = selImage
                trainCount+=1
                totalCount+=1
                continue

            #switch to test dataset
            if(totalCount%2==1 and testCount<sampleSize):
                xTest[testCount] = xArray[:,i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size]
                yTest[testCount] = selImage
                testCount+=1
                totalCount+=1
                continue

            if totalCount == sampleSize*2:
                break

    #return the final dataset
    return xTrain, yTrain, xTest, yTest
