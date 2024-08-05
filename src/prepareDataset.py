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
  
def prepareDataset(dataPath, image_file, alignMatrix, patchsize):

  # load the raw single molecule spatial dataset from xenium
  sdata = sd_io.xenium(dataPath)
  image = xenium_aligned_image(image_file, alignMatrix)
  sdata['if'] = image

  
