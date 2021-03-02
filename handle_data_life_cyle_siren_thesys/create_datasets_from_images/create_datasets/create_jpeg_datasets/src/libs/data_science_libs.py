import numpy as np
import pandas as pd
import scipy
import sklearn

# import dataframe_image as dfi

# from pandas.table.plotting import table

# skimage - imports
# ----------------------------------------------- #
import skimage
import skimage.metrics as skmetrics
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error

# sklearn - imports
# ----------------------------------------------- #
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

from sklearn.model_selection import ParameterGrid
