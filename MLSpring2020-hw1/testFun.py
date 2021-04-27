import pandas as pd
import numpy as np
import xgboost_pred as xgb
import matplotlib.pyplot as plt

a = np.float32([[1, 2, 3], [4, 5, 6]])
b = np.float32([[7], [8], [9]])
print(a)
print(b)
print(np.matmul(a, b))
