print(">>> Starting environment check...")

import sys
import pandas as pd
import numpy as np
import sklearn
import yfinance as yf
import statsmodels
import arch
import joblib
import matplotlib
import seaborn as sns

print("Python:", sys.version)
print("pandas:", pd.__version__)
print("numpy:", np.__version__)
print("sklearn:", sklearn.__version__)
print("yfinance:", yf.__version__)
print("statsmodels:", statsmodels.__version__)
print("arch:", arch.__version__)
print("matplotlib:", matplotlib.__version__)
print("seaborn:", sns.__version__)

print(">>> Environment check passed.")
