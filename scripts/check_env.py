import sys

import arch
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
import statsmodels
import yfinance as yf

print(">>> Starting environment check...")
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
