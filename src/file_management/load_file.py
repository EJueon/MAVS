import numpy as np
import pandas as pd
def read_excel(fname):
    
    df=pd.read_excel(fname, 
                    header=1,
                    usecols="C:F",
                    convert_float=True)
    df = df.dropna()
    print(df)
    arr=df.to_numpy()
    arr=arr.tolist()
    return arr


