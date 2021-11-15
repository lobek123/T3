import pandas as pd
def null_check(file):
    df = pd.read_csv(file)
    print(df.isnull().values.any())
    print(df.isnull().sum())
