import pandas as pd
import numpy as np

data = {
    'a': [1, 2, np.nan, np.nan, 5], # 결측치 2개
    'b': [np.nan, 2, np.nan, 4, np.nan],  # 결측치 3개
}

df = pd.DataFrame(data)
print(df)

print(df.info())

missing_value_count = df.isna().sum()
print(missing_value_count)

df_filled = df.apply(lambda col : col.fillna(col.mean()))
print(df_filled)

print(df_filled.info())
