import pandas as pd

data = {
    'Color':['Red', 'Blue', 'Green', 'Blue', 'Red'],
    'Size': ['S', 'M', 'L', 'M', 'S'],
    'Price': [10, 15, 20, 15, 10]
}
df = pd.DataFrame(data)
print(df)

df_encoded = pd.get_dummies(df, columns=['Color', 'Size'])
df_encoded = df_encoded.astype(int) # boolean 타입을 int 타입으로 변환
print(df_encoded)