
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

plt.rc('font', family='Malgun Gothic')

from scipy.spatial.distance import euclidean, cosine, pdist, squareform

data = [
    [2, 5, 3, 4, 4],
    [2, 4, 3, 3, 5],
    [4, 5, 3, 4, 4],
    [5, 6, 3, 3, 3],
    [4, 1, 3, 2, 2]
]

mat = np.array(data)

# 5개의 음식 정보에 대한 5개의 특성(feature)
myindex = ['짜장면', '짬뽕', '라면', '우동', '돈가스']
mycolumns = ['달달함', '목넘김', '고소함', '기름짐', '매콤함']
df = pd.DataFrame(mat, index=myindex, columns=mycolumns)
print(df)

def calculate_euclidean(df):
    distances = pd.DataFrame(index=df.index, columns=df.index)
    for row in df.index:
        for col in df.index:
            distances.loc[row, col] = euclidean(df.loc[row], df.loc[col])
    return distances

euclidean_distances = calculate_euclidean(df)
print(f'유클리디언 거리\n', euclidean_distances)

def calculate_cosine_similarity(df):
    similarities = pd.DataFrame(index=df.index, columns=df.index)
    for row in df.index:
        for col in df.index:
            similarities.loc[row, col] = 1 - cosine(df.loc[row], df.loc[col])
    return similarities

cosine_similarities = calculate_cosine_similarity(df)
print(f'코사인 유사도\n', cosine_similarities)

# pdist 함수는 1차원 벡터로 반환
distances = pdist(df, metric='euclidean')

# 거리 벡터 : 각 항목들의 거리 정보
distance_matrix = squareform(distances)

print('\n거리 벡터:')
print(distances)

print('\n거리 행렬:')
print(distance_matrix)

# 히트맵 그리기
plt.figure(figsize=(6, 5))
sns.heatmap(distance_matrix, annot=True, cmap='coolwarm', xticklabels=myindex, yticklabels=myindex)
plt.title('음식간 거리 히트맵')

dataOut = './../dataOut/'
plt.savefig(dataOut + 'distance_matrix.png')








