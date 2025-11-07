import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

matplotlib.rcParams['font.family'] = ['Malgun Gothic']
matplotlib.rcParams['axes.unicode_minus'] = False

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

df = pd.read_csv('고객복지데이터셋Cleaned.csv')

# print("\n꺾은 선 그래프(Line Graph)")
# plt.savefig('c02.correlation_heatmap_lower_triangle.png')
# print("\n산점도 그래프(Scatter Plot)")
#

# print("\n히트맵(Heat Map)")
#
# print("\n막대 그래프(countplot)")
#
# print("\n분포 시각화(displot)")
# 결측치 제거
# score = df['만족도점수']
#
# # displot으로 분포 확인
# sns.displot(score,
#             kde=True,  # 커널 밀도 추정 곡선 표시
#             bins=20,  # 히스토그램 막대 개수
#             color='skyblue',  # 색상 지정
#             height=6,  # 그래프 높이
#             aspect=1.5  # 가로 세로 비율
#             )
#
# plt.title("만족도점수 분포")
# plt.xlabel("만족도점수")
# plt.ylabel("빈도수")
# plt.savefig('e01.xxx.png')
#
# print("\n쌍그래프 함수(pairplot)")
#
# print("\n바이올린 그래프(violinplot)")
#
# print("\n선형 회귀 그래프(lmplot)")
#
# print("\n관계 시각화 그래프(relplot)")
#
# print("\n이변량 관계 그래프(jointplot)")
#
# print("\n막대 그래프(barplot)")
#
# print("\n커널 밀도 추정 그래프(kdeplot)")
#
# print("\n상자 수염 그래프(boxplot)")
#
# print("\n산점도 그래프(scatterplot)")
#
# print("\n상관 분석")
#
# print("\n상관 분석")
#
# print("\n상관 분석")
# plt.savefig('c02.correlation_heatmap_lower_triangle.png')
