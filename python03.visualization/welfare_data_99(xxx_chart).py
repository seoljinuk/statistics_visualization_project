import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

matplotlib.rcParams['font.family'] = ['Malgun Gothic']
matplotlib.rcParams['axes.unicode_minus'] = False

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

dataIn = './../dataIn/'
dataOut = './../dataOut/'

df = pd.read_csv(dataIn + '고객복지데이터셋Cleaned.csv')


'''
subset_df_04 = df[['복지등급', '보육지원']]
cross_df_04 = pd.crosstab(index=subset_df_04['보육지원'],
                       columns=subset_df_04['복지등급'])
print(cross_df_04)

각 object 컬럼의 unique 개수 출력
고객ID: 1000개
월소득: 934개
성별: 3개
결혼상태: 4개
고용형태: 5개
학력: 5개
복지등급: 5개
지역: 8개
장애여부: 3개
웰니스참여: 2개
보육지원: 2개
정신건강지원: 2개
기록일자: 1개

각 숫자형 컬럼의 unique 개수 출력
나이: 817개
근속연수: 817개
복지비사용액: 817개
만족도점수: 817개
건강지수: 817개
지원인원: 817개
'''

# print("\n꺾은 선 그래프(Line Graph)")
# plt.savefig('c02.correlation_heatmap_lower_triangle.png')

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
