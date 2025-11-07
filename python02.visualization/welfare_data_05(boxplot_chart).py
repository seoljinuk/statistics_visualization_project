import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['font.family'] = ['Malgun Gothic']
matplotlib.rcParams['axes.unicode_minus'] = False

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

dataIn = './../dataIn/'
dataOut = './../dataOut/'

df = pd.read_csv(dataIn + '고객복지데이터셋Cleaned.csv')

###############################################################################
def MakeBoxPlotChart01(dataframe, file_name):
    """
    성별별 근속연수의 상자 수염 그래프를 그리는 함수
    - 두 가지 유형(Rectangular / Notched)을 한 화면에 표시
    """
    plt.style.use('ggplot')

    # 데이터 준비
    male_data = dataframe.loc[dataframe['성별'] == '남성', '근속연수']
    female_data = dataframe.loc[dataframe['성별'] == '여성', '근속연수']

    chartdata = [np.array(male_data), np.array(female_data)]
    xtick_label = ['남성', '여성']

    # Figure 생성
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(9, 4))

    # ─────────────────────────────
    # ① Rectangular Box Plot
    bplot1 = ax1.boxplot(chartdata,
                         vert=True,
                         patch_artist=True,
                         tick_labels=xtick_label)
    ax1.set_title('Rectangular Box Plot')

    # ─────────────────────────────
    # ② Notched Box Plot
    bplot2 = ax2.boxplot(chartdata,
                         notch=True,
                         vert=True,
                         patch_artist=True,
                         tick_labels=xtick_label)
    ax2.set_title('Notched Box Plot')

    # 색상 지정
    colors = ['lightblue', 'lightpink']
    for bplot in (bplot1, bplot2):
        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)

    # 공통 설정
    for ax in [ax1, ax2]:
        ax.yaxis.grid(True)
        ax.set_xlabel('성별')
        ax.set_ylabel('근속연수(년)')
        ax.set_ylim(bottom=0)

    # 그래프 저장
    plt.tight_layout()
    plt.savefig(dataOut + file_name, dpi=400)
# end def MakeBoxPlotChart01

subset_df_08 = df[['성별', '근속연수']]

tenure = df['근속연수']
MakeBoxPlotChart01(subset_df_08, 'g01.boxplot_chart.png')

# print("\n근속연수 통계 정보")
print(tenure.describe())
###############################################################################
def MakeBoxPlotChart02(dataframe, file_name):
    """
    성별에 따른 근속연수의 분포를 Box plot과 Violin plot으로 비교하는 함수

    Parameters
    ----------
    dataframe : pandas.DataFrame
        '성별'과 '근속연수' 컬럼을 포함한 데이터프레임
    """
    # 남성과 여성 데이터 분리
    male_data = dataframe.loc[dataframe['성별'] == '남성', '근속연수']
    female_data = dataframe.loc[dataframe['성별'] == '여성', '근속연수']

    chartdata = [np.array(male_data), np.array(female_data)]
    xtick_label = ['남성', '여성']

    # Figure, Subplots
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(9, 4))

    # Violin Plot
    axs[0].violinplot(chartdata, showmeans=False, showmedians=True)
    axs[0].set_title('성별별 근속연수 분포 (Violin Plot)')

    # Box Plot
    axs[1].boxplot(chartdata, patch_artist=True)
    axs[1].set_title('성별별 근속연수 분포 (Box Plot)')

    # 색상 지정 (남성-파랑, 여성-핑크)
    colors = ['lightblue', 'pink']
    for patch, color in zip(axs[1].artists, colors):
        patch.set_facecolor(color)

    # 공통 설정
    for ax in axs:
        ax.yaxis.grid(True)
        ax.set_xticks([y + 1 for y in range(len(chartdata))])
        ax.set_xlabel('성별')
        ax.set_ylabel('근속연수 (연 단위)')
        ax.set_xticklabels(xtick_label)


    plt.savefig(dataOut + file_name, dpi=400)
# end def MakeBoxPlotChart02

MakeBoxPlotChart02(df, 'g02.boxplot_chart.png')
###############################################################################


###############################################################################

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
