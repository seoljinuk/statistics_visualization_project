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

###############################################################################
def MakeScatterChart01(chartdata, file_name):
    """
    소득과 나이 간의 관계를 산점도로 시각화
    """
    print('소득과 나이 간의 산점도 그래프를 그려 봅니다.')

    # -------------------------------------------------------------------------
    # Step 1. 데이터 준비
    # -------------------------------------------------------------------------
    # 교차표 형태를 산점도용으로 변환
    df_reset = chartdata.reset_index()
    df_melted = df_reset.melt(id_vars='소득', var_name='나이', value_name='빈도수')

    # 빈도수가 0인 데이터는 제외
    df_melted = df_melted[df_melted['빈도수'] > 0]

    # -------------------------------------------------------------------------
    # Step 2. 시각화 스타일 및 폰트 설정
    # -------------------------------------------------------------------------
    plt.rc('font', family='Malgun Gothic')
    plt.style.use('ggplot')

    # -------------------------------------------------------------------------
    # Step 3. 산점도 생성
    # -------------------------------------------------------------------------
    plt.figure(figsize=(8, 5))
    plt.scatter(df_melted['나이'], df_melted['소득'],
                s=df_melted['빈도수'] * 10,  # 빈도수 크기에 따라 점 크기 조절
                c='steelblue', alpha=0.7, edgecolors='k')

    # -------------------------------------------------------------------------
    # Step 4. 그래프 제목, 축 라벨, 격자
    # -------------------------------------------------------------------------
    plt.title('나이와 소득 간의 관계 (산점도)', fontsize=13)
    plt.xlabel('나이', fontsize=11)
    plt.ylabel('소득', fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.6)

    # -------------------------------------------------------------------------
    # Step 5. 저장 및 닫기
    # -------------------------------------------------------------------------
    plt.tight_layout()
    plt.savefig(dataOut + file_name, dpi=400)
    print(dataOut + file_name + ' 파일이 저장되었습니다.')
    plt.close()

# end def MakeScatterChart01

subset_df_05 = df[['나이', '소득', '성별']]
cross_df_05 = pd.crosstab(index=subset_df_05['소득'],
                       columns=subset_df_05['나이'])
# print(cross_df_05.head())

MakeScatterChart01(cross_df_05, 'f01_scatter_01.png')
###############################################################################
def MakeScatterChart02(chartdata, file_name):
    """
    성별에 따라 색상을 구분하여 나이-소득 관계를 시각화하는 산점도 그래프 생성 함수

    Parameters
    ----------
    chartdata : pd.DataFrame
        '나이', '소득', '성별' 컬럼을 포함하는 데이터프레임
    file_name : str
        저장할 파일명 (예: 'e01_scatter_02.png')
    """
    print('성별에 따른 나이-소득 산점도를 그려 봅니다.')

    plt.rc('font', family='Malgun Gothic')
    plt.figure(figsize=(7, 5))
    plt.style.use('ggplot')

    # 색상 및 성별 한글 레이블 매핑
    mycolors = ['r', 'g']
    gender_dict = {'남성': '남성', '여성': '여성'}

    labels = chartdata['성별'].dropna().unique()
    idx = 0

    for gender in labels:
        xdata = chartdata.loc[chartdata['성별'] == gender, '나이']
        ydata = chartdata.loc[chartdata['성별'] == gender, '소득']
        plt.plot(xdata, ydata,
                 color=mycolors[idx % len(mycolors)],
                 marker='o',
                 linestyle='None',
                 label=gender_dict.get(gender, gender))
        idx += 1

    plt.legend(title='성별')
    plt.xlabel("나이")
    plt.ylabel("소득")
    plt.title("성별에 따른 나이-소득 산점도")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(dataOut + file_name, dpi=400)
# end def MakeScatterChart02

subset_df_06 = df[['나이', '소득', '성별']]
MakeScatterChart02(subset_df_06, 'f01_scatter_02.png')
###############################################################################
def MakeScatterChart03(chartdata, file_name):
    """
    산점도 + 주변 히스토그램을 그리는 함수
    - x축: 나이
    - y축: 소득
    - 색상: 성별
    """
    # 색상 팔레트 설정
    gender_colors = {'남성': 'green', '여성': 'red'}

    # Figure 및 grid 생성
    fig = plt.figure(figsize=(16, 10), dpi=100)
    grid = plt.GridSpec(4, 4, hspace=0.5, wspace=0.2)

    # 메인 및 보조 축
    ax_main = fig.add_subplot(grid[:-1, :-1])
    ax_right = fig.add_subplot(grid[:-1, -1], xticklabels=[], yticklabels=[])
    ax_bottom = fig.add_subplot(grid[-1, 0:-1], xticklabels=[], yticklabels=[])

    # 산점도 (성별별 색상 구분)
    for gender, color in gender_colors.items():
        subset = chartdata[chartdata['성별'] == gender]
        ax_main.scatter(subset['나이'], subset['소득'],
                        label=gender, alpha=0.8, edgecolor='gray',
                        s=70, color=color)

    # 하단 histogram (나이 분포)
    for gender, color in gender_colors.items():
        subset = chartdata[chartdata['성별'] == gender]
        ax_bottom.hist(subset['나이'], bins=30, color=color, alpha=1.0)

    # 오른쪽 histogram (소득 분포)
    for gender, color in gender_colors.items():
        subset = chartdata[chartdata['성별'] == gender]
        ax_right.hist(subset['소득'], bins=30, color=color, alpha=1.0, orientation='horizontal')

    # 하단 축 반전 (디자인 통일)
    ax_bottom.invert_yaxis()

    # 제목 및 라벨 설정
    ax_main.set(title='산점도 (나이 vs 소득)', xlabel='나이', ylabel='소득')
    ax_main.title.set_fontsize(20)
    ax_main.legend(title='성별', fontsize=12, title_fontsize=13)

    # 폰트 크기 조정
    for item in ([ax_main.xaxis.label, ax_main.yaxis.label] + ax_main.get_xticklabels() + ax_main.get_yticklabels()):
        item.set_fontsize(13)

    # 저장 및 출력
    plt.savefig(dataOut + file_name, dpi=400)
# end def MakeScatterChart03

subset_df_07 = df[['나이', '소득', '성별']]
MakeScatterChart03(subset_df_07, 'f01_scatter_03.png')
###############################################################################
'''
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