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
def MakeLineChart01(region_data, file_name, YTICKS_INTERVAL=20):
    """
    지역별 건수 데이터를 이용하여 선 그래프(line chart)를 그리고 파일로 저장

    Parameters
    ----------
    region_data : pandas.Series
        지역별 건수 데이터 (index: 지역, values: 건수)
    file_name : str
        저장할 파일명
    YTICKS_INTERVAL : int, optional
        y축 눈금 간격 (기본값 50)
    """
    # 최대값 계산
    maxlim = (int(region_data.max() / YTICKS_INTERVAL) + 1) * YTICKS_INTERVAL

    # 그래프 생성
    plt.figure(figsize=(8, 5))
    plt.plot(region_data, color='blue', linestyle='solid', marker='o')

    # y축 설정
    values = np.arange(0, maxlim + 1, YTICKS_INTERVAL)
    plt.yticks(values, ['%s' % format(val, ',') for val in values])

    # 그리드, 레이블, 제목 설정
    plt.grid(True)
    plt.xlabel('지역')
    plt.ylabel('고객 수')
    plt.title('지역별 고객 수')

    # 파일 저장
    plt.tight_layout()
    plt.savefig(dataOut + file_name, dpi=400)
    plt.close()

# end def MakeLineChart01

region_counts = df['지역'].value_counts()
MakeLineChart01(region_counts, 'i01.line_chart.png')

###############################################################################

###############################################################################

###############################################################################

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
