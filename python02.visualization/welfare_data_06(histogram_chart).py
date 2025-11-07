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
def MakeHistogramChart01(series, file_name):
    """
    전달받은 Series 데이터를 이용해 히스토그램을 그리고,
    정규분포 곡선을 함께 표시한 후, 지정된 파일명으로 저장합니다.

    Parameters
    ----------
    series : pandas.Series
        히스토그램으로 표현할 수치형 데이터
    save_filename : str
        저장할 이미지 파일 이름 (예: 'dd.png')
    """

    # 한글 폰트 설정
    plt.rc('font', family='Malgun Gothic')
    plt.rcParams['axes.unicode_minus'] = False

    # 데이터 확인
    if not isinstance(series, (list, np.ndarray)) and not hasattr(series, "values"):
        raise TypeError("첫 번째 매개변수는 pandas Series 또는 리스트여야 합니다.")

    x = np.array(series)
    num_bins = 20

    # 그래프 객체 생성
    fig, ax = plt.subplots(figsize=(7, 5))

    # 히스토그램
    n, bins, patches = ax.hist(x, num_bins, density=True,
                               color='skyblue', edgecolor='black', alpha=0.7)

    # 평균과 표준편차 계산
    mu = x.mean()
    sigma = x.std()
    print(f"평균(mu): {mu:.3f}")
    print(f"표준편차(sigma): {sigma:.3f}")

    # 정규분포 곡선 추가
    y = ((1 / (np.sqrt(2 * np.pi) * sigma)) *
         np.exp(-0.5 * ((bins - mu) / sigma) ** 2))
    ax.plot(bins, y, '--', color='red', label='정규분포 곡선')

    # 제목, 라벨 설정
    ax.set_title('근속연수 히스토그램', fontsize=13)
    ax.set_xlabel('근속연수 (Tenure)')
    ax.set_ylabel('밀도 (Density)')
    ax.legend()

    # 여백 조정 및 저장
    fig.tight_layout()
    plt.savefig(dataOut + file_name, dpi=400)
# end def MakeHistogramChart01

print("\n근속연수 히스토그램")
tenure = df['근속연수']
MakeHistogramChart01(tenure, 'h01.histogram_chart.png')
###############################################################################
def MakeHistogramChart02(dataframe, file_name):
    """
    df 데이터프레임에서 '근속연수'와 '성별' 컬럼을 사용하여
    성별에 따른 근속연수 분포 히스토그램을 그린 후 저장합니다.

    Parameters
    ----------
    df : pandas.DataFrame
        '근속연수'와 '성별' 컬럼을 포함하는 데이터프레임
    file_name : str
        저장할 이미지 파일 이름 (예: 'tenure_gender.png')
    """
    # 컬럼 존재 여부 확인
    if not {'근속연수', '성별'}.issubset(df.columns):
        raise KeyError("데이터프레임에 '근속연수' 또는 '성별' 컬럼이 없습니다.")

    # 성별별 데이터 추출
    male_tenure = dataframe[dataframe['성별'] == '남성']['근속연수']
    female_tenure = dataframe[dataframe['성별'] == '여성']['근속연수']

    # 그래프 설정
    plt.figure(figsize=(7, 5))
    plt.hist(male_tenure, bins=20, alpha=0.6, color='skyblue', label='남성', edgecolor='black')
    plt.hist(female_tenure, bins=20, alpha=0.6, color='lightcoral', label='여성', edgecolor='black')

    # 제목, 라벨
    plt.title('성별에 따른 근속연수 분포', fontsize=13)
    plt.xlabel('근속연수 (Tenure)')
    plt.ylabel('빈도수 (Frequency)')
    plt.legend()
    plt.grid(alpha=0.3)

    # 저장
    plt.tight_layout()
    plt.savefig(dataOut + file_name, dpi=400)
# end def MakeHistogramChart02


MakeHistogramChart02(df, 'h02.histogram_chart.png')
###############################################################################
def MakeHistogramChart03(human, file_name_01, file_name_02):
    """
    거인국과 소인국 데이터를 이용하여 히스토그램을 그립니다.
    1) 별개의 subplot에 각각 히스토그램
    2) 같은 plot에 두 히스토그램 겹치기
    파일로 저장하며, 저장 파일명은 cnt_start부터 순차적으로 증가

    Parameters
    ----------
    human : pandas.DataFrame
        '거인국'과 '소인국' 컬럼을 가진 데이터프레임
    cnt_start : int
        저장 파일명 번호 시작 값
    """

    giant = human['거인국']
    dwarf = human['소인국']

    # 1) 별개의 subplot에 히스토그램
    print('\n# 별개의 데이터에 대한 histogram 서브 플로팅')
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

    axes[0].hist(giant, range=(210, 290), bins=20, alpha=0.6, color='skyblue', edgecolor='black')
    axes[0].set_title('거인국의 키(height)')

    axes[1].hist(dwarf, range=(100, 180), bins=20, alpha=0.6, color='lightgreen', edgecolor='black')
    axes[1].set_title('소인국의 키(height)')

    plt.tight_layout()
    plt.savefig(dataOut + file_name_01, dpi=400)
    plt.close(fig)

    # 2) 같은 plot에 두 히스토그램 겹치기
    print('\n# 2개의 histogram 같이 그리기')
    fig, ax = plt.subplots(figsize=(7, 5))

    ax.hist(giant, bins=20, alpha=0.6, color='skyblue', label='거인국', edgecolor='black')
    ax.hist(dwarf, bins=20, alpha=0.6, color='lightgreen', label='소인국', edgecolor='black')

    ax.set_title('거인국 vs 소인국 히스토그램')
    ax.legend()

    plt.tight_layout()
    plt.savefig(dataOut + file_name_02, dpi=400)
    plt.close(fig)


# end def MakeHistogramChart03
'''
거인국: 키가 평균 250cm, 표준편차 10cm
소인국: 키가 평균 120cm, 표준편차 5cm
각각 1000명의 데이터를 생성
'''
np.random.seed(42) # 랜덤 시드 고정 (재현용)
giantland_height = np.random.normal(loc=250, scale=10, size=1000) # 거인국 데이터
tinyland_height = np.random.normal(loc=120, scale=5, size=1000)# 소인국 데이터

human = pd.DataFrame({
    '거인국': giantland_height,
    '소인국': tinyland_height
})

MakeHistogramChart03(human, 'h03.histogram_chart.png', 'h04.histogram_chart.png')
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
