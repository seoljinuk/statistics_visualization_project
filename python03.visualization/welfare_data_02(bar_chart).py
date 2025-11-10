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

df = pd.read_csv(dataIn+ '고객복지데이터셋Cleaned.csv')

print("\n막대(Bar) 그래프")
employment_type = df['고용형태']

print("\n데이터의 고유값:", employment_type.unique())
print("데이터의 고유값 개수:", employment_type.nunique())

# 빈도수 출력
print("\n데이터의 빈도수:")
employment_type_value_counts = employment_type.value_counts()
print(employment_type_value_counts)

###############################################################################
# plt.bar() 메소드를 사용한 막대 그래프
def MakeBarChart01(x, y, color, xlabel, ylabel, title):
    plt.figure()
    plt.bar(x, y, color=color, alpha=0.7)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    # plt.grid(True)

    YTICKS_INTERVAL = 50

    maxlim = (int(y.max() / YTICKS_INTERVAL) + 1) * YTICKS_INTERVAL
    print(maxlim)

    values = np.arange(0, maxlim + 1, YTICKS_INTERVAL)

    plt.yticks(values, ['%s' % format(val, ',') for val in values])

    # 그래프 위에 건수와 비율 구하기
    ratio = 100 * y / y.sum()
    print(ratio)
    print('-' * 40)

    plt.rc('font', size=6)
    for idx in range(y.size):
        value = format(y.iloc[idx], ',') + '건'# 예시 : 60건
        ratioval = '%.1f%%' % (ratio.iloc[idx])  # 예시 : 20.0%
        # 그래프의 위에 "건수" 표시
        plt.text(x=idx, y=y.iloc[idx] + 1, s=value, horizontalalignment='center', fontsize=7)
        # 그래프의 중간에 비율 표시
        plt.text(x=idx, y=y.iloc[idx] / 2, s=ratioval, horizontalalignment='center', fontsize=7)

    # 평균 값을 수평선으로 그리기
    meanval = y.mean()
    print(meanval)
    print('-' * 40)

    average = '평균 : %d건' % meanval
    plt.axhline(y=meanval, color='r', linewidth=1, linestyle='dashed')
    plt.text(x=y.size - 1.5, y=meanval + 10, s=average, horizontalalignment='center', fontsize=7)

    file_name = 'd01.barchart_01.png'
    plt.savefig(dataOut + file_name, dpi=400)
    print(file_name + ' 파일이 저장되었습니다.')
# def MakeBarChart01

'''
그래프에 대한 색상을 지정하는 리스트입니다.
예시에서 "w"는 흰색이라서 제외하도록 합니다.
'''
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

mycolor = colors[0:len(employment_type)]

'''
데이터 프레임을 이용하여 막대 그래프를 그려 주는 함수를 호출합니다.
'''
MakeBarChart01(x=employment_type_value_counts.index, y=employment_type_value_counts, color=mycolor, xlabel='고용 형태', ylabel='인원수', title='고용 형태 분포')

subset_df_01 = df[['결혼상태', '정신건강지원']]
print(subset_df_01.head())

'''
데이터 프레임을 사용하여 막대 그래프를 그려 주는 함수입니다.
'''

###############################################################################
def MakeBarChart02(chartdata, rotation, title, file_name, ylim=None, stacked=False, yticks_interval=50):
    plt.figure()
    # 범례에 제목을 넣으려면 plot() 메소드의 legend 옵션을 사용해야 합니다.
    chartdata.plot(kind='bar', rot=rotation, title=title, legend=True, stacked=stacked, width=0.7)

    plt.legend(loc='best')
    plt.xlabel(chartdata.index.name, fontsize=7)  # 행 인덱스
    plt.ylabel(chartdata.columns.name, fontsize=7)  # 열 인덱스

    print('chartdata')
    print(chartdata)

    if stacked == False:
        # max(chartdata.max())은 항목들 값 중에서 최대 값을 의미합니다.
        maxlim = (int(max(chartdata.max()) / yticks_interval) + 1) * yticks_interval
        print('maxlim : ', maxlim)
        values = np.arange(0, maxlim + 1, yticks_interval)
        plt.yticks(values, ['%s' % format(val, ',') for val in values])
    else:  # 누적 막대 그래프
        # 국가별 누적 합인 chartdata.sum(axis=1))의 최대 값에 대한 연산이 이루어 져야 합니다.
        maxlim = (int(max(chartdata.sum(axis=1)) / yticks_interval) + 1) * yticks_interval
        print('maxlim : ', maxlim)
        values = np.arange(0, maxlim + 1, yticks_interval)
        plt.yticks(values, ['%s' % format(val, ',') for val in values])

    # y축의 상하한 값이 주어 지는 경우에만 설정합니다.
    if ylim != None:
        plt.ylim(ylim)

    plt.savefig(dataOut + file_name, dpi=400)
# end def MakeBarChart02

# def MakeBarChart02
# 교차표 생성
cross_df_01 = pd.crosstab(index=subset_df_01['정신건강지원'],
                       columns=subset_df_01['결혼상태'])

cross_df_01.index = ['아니오', '예']
cross_df_01.index.name = '정신건강지원'
print(cross_df_01)

MakeBarChart02(chartdata=cross_df_01, rotation=0, title="'정신건강지원' '결혼상태' 발생 건수", file_name='d01.barchart_02.png')

# 전치 프레임을 그래프로 그려 보기
cross_df_T_01 = cross_df_01.T
MakeBarChart02(chartdata=cross_df_T_01, rotation=0, title="'결혼상태' '정신건강지원' 발생 건수", file_name='d01.barchart_03.png')
###############################################################################
print("\n'사별'과 '이혼'은 제거하기")
cross_df_T_01 = cross_df_T_01.drop(['사별', '이혼'], axis=0)
print(cross_df_T_01.head())

ymax = cross_df_T_01.sum(axis=1)
ymaxlimit = ymax.max() + 10

MakeBarChart02(chartdata=cross_df_T_01, rotation=0, title="'정신건강지원' '결혼상태' 발생 건수(누적)", file_name='d01.barchart_04.png', ylim=[0, ymaxlimit], stacked=True, yticks_interval=50000)
###############################################################################
def MakeBarChart03(chartdata, title='수평 누적 막대 그래프', file_name='chart.png'):
    """
    Parameters
    ----------
    chartdata : pandas.DataFrame
        행은 범주(label), 열은 항목(category)으로 구성된 데이터프레임.
    title : str
        그래프 제목
    file_name : str
        저장할 파일 이름 (예: 'chart.png')
    """

    # 데이터 확인
    print("입력 데이터:")
    print(chartdata)

    # DataFrame을 numpy로 변환
    labels = chartdata.index.tolist()
    column_names = chartdata.columns.tolist()
    data = chartdata.to_numpy()

    # 누적합 계산
    data_cum = data.cumsum(axis=1)
    category_colors = plt.get_cmap('RdYlGn')(np.linspace(0.15, 0.85, data.shape[1]))

    # 그래프 설정
    fig, ax = plt.subplots(figsize=(9.2, 5))
    ax.invert_yaxis()  # 상단부터 표시
    ax.xaxis.set_visible(True)
    ax.set_xlim(0, np.sum(data, axis=1).max() * 1.1)

    # 막대 그리기
    for i, (colname, color) in enumerate(zip(column_names, category_colors)):
        widths = data[:, i]
        starts = data_cum[:, i] - widths
        ax.barh(labels, widths, left=starts, height=0.5, label=colname, color=color)

        # 막대 위의 값 표시
        xcenters = starts + widths / 2
        for y, (x, c) in enumerate(zip(xcenters, widths)):
            ax.text(x, y, str(int(c)), ha='center', va='center',
                    color='white', fontsize=10, fontweight='bold')

    # 범례, 제목, 라벨 설정
    ax.legend(ncol=len(column_names), bbox_to_anchor=(0, 1),
              loc='lower left', fontsize='small')
    ax.set_title(title, fontsize=14, pad=20)
    ax.set_xlabel('응답 수', fontsize=12)
    ax.set_ylabel('복지등급', fontsize=12)

    plt.tight_layout()
    plt.savefig(dataOut +  file_name, dpi=400)

    return fig, ax

# end def MakeBarChart03

print("\nt수평 누적 막대 그래프")
subset_df_02 = df[['복지등급', '웰니스참여']]
cross_df_02 = pd.crosstab(index=subset_df_02['복지등급'],
                       columns=subset_df_02['웰니스참여'])
print(cross_df_02)
MakeBarChart03(cross_df_02.T, title='복지등급별 웰니스 참여 현황', file_name='d01.barchart_04.png')
###############################################################################
def MakeBarChart04(chartdata, suptitle, file_name='chart04.png'):
    """
    Parameters
    ----------
    chartdata : pandas.DataFrame or Series
        막대그래프로 표시할 데이터
    suptitle : str
        그래프 전체 제목
    file_name : str
        저장할 파일 이름 (기본값: chart04.png)
    """
    # Series 형태인 경우 DataFrame으로 변환
    if isinstance(chartdata, pd.Series):
        chartdata = chartdata.to_frame(name='인원수')

    # 서브플롯 (2행 1열)
    plt.clf()  # 🔹 기존 그래프 초기화
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

    # 수직 막대그래프
    chartdata.plot(kind='bar', ax=axes[0], rot=0, alpha=0.7, color='skyblue', legend=False)
    axes[0].set_title('지역별 인원수 (수직 막대)', fontsize=12)
    axes[0].set_ylabel('인원수')

    # 수평 막대그래프
    chartdata.plot(kind='barh', ax=axes[1], color='m', alpha=0.7, legend=False)
    axes[1].set_title('지역별 인원수 (수평 막대)', fontsize=12)
    axes[1].set_xlabel('인원수')

    # 전체 제목
    fig.suptitle(suptitle, fontsize=14, fontweight='bold', y=0.95)

    plt.tight_layout()
    plt.savefig(dataOut + file_name, dpi=400)
# end def MakeBarChart04

print("\n서브 플로팅하기")
region_counts = df['지역'].value_counts()
MakeBarChart04(region_counts, suptitle='지역별 인원수 비교', file_name='d01.barchart_05.png')
###############################################################################
def MakeBarChart05(chartdata, suptitle, file_name='chart_table.png', yticks_interval=10):
    """
    테이블이 함께 표시되는 막대 그래프 생성 함수

    Parameters
    ----------
    chartdata : pd.DataFrame
        행: 그룹 (예: 지역)
        열: 카테고리 (예: 학력)
    suptitle : str
        그래프 상단 제목
    file_name : str, optional
        저장할 파일명 (기본값: 'chart_table.png')
    yticks_interval : int, optional
        y축 눈금 간격 (기본값: 10)
    """

    # 기존 그래프 잔상 제거
    plt.clf()

    # 학력 순서 재정렬
    order = ['고졸', '전문학사', '학사', '석사', '박사']
    chartdata = chartdata[order]

    # 인덱스(행: 지역), 컬럼(열: 학력)
    rows = list(chartdata.index)
    columns = list(chartdata.columns)

    n_rows = len(rows)
    left_margin = 0.3
    index = np.arange(len(columns)) + left_margin
    bar_width = 1 - 2 * left_margin

    # 누적용 초기값 (y_offset)
    y_offset = np.zeros(len(columns))

    cell_text = []
    plt.figure(figsize=(9, 7))

    # 각 지역별 누적 막대그래프
    for row in chartdata.index:
        values = chartdata.loc[row].tolist()
        plt.bar(index, values, bar_width, bottom=y_offset, label=row)
        y_offset += values
        cell_text.append([format(x, ',') for x in values])

    # 테이블은 위에서부터 아래로 가므로 반전 필요
    cell_text.reverse()
    rows.reverse()

    # 테이블 추가
    the_table = plt.table(
        cellText=cell_text,
        rowLabels=rows,
        colLabels=columns,
        loc='bottom'
    )

    plt.legend(loc='best')
    plt.subplots_adjust(left=0.15, bottom=0.25)
    plt.ylabel("인원 수")
    plt.title(suptitle)

    # y축 눈금 계산
    maxlim = (int(y_offset.max() / yticks_interval) + 1) * yticks_interval
    values = np.arange(0, maxlim + 1, yticks_interval)
    plt.yticks(values, [f"{val:,}" for val in values])
    plt.xticks([])

    # 그래프 저장
    plt.savefig(dataOut + file_name, dpi=400)
# end def MakeBarChart05

print("\n테이블이 존재하는 막대 그래프")
subset_df_03 = df[['지역', '학력']]
cross_df_03 = pd.crosstab(index=subset_df_03['지역'],
                       columns=subset_df_03['학력'])
print(cross_df_03)
MakeBarChart05(cross_df_03, "지역별 학력 분포 (테이블 포함)", "d01.barchart_06.png")
###############################################################################

# print("\n막대(Bar) 그래프")
# plt.figure(figsize=(6,4))
# sns.countplot(x=employment_type, hue=employment_type, palette='pastel')  # 색상 변경 가능
# plt.title('고용 형태 분포(기본 세로형)')
# plt.xlabel('고용 형태')
# plt.ylabel('빈도수')
# plt.savefig('d01.countplot_plot_01.png')
#
# plt.figure(figsize=(6,4))
# sns.countplot(y=employment_type, hue=employment_type, palette='pastel')  # 색상 변경 가능
# plt.title('고용 형태 분포(가로형)')
# plt.xlabel('고용 형태')
# plt.ylabel('빈도수')
# plt.savefig('d02.countplot_plot_02.png')
