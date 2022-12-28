import pandas as pd
import os

input_data = pd.read_csv('pre_data1.csv',index_col='고객번호').drop(columns = 'Unnamed: 0')
mf_data = pd.read_csv('mf_data.csv', index_col = '고객번호')

input_data_filter = input_data.groupby(['고객번호','상품코드'])['구매수량'].agg('count')
input_data_filter = input_data_filter.unstack('상품코드').fillna(0)

while 1:
    want = input('\nVV 추천해주고 싶은 고객의 고객번호를 입력하시오(종료를 원하시면 [종료]두글자를 입력하시오) : ')
    if want == '종료':
        print('XXXXX추천 시스템을 종료합니다.XXXXX')
        break
    count = 0
    recommend = {}

    for f, o in zip((mf_data.loc[want]).values,(input_data_filter.loc[want]).values):
        if f >= 0 and o == 0:
            recommend[input_data_filter.columns[count]] = f
        count += 1

    recommend = sorted(recommend.items(), reverse=True, key = lambda item: item[1])
    print("***** 추천 상품 목록 10개 *****\n")

    print("\n".join(str(x[0]) for x in recommend[:10]))

