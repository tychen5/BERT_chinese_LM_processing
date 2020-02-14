import pandas as pd
import numpy as np
import pickle
from collections import Counter


def sentiment_score(tok_news_df, four_algos, by='month'):
    tok_news_df['input_id'] = tok_news_df.index

    def convert_date(s):
        return s.split('/')[take]

    if by == 'month':
        take = 1
        four_algos['Month'] = four_algos.Date.map(convert_date)
    elif by == 'day':
        take = 2
        four_algos['Day'] = four_algos.Date.map(convert_date)
    else:
        print('setting failed...please check!')
        print('set default as month...')
        take = 1
        four_algos['Month'] = four_algos.Date.map(convert_date)

    four_algos['final_sentiment'] = (four_algos['tb_score'] * 2.3 + four_algos['sn_score'] * 0.9 + four_algos[
        'bx_score'] * 3 + four_algos['senta_score'] * 0.6) / 6.8
    four_algos1 = four_algos[four_algos.isna().any(axis=1)]
    print(four_algos1)
    col1 = four_algos.tb_score.mean()
    col2 = four_algos.final_sentiment.mean()
    naid_li = four_algos1.input_id.tolist()
    for naid in naid_li:
        if four_algos.loc[naid].isna()['tb_score']:  # deal with possible error in TextBlob pkg (missing value)
            four_algos.loc[naid, 'tb_score'] = ((four_algos.loc[naid, 'sn_score'] + 2.1 * four_algos.loc[
                naid, 'bx_score'] + 0.9 * four_algos.loc[naid, 'senta_score']) / 4 * 3 + col1 + col2) / 5
            if four_algos.loc[naid].isna()['senta_score']:
                four_algos.loc[naid, 'senta_score'] = (four_algos.loc[naid, 'bx_score'] + four_algos.loc[
                    naid, 'tb_score']) / 2
            four_algos.loc[naid, 'final_sentiment'] = (four_algos.loc[naid, 'tb_score'] * 2 + four_algos.loc[
                naid, 'sn_score'] + four_algos.loc[naid, 'bx_score'] * 2.3 + four_algos.loc[
                                                           naid, 'senta_score'] * 0.8) / 6.1
        elif four_algos.loc[naid].isna()['senta_score']:
            four_algos.loc[naid, 'senta_score'] = (four_algos.loc[naid, 'bx_score'] + four_algos.loc[naid, 'tb_score'])/2
            four_algos.loc[naid, 'final_sentiment'] = (four_algos['tb_score'] * 2.3 + four_algos['sn_score'] * 0.9 + four_algos[
        'bx_score'] * 3 + four_algos['senta_score'] * 0.6) / 6.8
        else:
            print("fillna error, because the nan column is not TextBlob slot! => ", four_algos.loc[naid])
            print("please check why there's empty value in some other fields")
    assert len(four_algos[four_algos.isna().any(axis=1)]) == 0

    final_algo = four_algos.filter(['input_id', 'Month', 'final_sentiment'])
    final_df = pd.merge(tok_news_df, final_algo, on='input_id', how='outer')  # left_on='Unnamed: 0', right_
    final_df = final_df.drop(['input_id'], axis=1)
    if take == 1:
        senti_df = final_df.filter(['Month', 'final_sentiment'])
        # print(final_df.columns)
        # print(senti_df.columns)
        senti_df = senti_df.groupby('Month').mean()
        senti_df.reset_index(inplace=True)
        senti_df['Month'] = senti_df['Month'].astype('int')
        senti_df = senti_df.sort_values(by=['Month'])
        senti_df.reset_index(inplace=True, drop=True)
        return senti_df,final_df
    else:
        print('Not Implement Yet!')
