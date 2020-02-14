import pandas as pd
import numpy as np
import pickle
from collections import Counter


def sentiment_score(tok_news_df, four_algos, by='month'):
    tok_news_df['input_id'] = tok_news_df.index
    define_dict_path = "./TokSentLeo/user_dict/company_dict.txt"
    company_list = []
    with open(define_dict_path, "r", encoding='utf8') as file:
        for line in file:
            key, value = line.split()
            company_list.append(str(key))
    company_list.extend(['說明', '表示', '永豐', '李永豐', '演出', '目前', '今天', '今年', '時間', '公司', '台北', '永豐銀', '專案',
                         '留言', '使用', '去年', '金控', '只要', '台灣', '推出', '登入', '功能', '以上', '明年', '金幣',
                         '大獎', '管理', '閱讀', '可以', '肌力', '積極', '亞洲', '第二', 'qr', '小時', '運用', '機會', '配合', '太陽', '建置',
                         '規劃', '規模', '新年', '臺北', '展望', '盤中', '現場', '應用', '一方面', '結合', '透過', '經營', '包括',
                         'qi', '新台幣', '第三', '國內', '信用', '登錄'])  # hand-made black list

    def remove_company(s):
        """
        Input: Series of keyWord_algorithm
        Ouput: remove not needed words
        """
        words = s.split("、")
        need_li = []
        for word in words:
            if word not in company_list:
                need_li.append(word)
        return need_li  # "、".join(need_li)


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
            four_algos.loc[naid, 'senta_score'] = (four_algos.loc[naid, 'bx_score'] + four_algos.loc[
                naid, 'tb_score']) / 2
            four_algos.loc[naid, 'final_sentiment'] = (four_algos['tb_score'] * 2.3 + four_algos['sn_score'] * 0.9 +
                                                       four_algos[
                                                           'bx_score'] * 3 + four_algos['senta_score'] * 0.6) / 6.8
        else:
            print("fillna error, because the nan column is not TextBlob slot! => ", four_algos.loc[naid])
            print("please check why there's empty value in some other fields")
    assert len(four_algos[four_algos.isna().any(axis=1)]) == 0

    final_algo = four_algos.filter(['input_id', 'Month', 'final_sentiment'])
    final_df = pd.merge(tok_news_df, final_algo, on='input_id', how='outer')  # left_on='Unnamed: 0', right_
    final_df = final_df.drop(['input_id'], axis=1)
    neg_df = final_df[final_df['final_sentiment'] < -0.25]
    neg_df['neg_words'] = neg_df.keyWord_algorithm.map(remove_company)
    all_neg_words = neg_df['neg_words'].tolist()
    all_neg_words = sum(all_neg_words, [])
    neg_word_freq = dict(Counter(all_neg_words))
    take_neg_word = []
    for k, v in neg_word_freq.items():
        if (v > 1) and (k not in company_list):
            take_neg_word.append(k)
    # print(take_neg_word)

    def take_neg_words(s):
        """
        Remove not needed words of neg_df
        """
        take = []
        for word in s:
            if (word in take_neg_word) and (word not in company_list):
                take.append(word)
        return take

    neg_df['neg_words'] = neg_df.neg_words.map(take_neg_words)
    neg_df = neg_df.filter(['Month', 'neg_words'])
    neg_month_df = neg_df.groupby('Month')['neg_words'].apply(list).reset_index(drop=False)
    neg_month_df = pd.DataFrame(neg_month_df)

    def combine_li(s):
        """
        combine list of list to list
        """
        s_ = sum(s, [])
        return "、".join(s_)

    neg_month_df['neg_words'] = neg_month_df.neg_words.map(combine_li)
    neg_month_df.Month = neg_month_df.Month.astype('int32')

    ##Positive word
    pos_df = final_df[final_df['final_sentiment'] > 0.2]
    pos_df['pos_words'] = pos_df.keyWord_algorithm.map(remove_company)
    all_pos_words = pos_df['pos_words'].tolist()
    all_pos_words = sum(all_pos_words, [])
    pos_word_freq = dict(Counter(all_pos_words))
    take_pos_word = []
    for k, v in pos_word_freq.items():
        if (v > 1) and (k not in company_list):
            take_pos_word.append(k)
    # print(take_pos_word)
    def take_pos_words(s):
        '''
        Remove not needed words of neg_df
        '''
        take = []
        for word in s:
            if (word in take_pos_word) and (word not in company_list):
                take.append(word)
        return take

    pos_df['pos_words'] = pos_df.pos_words.map(take_pos_words)
    pos_df = pos_df.filter(['Month', 'pos_words'])
    pos_month_df = pos_df.groupby('Month')['pos_words'].apply(list).reset_index(drop=False)
    pos_month_df = pd.DataFrame(pos_month_df)
    pos_month_df['pos_words'] = pos_month_df.pos_words.map(combine_li)
    pos_month_df.Month = pos_month_df.Month.astype('int32')
    if take == 1:
        senti_df = final_df.filter(['Month', 'final_sentiment'])
        # print(final_df.columns)
        # print(senti_df.columns)
        senti_df = senti_df.groupby('Month').mean()
        senti_df.reset_index(inplace=True)
        senti_df['Month'] = senti_df['Month'].astype('int')
        senti_df = senti_df.sort_values(by=['Month'])
        senti_df.reset_index(inplace=True, drop=True)
        senti_df = senti_df.merge(neg_month_df, on='Month', how='outer')
        senti_df = senti_df.merge(pos_month_df, on='Month', how='outer')
        return senti_df, final_df
    else:
        print('Not Implement Yet!')
