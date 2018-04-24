import pandas as pd
import numpy as np
import sys
import os
import argparse
import nltk.data
import random
import re
from gensim.parsing import preprocessing
from multiprocessing import cpu_count, Pool

review_columns = ['user_id', 'stars', 'text']
user_columns = ['user_id', 'friends']
SVM_columns = ['user_id', 'stars', 'word_tokenize_and_remove_stop_words']
NN_columns = ['user_id', 'stars',
              'sentence_tokenize_and_word_tokenize_and_remove_stop_words']
swe_columns = ['user_id', 'word_tokenize']
w2v_columns = ['word_tokenize']

# those people who published reveiws less than 30 will be treated as unknown user, we do not learn user vector for them.
thre = 30
debug_number = None
output_path = '../data'
num_cores = cpu_count()
num_partitions = num_cores


def parallelize_dataframe(df, func, need_concat=False):
    total = df.shape[0]
    df_split = np.array_split(df, num_partitions)
    pool = Pool(num_cores)
    result = pool.map(func, df_split)
    if need_concat:
        df = pd.concat(result)
    pool.close()
    pool.join()
    print('%d / %d...\nFinished!\n' % (total, total))
    return df


def clean_user_data(df, user_set):
    df['user_is_known'] = df['user_id'].apply(lambda x: x in user_set)
    df = df[df['user_is_known']]
    df['known_friends'] = df['friends'].apply(
        lambda x: [y for y in eval(x) if y in user_set])
    df['known_friends_size'] = df['known_friends'].apply(len)
    return df


def generate_user_graph(user_set, input_path, yelp_round):
    user_file = os.path.join(output_path, 'user_file_rd%d.txt' % (yelp_round))
    user_graph = os.path.join(
        output_path, 'user_graph_rd%d.txt' % (yelp_round))
    user_weight = os.path.join(
        output_path, 'user_weight_rd%d.txt' % (yelp_round))
    print('generate the user_file')
    with open(user_file, 'w') as fo:
        for x in user_set:
            fo.write('%s\n' % x)
    print('generate the user_graph and user_weight')
    print('loading yelp_academic_dataset_user.csv...')
    yelp_user_df = pd.read_csv(os.path.join(
        input_path, 'yelp_academic_dataset_user.csv'), usecols=user_columns)
    print('generating the user_graph...')
    yelp_user_df = clean_user_data(yelp_user_df, user_set)
    fo = open(user_graph, 'w')
    for row in yelp_user_df.iterrows():
        if row[1]['known_friends_size'] > 0:
            fo.write("%s\t%d\t%s\n" % (
                row[1]['user_id'], row[1]['known_friends_size'], ' '.join(row[1]['known_friends'])))
    fo.close()


def split_review_rows(df):
    review_number = df.shape[0]
    user_review_total = df['user_id'].value_counts()
    user_review_total = user_review_total.to_frame().reset_index()
    user_review_total.columns = ['user_id', 'review_count']
    df = pd.merge(df, user_review_total, on='user_id')
    unknown_review_number = df[df['review_count'] < thre].shape[0]

    unknown_review_test_lowerbound = int(unknown_review_number * 0.1)
    unknown_review_dev_lowerbound = int(unknown_review_number * 0.2)
    unknown_review_cnt = 0

    row_cnt = 0
    cnt = 0
    user_total = 0

    new_df = pd.DataFrame(columns=['user_id', 'stars', 'text', 'd_type'])

    for row in df.iterrows():
        if row_cnt % 10000 == 0:
            print('%d / %d...' % (row_cnt, review_number))
        x = row[1]
        if cnt == user_total:
            cnt = 0
            user_total = x['review_count']
            if user_total < thre:
                test_lowerbound = unknown_review_test_lowerbound
                dev_lowerbound = unknown_review_dev_lowerbound
            else:
                test_lowerbound = int(user_total * 0.1)
                dev_lowerbound = int(user_total * 0.2)
            # print(known_review_test_lowerbound, known_review_dev_lowerbound, user_total)

        if user_total < thre:
            # df.loc[index, ['user_id']] = 'unknown_user_id'
            if unknown_review_cnt < test_lowerbound:
                new_df.loc[row_cnt, ['user_id', 'stars', 'text', 'd_type']] = [
                    'unknown_user_id', x['stars'], x['text'], 'test']
            elif unknown_review_cnt < dev_lowerbound:
                new_df.loc[row_cnt, ['user_id', 'stars', 'text', 'd_type']] = [
                    'unknown_user_id', x['stars'], x['text'], 'dev']
            else:
                new_df.loc[row_cnt, ['user_id', 'stars', 'text', 'd_type']] = [
                    'unknown_user_id', x['stars'], x['text'], 'train']
            unknown_review_cnt += 1
        else:
            if cnt < test_lowerbound:
                new_df.loc[row_cnt, ['user_id', 'stars', 'text', 'd_type']] = [
                    x['user_id'], x['stars'], x['text'], 'test']
            elif cnt < dev_lowerbound:
                new_df.loc[row_cnt, ['user_id', 'stars', 'text', 'd_type']] = [
                    x['user_id'], x['stars'], x['text'], 'dev']
            else:
                new_df.loc[row_cnt, ['user_id', 'stars', 'text', 'd_type']] = [
                    x['user_id'], x['stars'], x['text'], 'train']
        cnt += 1
        row_cnt += 1
    return new_df


def split_review(input_path, yelp_round):
    print('split the review data')
    print('loading yelp_academic_dataset_review.csv...')
    review_df = pd.read_csv(os.path.join(
        input_path, 'yelp_academic_dataset_review.csv'), usecols=review_columns, nrows=debug_number)
    review_df = review_df.sort_values('user_id')
    review_df['d_type'] = 0
    # 0: train, 1: test, 2: dev
    print('spliting:')
    review_df = parallelize_dataframe(
        review_df, split_review_rows, need_concat=True)

    user_set = set(review_df['user_id'].unique())
    user_set.remove('unknown_user_id')
    train_df = review_df[review_df['d_type'] == 'train']
    dev_df = review_df[review_df['d_type'] == 'dev']
    test_df = review_df[review_df['d_type'] == 'test']

    return train_df, dev_df, test_df, user_set


def word_tokenize(text):
    try:
        if (isinstance(text, str)):
            words = text.lower().split()
        else:
            words = str(text).lower().split()

        if len(words) == 0:
            return ''
        text = ' '.join(words)
        text = preprocessing.strip_punctuation(text)
        text = preprocessing.strip_non_alphanum(text)
        text = preprocessing.strip_numeric(text)
        text = preprocessing.strip_tags(text)
        text = preprocessing.strip_multiple_whitespaces(text)
        return text.encode('utf-8')
    except UnicodeDecodeError as e:
        return ''


def df_word_tokenize(df):
    row_number = df.shape[0]
    df['word_tokenize'] = df['text'].apply(word_tokenize)
    print('%d / %d...' % (row_number, row_number))
    return df


def word_tokenize_and_remove_stop_words(text, stop_word1, stop_word2):
    try:
        if isinstance(text, str):
            words = text.lower().split()
        else:
            words = str(text).lower().split()

        if len(words) == 0:
            return ''

        text = ' '.join(filter(lambda x: x not in stop_word1, words))
        text = preprocessing.strip_punctuation(text)
        text = preprocessing.strip_non_alphanum(text)
        text = preprocessing.strip_numeric(text)
        text = preprocessing.strip_tags(text)
        text = preprocessing.strip_multiple_whitespaces(text)
        words = text.split()
        if len(words) == 0:
            return ''

        text = ' '.join(filter(lambda x: x not in stop_word2, words))
        return text.encode('utf-8')
    except UnicodeDecodeError as e:
        return ''


def df_word_tokenize_and_remove_stop_words(df):
    stop_file = 'english_stop.txt'
    stop_word1 = []
    stop_word2 = []
    with open(stop_file, 'r') as fs:
        for l in fs:
            s = l.strip('\n')
            if "'" in s:
                stop_word1.append(s)
            else:
                stop_word2.append(s)
    row_number = df.shape[0]
    df['word_tokenize_and_remove_stop_words'] = df['text'].apply(
        word_tokenize_and_remove_stop_words, args=(stop_word1, stop_word2))
    print('%d / %d...' % (row_number, row_number))
    return df


def sentence_tokenize_and_word_tokenize_and_remove_stop_words(text, tokenizer, stop_word1, stop_word2):
    try:
        if isinstance(text, str):
            sentences = tokenizer.tokenize(text.lower())
        else:
            sentences = tokenizer.tokenize(str(text).lower())
    except UnicodeDecodeError as e:
        return ''
    if len(sentences) == 0:
        return ''
    text_total = ''
    for sentence in sentences:
        words = sentence.split()
        if len(words) == 0:
            continue
        text = ' '.join(filter(lambda x: x not in stop_word1, words))
        try:
            text = preprocessing.strip_punctuation(text)
            text = preprocessing.strip_non_alphanum(text)
            text = preprocessing.strip_numeric(text)
            text = preprocessing.strip_tags(text)
            text = preprocessing.strip_multiple_whitespaces(text)
            words = text.split()
            if len(words) == 0:
                continue
            text = ' '.join(filter(lambda x: x not in stop_word2, words))
            text_total = text_total + text.encode('utf-8') + '#'
        except UnicodeDecodeError as e:
            pass
    return text_total


def df_sentence_tokenize_and_word_tokenize_and_remove_stop_words(df):
    stop_file = 'english_stop.txt'
    stop_word1 = []
    stop_word2 = []
    with open(stop_file, 'r') as fs:
        for l in fs:
            s = l.strip('\n')
            if "'" in s:
                stop_word1.append(s)
            else:
                stop_word2.append(s)
    # -LCB-, -LRB-, -RCB-, -RRB-
    stop_word2.append('RRB')
    stop_word2.append('LRB')
    stop_word2.append('LCB')
    stop_word2.append('RCB')

    tokenizer = nltk.data.load('english.pickle')
    row_number = df.shape[0]
    df['sentence_tokenize_and_word_tokenize_and_remove_stop_words'] = df['text'].apply(
        sentence_tokenize_and_word_tokenize_and_remove_stop_words, args=(tokenizer, stop_word1, stop_word2))
    print('%d / %d...' % (row_number, row_number))
    return df


def SVM_preprocess(train_df, dev_df, test_df, yelp_round):
    print('SVM_preprocess...')
    train_file = os.path.join(output_path, 'SVM_train_rd%d.txt' % (yelp_round))
    dev_file = os.path.join(output_path, 'SVM_dev_rd%d.txt' % (yelp_round))
    test_file = os.path.join(output_path, 'SVM_test_rd%d.txt' % (yelp_round))
    for df, output_file in [(train_df, train_file), (dev_df, dev_file), (test_df, test_file)]:
        df['word_tokenize_and_remove_stop_words'] = ''
        df = parallelize_dataframe(
            df, df_word_tokenize_and_remove_stop_words, need_concat=True)
        df.to_csv(output_file, columns=SVM_columns,
                  index=False, header=False, sep='\t')


def NN_preprocess(train_df, dev_df, test_df, yelp_round):
    print('NN_preprocess...')
    train_file = os.path.join(output_path, 'NN_train_rd%d.txt' % (yelp_round))
    dev_file = os.path.join(output_path, 'NN_dev_rd%d.txt' % (yelp_round))
    test_file = os.path.join(output_path, 'NN_test_rd%d.txt' % (yelp_round))
    for df, output_file in [(train_df, train_file), (dev_df, dev_file), (test_df, test_file)]:
        df['sentence_tokenize_and_word_tokenize_and_remove_stop_words'] = ''
        df = parallelize_dataframe(
            df, df_sentence_tokenize_and_word_tokenize_and_remove_stop_words, need_concat=True)
        df.to_csv(output_file, columns=NN_columns,
                  index=False, header=False, sep='\t')


def Train_preprocess(train_df, yelp_round):
    swe_train = os.path.join(output_path, 'swe_train_rd%d.txt' % (yelp_round))
    w2v_train = os.path.join(output_path, 'w2v_train_rd%d.txt' % (yelp_round))
    print('Train_preprocess')
    df = train_df
    df['word_tokenize'] = ''
    df = parallelize_dataframe(df, df_word_tokenize, need_concat=True)
    df.to_csv(swe_train, columns=swe_columns,
              index=False, header=False, sep='\t')
    df.to_csv(w2v_train, columns=w2v_columns,
              index=False, header=False, sep='\t')


def SVM_one_fifth_data(yelp_round):
    for d_type in ['train', 'dev', 'test']:
        input_file = os.path.join(
            output_path, 'SVM_%s_rd%d.txt' % (d_type, yelp_round))
        output_file = os.path.join(
            output_path, 'SVM_%s_one_fifth_rd%d.txt' % (d_type, yelp_round))
        print('split %s to its one fifth' % (input_file))
        fin = open(input_file, 'r')
        fo = open(output_file, 'w')
        for i, line in enumerate(fin):
            if i % 5 == 0:
                fo.write(line.strip('\n'))
                fo.write('\n')


def NN_one_fifth_data(yelp_round):
    for d_type in ['train', 'dev', 'test']:
        input_file = os.path.join(
            output_path, 'NN_%s_rd%d.txt' % (d_type, yelp_round))
        output_file = os.path.join(
            output_path, 'NN_%s_one_fifth_rd%d.txt' % (d_type, yelp_round))
        print('split %s to its one fifth' % (input_file))
        fin = open(input_file, 'r')
        fo = open(output_file, 'w')
        for i, line in enumerate(fin):
            if i % 5 == 0:
                fo.write(line.strip('\n'))
                fo.write('\n')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Preprocess Yelp Data')

    parser.add_argument('--input', default=None, type=str,
                        help='Path to yelp dataset')

    parser.add_argument('--yelp_round', default=9, type=int, choices={9, 10},
                        help="The round number of yelp data")

    args = parser.parse_args()

    parser.print_help()

    train_df, dev_df, test_df, user_set = split_review(
        args.input, args.yelp_round)

    generate_user_graph(user_set, args.input, args.yelp_round)

    Train_preprocess(train_df, args.yelp_round)

    SVM_preprocess(train_df, dev_df, test_df, args.yelp_round)

    SVM_one_fifth_data(args.yelp_round)

    NN_preprocess(train_df, dev_df, test_df, args.yelp_round)

    NN_one_fifth_data(args.yelp_round)
