import sys
import os
import argparse
import numpy as np
import pandas as pd
from prettytable import PrettyTable
from multiprocessing import cpu_count

num_cores = cpu_count()

# model_types = ['w2v', 'swe', 'swe_with_randomwalk', 'swe_with_2nd_randomwalk', 'swe_with_bias_randomwalk', 'swe_with_deepwalk', 'swe_with_node2vec']
model_types = ['swe_with_bias_randomwalk']
factor_array = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
data_path = '../data'
embs_path = '../embs'
model_path = 'models'
result_path = 'results'
format_data_path = 'format_data'

begin_c = 0.015625
run_times = 10


def SVM_format(model_type, input_file, yelp_round, para_lambda, para_r, para_path, para_path_length, para_p, para_q, para_alpha, para_bias_weight):
    if model_type not in model_types:
        print('Error: not support %s' % (model_type))
        return ''
    basename = ''
    if input_file.find('head') != -1:
        basename += 'head'
    elif input_file.find('tail') != -1:
        basename += 'tail'
    if input_file.find('train') != -1:
        basename += 'train'
    elif input_file.find('test') != -1:
        basename += 'test'
    elif input_file.find('dev') != -1:
        basename += 'dev'
    basename += '_rd%d' % (yelp_round)
    lambda_str = str(para_lambda)
    lambda_index = lambda_str.index('.')
    lambda_str = lambda_str[0:lambda_index] + \
        'p' + lambda_str[lambda_index + 1:]
    r_str = str(para_r)
    r_index = r_str.index('.')
    r_str = r_str[0:r_index] + 'p' + r_str[r_index + 1:]
    p_str = str(para_p)
    p_index = p_str.index('.')
    p_str = p_str[0:p_index] + 'p' + p_str[p_index + 1:]
    q_str = str(para_q)
    q_index = q_str.index('.')
    q_str = q_str[0:q_index] + 'p' + q_str[q_index + 1:]
    alpha_str = str(para_alpha)
    alpha_index = alpha_str.index('.')
    alpha_str = alpha_str[0:alpha_index] + 'p' + alpha_str[alpha_index + 1:]
    bias_weight_str = str(para_bias_weight)
    bias_weight_index = bias_weight_str.index('.')
    bias_weight_str = bias_weight_str[0:bias_weight_index] + \
        'p' + bias_weight_str[bias_weight_index + 1:]

    if model_type == 'w2v':
        word_vec_file = os.path.join(
            embs_path, '%s_word_vec_rd%d.txt' % (model_type, yelp_round))
        format_file = os.path.join(
            format_data_path, 'F_%s_%s.txt' % (model_type, basename))
        if not os.path.isfile('./get_SVM_format_w2v'):
            command = 'gcc get_SVM_format_w2v.c -o get_SVM_format_w2v -lm -pthread -O3 -march=native -Wall -funroll-loops -Wno-unused-result'
            print(command)
            os.system(command)
        if not os.path.isfile(format_file):
            command = './get_SVM_format_w2v -input %s -word-vec %s -output %s' % (
                input_file, word_vec_file, format_file)
            print(command)
            os.system(command)
    elif model_type == 'swe':
        word_vec_file = os.path.join(embs_path, '%s_word_vec_rd%d_l%s_r%s.txt' % (
            model_type, yelp_round, lambda_str, r_str))
        user_vec_file = os.path.join(embs_path, '%s_user_vec_rd%d_l%s_r%s.txt' % (
            model_type, yelp_round, lambda_str, r_str))
        format_file = os.path.join(format_data_path, 'F_%s_%s_l%s_r%s.txt' % (
            model_type, basename, lambda_str, r_str))
        if not os.path.isfile('./get_SVM_format_swe'):
            command = 'gcc get_SVM_format_swe.c -o get_SVM_format_swe -lm -pthread -O3 -march=native -Wall -funroll-loops -Wno-unused-result'
            print(command)
            os.system(command)
        if not os.path.isfile(format_file):
            command = './get_SVM_format_swe -input %s -word-vec %s -user-vec %s -output %s' % (
                input_file, word_vec_file, user_vec_file, format_file)
            print(command)
            os.system(command)
    elif model_type.startswith('swe_with_randomwalk') or model_type.startswith('swe_with_deepwalk'):
        word_vec_file = os.path.join(embs_path, '%s_word_vec_rd%d_l%s_r%s_ph%d_pl%d.txt' % (
            model_type, yelp_round, lambda_str, r_str, para_path, para_path_length))
        user_vec_file = os.path.join(embs_path, '%s_user_vec_rd%d_l%s_r%s_ph%d_pl%d.txt' % (
            model_type, yelp_round, lambda_str, r_str, para_path, para_path_length))
        format_file = os.path.join(format_data_path, 'F_%s_%s_l%s_r%s_ph%d_pl%d.txt' % (
            model_type, basename, lambda_str, r_str, para_path, para_path_length))
        if not os.path.isfile('./get_SVM_format_swe'):
            command = 'gcc get_SVM_format_swe.c -o get_SVM_format_swe -lm -pthread -O3 -march=native -Wall -funroll-loops -Wno-unused-result'
            print(command)
            os.system(command)
        if not os.path.isfile(format_file):
            command = './get_SVM_format_swe -input %s -word-vec %s -user-vec %s -output %s' % (
                input_file, word_vec_file, user_vec_file, format_file)
            print(command)
            os.system(command)
    elif model_type.startswith('swe_with_node2vec') or model_type.startswith('swe_with_2nd_randomwalk'):
        word_vec_file = os.path.join(embs_path, '%s_word_vec_rd%d_l%s_r%s_ph%d_pl%d_p%s_q%s.txt' % (
            model_type, yelp_round, lambda_str, r_str, para_path, para_path_length, p_str, q_str))
        user_vec_file = os.path.join(embs_path, '%s_user_vec_rd%d_l%s_r%s_ph%d_pl%d_p%s_q%s.txt' % (
            model_type, yelp_round, lambda_str, r_str, para_path, para_path_length, p_str, q_str))
        format_file = os.path.join(format_data_path, 'F_%s_%s_l%s_r%s_ph%d_pl%d_p%s_q%s.txt' % (
            model_type, basename, lambda_str, r_str, para_path, para_path_length, p_str, q_str))
        if not os.path.isfile('./get_SVM_format_swe'):
            command = 'gcc get_SVM_format_swe.c -o get_SVM_format_swe -lm -pthread -O3 -march=native -Wall -funroll-loops -Wno-unused-result'
            print(command)
            os.system(command)
        if not os.path.isfile(format_file):
            command = './get_SVM_format_swe -input %s -word-vec %s -user-vec %s -output %s' % (
                input_file, word_vec_file, user_vec_file, format_file)
            print(command)
            os.system(command)
    elif model_type.startswith('swe_with_bias_randomwalk'):
        word_vec_file = os.path.join(embs_path, '%s_word_vec_rd%d_l%s_r%s_ph%d_pl%d_p%s_q%s_a%s_bw%s.txt' % (
            model_type, yelp_round, lambda_str, r_str, para_path, para_path_length, p_str, q_str, alpha_str, bias_weight_str))
        user_vec_file = os.path.join(embs_path, '%s_user_vec_rd%d_l%s_r%s_ph%d_pl%d_p%s_q%s_a%s_bw%s.txt' % (
            model_type, yelp_round, lambda_str, r_str, para_path, para_path_length, p_str, q_str, alpha_str, bias_weight_str))
        format_file = os.path.join(format_data_path, 'F_%s_%s_l%s_r%s_ph%d_pl%d_p%s_q%s_a%s_bw%s.txt' % (
            model_type, basename, lambda_str, r_str, para_path, para_path_length, p_str, q_str, alpha_str, bias_weight_str))
        if not os.path.isfile('./get_SVM_format_swe'):
            command = 'gcc get_SVM_format_swe.c -o get_SVM_format_swe -lm -pthread -O3 -march=native -Wall -funroll-loops -Wno-unused-result'
            print(command)
            os.system(command)
        if not os.path.isfile(format_file):
            command = './get_SVM_format_swe -input %s -word-vec %s -user-vec %s -output %s' % (
                input_file, word_vec_file, user_vec_file, format_file)
            print(command)
            os.system(command)

    return format_file


def tune_para_SVM(model_type, yelp_round, para_lambda, para_r, para_path, para_path_length, para_p, para_q, para_alpha, para_bias_weight):
    if model_type not in model_types:
        print('Error: not support %s' % (model_type))
        return ''
    lambda_str = str(para_lambda)
    lambda_index = lambda_str.index('.')
    lambda_str = lambda_str[0:lambda_index] + \
        'p' + lambda_str[lambda_index + 1:]
    r_str = str(para_r)
    r_index = r_str.index('.')
    r_str = r_str[0:r_index] + 'p' + r_str[r_index + 1:]
    p_str = str(para_p)
    p_index = p_str.index('.')
    p_str = p_str[0:p_index] + 'p' + p_str[p_index + 1:]
    q_str = str(para_q)
    q_index = q_str.index('.')
    q_str = q_str[0:q_index] + 'p' + q_str[q_index + 1:]
    alpha_str = str(para_alpha)
    alpha_index = alpha_str.index('.')
    alpha_str = alpha_str[0:alpha_index] + 'p' + alpha_str[alpha_index + 1:]
    bias_weight_str = str(para_bias_weight)
    bias_weight_index = bias_weight_str.index('.')
    bias_weight_str = bias_weight_str[0:bias_weight_index] + \
        'p' + bias_weight_str[bias_weight_index + 1:]

    input_dev_file = os.path.join(
        data_path, 'SVM_dev_one_fifth_rd%d.txt' % (yelp_round))
    # basename = os.path.basename(input_dev_file).split('.')[0]
    basename = 'dev_rd%d' % (yelp_round)
    if model_type == 'w2v':
        c_file = os.path.join(model_path, 'C_%s_%s.txt' %
                              (model_type, basename))
    elif model_type == 'swe':
        c_file = os.path.join(model_path, 'C_%s_%s_l%s_r%s.txt' % (
            model_type, basename, lambda_str, r_str))
    elif model_type.startswith('swe_with_randomwalk') or model_type.startswith('swe_with_deepwalk'):
        c_file = os.path.join(model_path, 'C_%s_%s_l%s_r%s_ph%d_pl%d.txt' % (
            model_type, basename, lambda_str, r_str, para_path, para_path_length))
    elif model_type.startswith('swe_with_node2vec') or model_type.startswith('swe_with_2nd_randomwalk'):
        c_file = os.path.join(model_path, 'C_%s_%s_l%s_r%s_ph%d_pl%d_p%s_q%s.txt' % (
            model_type, basename, lambda_str, r_str, para_path, para_path_length, p_str, q_str))
    elif model_type.startswith('swe_with_bias_randomwalk'):
        c_file = os.path.join(model_path, 'C_%s_%s_l%s_r%s_ph%d_pl%d_p%s_q%s_a%s_bw%s.txt' % (
            model_type, basename, lambda_str, r_str, para_path, para_path_length, p_str, q_str, alpha_str, bias_weight_str))

    if not os.path.isfile(c_file):
        format_dev_file = SVM_format(model_type, input_dev_file, yelp_round, para_lambda,
                                     para_r, para_path, para_path_length, para_p, para_q, para_alpha, para_bias_weight)
        command = '../liblinear/train -n %d -C -v 5 -c %f %s | tee -a %s' % (
            num_cores, begin_c, format_dev_file, c_file)
        print(command)
        with open(c_file, 'w') as fo:
            fo.write(command + '\n')
        os.system(command)
        os.remove(format_dev_file)
    return c_file


def get_head_tail_review(yelp_round):
    train_file = os.path.join(
        data_path, 'SVM_train_one_fifth_rd%d.txt' % (yelp_round))
    basename = 'train_rd%d' % (yelp_round)
    user_file = os.path.join(data_path, 'user_file_rd%d.txt' % (yelp_round))
    head_file = os.path.join(data_path, 'head_user_%s.txt' % (basename))
    tail_file = os.path.join(data_path, 'tail_user_%s.txt' % (basename))
    if not (os.path.isfile(head_file) and os.path.isfile(tail_file)):
        user_df = pd.read_csv(user_file, header=None)
        user_df.columns = ['user_id']
        user_number = user_df.shape[0]
        print("total %d users" % (user_number))

        # count how many reviews of each user in training file, ignore unknown user
        review_count_dict = dict()

        def addKey(x, review_count_dict):
            review_count_dict[x] = 0
        user_df['user_id'].apply(addKey, args=(review_count_dict, ))
        review_number = 0
        with open(train_file, 'r') as f:
            line = f.readline()
            while line:
                user_id = line.split('\t', 1)[0]
                if user_id != 'unknown_user_id':
                    review_count_dict[user_id] += 1
                    review_number += 1
                line = f.readline()
        print("total %d reviews" % (review_number))
        user_df['review_count'] = user_df['user_id'].apply(
            review_count_dict.get)
        head_users = set()
        user_df_sorted = user_df.sort_values('review_count', ascending=False)
        head_user_cnt = 0
        review_cnt = 0
        review_number_half = review_number // 2
        for row in user_df_sorted.iterrows():
            review_cnt += row[1]['review_count']
            if (review_cnt < review_number_half):
                head_user_cnt += 1
                head_users.add(row[1]['user_id'])
            else:
                break

        print("%d head user" % (head_user_cnt))
        print("%d tail user" % (user_number - head_user_cnt))
        f_head = open(head_file, 'w')
        f_tail = open(tail_file, 'w')
        with open(train_file, 'r') as f_train:
            for line in f_train:
                user_id = line.split('\t', 1)[0]
                if user_id != 'unknown_user_id':
                    if user_id in head_users:
                        f_head.write(line)
                    else:
                        f_tail.write(line)

        f_head.close()
        f_tail.close()
    return head_file, tail_file


def run_SVM_frac(user_train_files, model_type, factor, yelp_round, para_lambda, para_r, para_path, para_path_length, para_p, para_q, para_alpha, para_bias_weight):
    lambda_str = str(para_lambda)
    lambda_index = lambda_str.index('.')
    lambda_str = lambda_str[0:lambda_index] + \
        'p' + lambda_str[lambda_index + 1:]
    r_str = str(para_r)
    r_index = r_str.index('.')
    r_str = r_str[0:r_index] + 'p' + r_str[r_index + 1:]
    p_str = str(para_p)
    p_index = p_str.index('.')
    p_str = p_str[0:p_index] + 'p' + p_str[p_index + 1:]
    q_str = str(para_q)
    q_index = q_str.index('.')
    q_str = q_str[0:q_index] + 'p' + q_str[q_index + 1:]
    alpha_str = str(para_alpha)
    alpha_index = alpha_str.index('.')
    alpha_str = alpha_str[0:alpha_index] + 'p' + alpha_str[alpha_index + 1:]
    bias_weight_str = str(para_bias_weight)
    bias_weight_index = bias_weight_str.index('.')
    bias_weight_str = bias_weight_str[0:bias_weight_index] + \
        'p' + bias_weight_str[bias_weight_index + 1:]
    factor_str = str(factor)
    factor_index = factor_str.index('.')
    factor_str = factor_str[0:factor_index] + \
        'p' + factor_str[factor_index + 1:]

    c_file = tune_para_SVM(model_type, yelp_round, para_lambda, para_r, para_path,
                           para_path_length, para_p, para_q, para_alpha, para_bias_weight)
    basename = os.path.basename(c_file).split('.')[0]
    model_file = os.path.join(model_path, '%s.model' % (basename))
    result_file = os.path.join(result_path, '%s.txt' % (basename))
    para_c = 0.0
    with open(c_file, 'r') as f:
        para_c = float(f.readlines()[-1].split()[3])

    test_file = os.path.join(
        data_path, 'SVM_test_one_fifth_rd%d.txt' % (yelp_round))
    format_test_file = SVM_format(model_type, test_file, yelp_round, para_lambda, para_r,
                                  para_path, para_path_length, para_p, para_q, para_alpha, para_bias_weight)
    # accuracy_array[i][r] means: user_type i, r-th accuracy
    accuracy_array = [[0] * run_times for i in range(len(user_train_files))]
    for i, input_train_file in enumerate(user_train_files):
        basename = os.path.basename(input_train_file).split('_', 1)[0]
        total = 0
        with open(input_train_file, 'r') as f_tail:
            for l, line in enumerate(f_tail):
                total += 1
        # print("%s_reviews %d" % (input_train_file.split('_', 1)[0], total))
        format_train_file = SVM_format(model_type, input_train_file, yelp_round, para_lambda,
                                       para_r, para_path, para_path_length, para_p, para_q, para_alpha, para_bias_weight)

        for r in range(run_times):
            if model_type == 'w2v':
                random_train_file = os.path.join(data_path, 'R_%s_%s_rd%d_f%s_t%d.txt' % (
                    model_type, basename, yelp_round, factor_str, r))
                model_file = os.path.join(model_path, '%s_%s_rd%d_f%s_t%d.model' % (
                    model_type, basename, yelp_round, factor_str, r))
                result_file = os.path.join(result_path, '%s_%s_rd%d_f%s_t%d.txt' % (
                    model_type, basename, yelp_round, factor_str, r))
            elif model_type == 'swe':
                random_train_file = os.path.join(data_path, 'R_%s_%s_rd%d_l%s_r%s_f%s_t%d.txt' % (
                    model_type, basename, yelp_round, lambda_str, r_str, factor_str, r))
                model_file = os.path.join(model_path, '%s_%s_rd%d_l%s_r%s_f%s_t%d.model' % (
                    model_type, basename, yelp_round, lambda_str, r_str, factor_str, r))
                result_file = os.path.join(result_path, '%s_%s_rd%d_l%s_r%s_f%s_t%d.txt' % (
                    model_type, basename, yelp_round, lambda_str, r_str, factor_str, r))
            elif model_type.startswith('swe_with_randomwalk') or model_type.startswith('swe_with_deepwalk'):
                random_train_file = os.path.join(data_path, 'R_%s_%s_rd%d_l%s_r%s_ph%d_pl%d_f%s_t%d.txt' % (
                    model_type, basename, yelp_round, lambda_str, r_str, para_path, para_path_length, factor_str, r))
                model_file = os.path.join(model_path, '%s_%s_rd%d_l%s_r%s_ph%d_pl%d_f%s_t%d.model' % (
                    model_type, basename, yelp_round, lambda_str, r_str, para_path, para_path_length, factor_str, r))
                result_file = os.path.join(result_path, '%s_%s_rd%d_l%s_r%s_ph%d_pl%d_f%s_t%d.txt' % (
                    model_type, basename, yelp_round, lambda_str, r_str, para_path, para_path_length, factor_str, r))
            elif model_type.startswith('swe_with_node2vec') or model_type.startswith('swe_with_2nd_randomwalk'):
                random_train_file = os.path.join(data_path, 'R_%s_%s_rd%d_l%s_r%s_ph%d_pl%d_p%s_q%s_f%s_t%d.txt' % (
                    model_type, basename, yelp_round, lambda_str, r_str, para_path, para_path_length, p_str, q_str, factor_str, r))
                model_file = os.path.join(model_path, '%s_%s_rd%d_l%s_r%s_ph%d_pl%d_p%s_q%s_f%s_t%d.model' % (
                    model_type, basename, yelp_round, lambda_str, r_str, para_path, para_path_length, p_str, q_str, factor_str, r))
                result_file = os.path.join(result_path, '%s_%s_rd%d_l%s_r%s_ph%d_pl%d_p%s_q%s_f%s_t%d.txt' % (
                    model_type, basename, yelp_round, lambda_str, r_str, para_path, para_path_length, p_str, q_str, factor_str, r))
            elif model_type.startswith('swe_with_bias_randomwalk'):
                random_train_file = os.path.join(data_path, 'R_%s_%s_rd%d_l%s_r%s_ph%d_pl%d_p%s_q%s_a%s_bw%s_f%s_t%d.txt' % (
                    model_type, basename, yelp_round, lambda_str, r_str, para_path, para_path_length, p_str, q_str, alpha_str, bias_weight_str, factor_str, r))
                model_file = os.path.join(model_path, '%s_%s_rd%d_l%s_r%s_ph%d_pl%d_p%s_q%s_a%s_bw%s_f%s_t%d.model' % (
                    model_type, basename, yelp_round, lambda_str, r_str, para_path, para_path_length, p_str, q_str, alpha_str, bias_weight_str, factor_str, r))
                result_file = os.path.join(result_path, '%s_%s_rd%d_l%s_r%s_ph%d_pl%d_p%s_q%s_a%s_bw%s_f%s_t%d.txt' % (
                    model_type, basename, yelp_round, lambda_str, r_str, para_path, para_path_length, p_str, q_str, alpha_str, bias_weight_str, factor_str, r))
            if r > 0 and factor == 1.0:
                full_model_file = '%s0.model' % (
                    os.path.splitext(model_file)[0][:-1])
                full_result_file = '%s0.txt' % (
                    os.path.splitext(result_file)[0][:-1])
                with open(model_file, 'w') as fw:
                    with open(full_model_file, 'r') as fr:
                        line = fr.readline()
                        while line:
                            fw.write('%s' % (line))
                            line = fr.readline()
                with open(result_file, 'w') as fw:
                    with open(full_result_file, 'r') as fr:
                        line = fr.readline()
                        while line:
                            fw.write('%s' % (line))
                            line = fr.readline()

            if not os.path.isfile(result_file):
                if not os.path.isfile(model_file):
                    if not os.path.isfile(random_train_file):
                        random_index = np.random.permutation(total)
                        random_index = set(random_index[0:int(total * factor)])
                        if not os.path.isfile(random_train_file):
                            with open(random_train_file, 'w') as fw:
                                with open(format_train_file, 'r') as fr:
                                    for l, line in enumerate(fr):
                                        if l in random_index:
                                            fw.write(line)

                    command = '../liblinear/train -n %d -c %f %s %s > /dev/null' % (
                        num_cores, para_c, random_train_file, model_file)
                    print(command)
                    os.system(command)
                    os.remove(random_train_file)
                command = '../liblinear/predict %s %s /dev/null | tee %s' % (
                    format_test_file, model_file, result_file)
                print(command)
                os.system(command)
            with open(result_file, 'r') as f:
                accuracy_array[i][r] = float(f.readline().split()[-2][:-1])
    # os.remove(format_test_file)
    return accuracy_array


def store_std_mean(accuracy_array, model_type, factor, yelp_round, para_lambda, para_r, para_path, para_path_length, para_p, para_q, para_alpha, para_bias_weight):
    lambda_str = str(para_lambda)
    lambda_index = lambda_str.index('.')
    lambda_str = lambda_str[0:lambda_index] + \
        'p' + lambda_str[lambda_index + 1:]
    r_str = str(para_r)
    r_index = r_str.index('.')
    r_str = r_str[0:r_index] + 'p' + r_str[r_index + 1:]
    p_str = str(para_p)
    p_index = p_str.index('.')
    p_str = p_str[0:p_index] + 'p' + p_str[p_index + 1:]
    q_str = str(para_q)
    q_index = q_str.index('.')
    q_str = q_str[0:q_index] + 'p' + q_str[q_index + 1:]
    alpha_str = str(para_alpha)
    alpha_index = alpha_str.index('.')
    alpha_str = alpha_str[0:alpha_index] + 'p' + alpha_str[alpha_index + 1:]
    bias_weight_str = str(para_bias_weight)
    bias_weight_index = bias_weight_str.index('.')
    bias_weight_str = bias_weight_str[0:bias_weight_index] + \
        'p' + bias_weight_str[bias_weight_index + 1:]
    factor_str = str(factor)
    factor_index = factor_str.index('.')
    factor_str = factor_str[0:factor_index] + \
        'p' + factor_str[factor_index + 1:]

    head_std = np.std(accuracy_array[0])
    head_mean = np.mean(accuracy_array[0])
    tail_std = np.std(accuracy_array[1])
    tail_mean = np.mean(accuracy_array[1])
    # print('model_type: %s\tyelp_round: %d\tlambda: %f\tr: %f\talpha: %f\n' % (model_type, yelp_round, para_lambda, para_r, para_alpha))
    # print('\thead_std: %f' % head_std)
    # print('\thead_mean: %f' % head_mean)
    # print('\ttail_std: %f' % tail_std)
    # print('\ttail_mean: %f' % tail_mean)
    if model_type == 'w2v':
        std_file = os.path.join(result_path, '%s_head_tail_std_rd%d_f%s.txt' % (
            model_type, yelp_round, factor_str))
        mean_file = os.path.join(result_path, '%s_head_tail_mean_rd%d_f%s.txt' % (
            model_type, yelp_round, factor_str))
    elif model_type == 'swe':
        std_file = os.path.join(result_path, '%s_head_tail_std_rd%d_l%s_r%s_f%s.txt' % (
            model_type, yelp_round, lambda_str, r_str, factor_str))
        mean_file = os.path.join(result_path, '%s_head_tail_mean_rd%d_l%s_r%s_f%s.txt' % (
            model_type, yelp_round, lambda_str, r_str, factor_str))
    elif model_type.startswith('swe_with_randomwalk') or model_type.startswith('swe_with_deepwalk'):
        std_file = os.path.join(result_path, '%s_head_tail_std_rd%d_l%s_r%s_ph%d_pl%d_f%s.txt' % (
            model_type, yelp_round, lambda_str, r_str, para_path, para_path_length, factor_str))
        mean_file = os.path.join(result_path, '%s_head_tail_mean_rd%d_l%s_r%s_ph%d_pl%d_f%s.txt' % (
            model_type, yelp_round, lambda_str, r_str, para_path, para_path_length, factor_str))
    elif model_type.startswith('swe_with_node2vec') or model_type.startswith('swe_with_2nd_randomwalk'):
        std_file = os.path.join(result_path, '%s_head_tail_std_rd%d_l%s_r%s_ph%d_pl%d_p%s_q%s_f%s.txt' % (
            model_type, yelp_round, lambda_str, r_str, para_path, para_path_length, p_str, q_str, factor_str))
        mean_file = os.path.join(result_path, '%s_head_tail_mean_rd%d_l%s_r%s_ph%d_pl%d_p%s_q%s_f%s.txt' % (
            model_type, yelp_round, lambda_str, r_str, para_path, para_path_length, p_str, q_str, factor_str))
    elif model_type.startswith('swe_with_bias_randomwalk'):
        std_file = os.path.join(result_path, '%s_head_tail_std_rd%d_l%s_r%s_ph%d_pl%d_p%s_q%s_a%s_bw%s_f%s.txt' % (
            model_type, yelp_round, lambda_str, r_str, para_path, para_path_length, p_str, q_str, alpha_str, bias_weight_str, factor_str))
        mean_file = os.path.join(result_path, '%s_head_tail_mean_rd%d_l%s_r%s_ph%d_pl%d_p%s_q%s_a%s_bw%s_f%s.txt' % (
            model_type, yelp_round, lambda_str, r_str, para_path, para_path_length, p_str, q_str, alpha_str, bias_weight_str, factor_str))
    with open(std_file, 'w') as f:
        f.write('head_std: %f\n' % head_std)
        f.write('tail_std: %f\n' % tail_std)
    with open(mean_file, 'w') as f:
        f.write('head_mean: %f\n' % head_mean)
        f.write('tail_mean: %f\n' % tail_mean)


def print_results(yelp_round, para_lambda, para_r, para_path, para_path_length, para_p, para_q, para_alpha, para_bias_weight):
    lambda_str = str(para_lambda)
    lambda_index = lambda_str.index('.')
    lambda_str = lambda_str[0:lambda_index] + \
        'p' + lambda_str[lambda_index + 1:]
    r_str = str(para_r)
    r_index = r_str.index('.')
    r_str = r_str[0:r_index] + 'p' + r_str[r_index + 1:]
    p_str = str(para_p)
    p_index = p_str.index('.')
    p_str = p_str[0:p_index] + 'p' + p_str[p_index + 1:]
    q_str = str(para_q)
    q_index = q_str.index('.')
    q_str = q_str[0:q_index] + 'p' + q_str[q_index + 1:]
    alpha_str = str(para_alpha)
    alpha_index = alpha_str.index('.')
    alpha_str = alpha_str[0:alpha_index] + 'p' + alpha_str[alpha_index + 1:]
    bias_weight_str = str(para_bias_weight)
    bias_weight_index = bias_weight_str.index('.')
    bias_weight_str = bias_weight_str[0:bias_weight_index] + \
        'p' + bias_weight_str[bias_weight_index + 1:]

    table_file = os.path.join(result_path, 'head_tail_result.txt')
    col, row = len(factor_array), len(model_types) * 4
    # head-std
    # head-mean
    # tail-std
    # tail-mean
    result_matrix = [['' for x in range(col)] for y in range(row)]
    head_tail_column = []
    for i, model_type in enumerate(model_types):
        head_tail_column.append('%s-std' % model_type)
        head_tail_column.append('%s-mean' % model_type)
        if model_type == 'w2v':
            std_basename = '%s_head_tail_std_rd%d' % (model_type, yelp_round)
            mean_basename = '%s_head_tail_mean_rd%d' % (model_type, yelp_round)
        elif model_type == 'swe':
            std_basename = '%s_head_tail_std_rd%d_l%s_r%s' % (
                model_type, yelp_round, lambda_str, r_str)
            mean_basename = '%s_head_tail_mean_rd%d_l%s_r%s' % (
                model_type, yelp_round, lambda_str, r_str)
        elif model_type.startswith('swe_with_randomwalk') or model_type.startswith('swe_with_deepwalk'):
            std_basename = '%s_head_tail_std_rd%d_l%s_r%s_ph%d_pl%d' % (
                model_type, yelp_round, lambda_str, r_str, para_path, para_path_length)
            mean_basename = '%s_head_tail_mean_rd%d_l%s_r%s_ph%d_pl%d' % (
                model_type, yelp_round, lambda_str, r_str, para_path, para_path_length)
        elif model_type.startswith('swe_with_node2vec') or model_type.startswith('swe_with_2nd_randomwalk'):
            std_basename = '%s_head_tail_std_rd%d_l%s_r%s_ph%d_pl%d_p%s_q%s' % (
                model_type, yelp_round, lambda_str, r_str, para_path, para_path_length, p_str, q_str)
            mean_basename = '%s_head_tail_mean_rd%d_l%s_r%s_ph%d_pl%d_p%s_q%s' % (
                model_type, yelp_round, lambda_str, r_str, para_path, para_path_length, p_str, q_str)
        elif model_type.startswith('swe_with_bias_randomwalk'):
            std_basename = '%s_head_tail_std_rd%d_l%s_r%s_ph%d_pl%d_p%s_q%s_a%s_bw%s' % (
                model_type, yelp_round, lambda_str, r_str, para_path, para_path_length, p_str, q_str, alpha_str, bias_weight_str)
            mean_basename = '%s_head_tail_mean_rd%d_l%s_r%s_ph%d_pl%d_p%s_q%s_a%s_bw%s' % (
                model_type, yelp_round, lambda_str, r_str, para_path, para_path_length, p_str, q_str, alpha_str, bias_weight_str)
        for j, factor in enumerate(factor_array):
            factor_str = str(factor)
            factor_index = factor_str.index('.')
            factor_str = factor_str[0:factor_index] + \
                'p' + factor_str[factor_index + 1:]
            std_file = os.path.join(result_path, '%s_f%s.txt' %
                                    (std_basename, factor_str))
            mean_file = os.path.join(
                result_path, '%s_f%s.txt' % (mean_basename, factor_str))
            with open(std_file, 'r') as f:
                lines = f.readlines()
                result_matrix[i * 4 + 0][j] = lines[0].split()[1]
                result_matrix[i * 4 + 2][j] = lines[1].split()[1]
            with open(mean_file, 'r') as f:
                lines = f.readlines()
                result_matrix[i * 4 + 1][j] = lines[0].split()[1]
                result_matrix[i * 4 + 3][j] = lines[1].split()[1]
    table = PrettyTable()
    table.add_column("Head", head_tail_column)
    for j, f in enumerate(factor_array):
        field = '%d%%' % (f * 100)
        head_column = []
        for i in range(len(model_types)):
            head_column.append(result_matrix[4 * i + 0][j])
            head_column.append(result_matrix[4 * i + 1][j])
        table.add_column(field, head_column)
    print('yelp_round: %d\tlambda: %f\tr: %f\tpath: %d\tpath_length: %d\tp: %f\tq: %f\talpha: %f\tbias_weight: %f\n' % (
        yelp_round, para_lambda, para_r, para_path, para_path_length, para_p, para_q, para_alpha, para_bias_weight))
    print(table)
    table_txt = table.get_string()
    with open(table_file, 'a+') as f:
        f.write('yelp_round: %d\tlambda: %f\tr: %f\tpath: %d\tpath_length: %d\tp: %f\tq: %f\talpha: %f\tbias_weight: %f\n' % (
            yelp_round, para_lambda, para_r, para_path, para_path_length, para_p, para_q, para_alpha, para_bias_weight))
        f.write(table.get_string())
        f.write('\n\n\n')

    table = PrettyTable()
    table.add_column("tail", head_tail_column)
    for j, f in enumerate(factor_array):
        field = '%d%%' % (f * 100)
        tail_column = []
        for i in range(len(model_types)):
            tail_column.append(result_matrix[4 * i + 2][j])
            tail_column.append(result_matrix[4 * i + 3][j])
        table.add_column(field, tail_column)
    print('yelp_round: %d\tlambda: %f\tr: %f\tpath: %d\tpath_length: %d\tp: %f\tq: %f\talpha: %f\tbias_weight: %f\n' % (
        yelp_round, para_lambda, para_r, para_path, para_path_length, para_p, para_q, para_alpha, para_bias_weight))
    print(table)
    table_txt = table.get_string()
    with open(table_file, 'a+') as f:
        f.write('yelp_round: %d\tlambda: %f\tr: %f\tpath: %d\tpath_length: %d\tp: %f\tq: %f\talpha: %f\tbias_weight: %f\n' % (
            yelp_round, para_lambda, para_r, para_path, para_path_length, para_p, para_q, para_alpha, para_bias_weight))
        f.write(table.get_string())
        f.write('\n\n\n')


#-------------------------------------------------------------
if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Sentiment Classification on head and tail users using liblinear')

    parser.add_argument('--yelp_round', default=9, type=int,
                        choices={9, 10}, help="The round number of yelp data")
    parser.add_argument('--para_lambda', default=1.0, type=float,
                        help='The trade off parameter between log-likelihood and regularization term')
    parser.add_argument('--para_r', default=1.0, type=float,
                        help="The constraint of the L2-norm")
    parser.add_argument('--para_path', default=100, type=int,
                        help="The number of random walk paths for every review")
    parser.add_argument('--para_path_length', default=10, type=int,
                        help="The length of random walk paths for every review")
    parser.add_argument('--para_p', default=1.0, type=float,
                        help="The return parameter for the second-order random walk")
    parser.add_argument('--para_q', default=1.0, type=float,
                        help="The in-out parameter for the second-order random walk")
    parser.add_argument('--para_alpha', default=0.02, type=float,
                        help="The restart parameter for the bias random walk")
    parser.add_argument('--para_bias_weight', default=0.2,
                        type=float, help="The bias parameter for the bias random walk")

    args = parser.parse_args()

    parser.print_help()

    user_train_files = get_head_tail_review(args.yelp_round)

    for model_type in model_types:
        for frac in factor_array:
            accuracy_array = run_SVM_frac(user_train_files, model_type, frac, args.yelp_round, args.para_lambda, args.para_r,
                                          args.para_path, args.para_path_length, args.para_p, args.para_q, args.para_alpha, args.para_bias_weight)
            store_std_mean(accuracy_array, model_type, frac, args.yelp_round, args.para_lambda, args.para_r,
                           args.para_path, args.para_path_length, args.para_p, args.para_q, args.para_alpha, args.para_bias_weight)

    print_results(args.yelp_round, args.para_lambda, args.para_r, args.para_path,
                  args.para_path_length, args.para_p, args.para_q, args.para_alpha, args.para_bias_weight)
