import sys
import os
import argparse
import numpy as np
from prettytable import PrettyTable
from multiprocessing import cpu_count

num_cores = cpu_count()

# model_types = ['w2v', 'swe', 'swe_with_randomwalk', 'swe_with_2nd_randomwalk', 'swe_with_bias_randomwalk', 'swe_with_deepwalk', 'swe_with_node2vec']
model_types = ['swe_with_bias_randomwalk']

data_path = '../data'
embs_path = '../embs'
model_path = 'models'
result_path = 'results'
format_data_path = 'format_data'
begin_c = 0.015625


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


def test(model_types, yelp_round, para_lambda, para_r, para_path, para_path_length, para_p, para_q, para_alpha, para_bias_weight):
    if model_type not in model_types:
        print('Error: not support %s' % (model_type))
        return ''

    input_train_file = os.path.join(
        data_path, 'SVM_train_one_fifth_rd%d.txt' % (yelp_round))
    input_test_file = os.path.join(
        data_path, 'SVM_test_one_fifth_rd%d.txt' % (yelp_round))
    c_file = tune_para_SVM(model_type, yelp_round, para_lambda, para_r, para_path,
                           para_path_length, para_p, para_q, para_alpha, para_bias_weight)
    basename = os.path.basename(c_file).split('.')[0]
    model_file = os.path.join(model_path, '%s.model' % (basename))
    result_file = os.path.join(result_path, 'result_%s.txt' % (basename))
    para_c = 0.0
    with open(c_file, 'r') as f:
        para_c = float(f.readlines()[-1].split()[3])

    if not os.path.isfile(result_file):
        if not os.path.isfile(model_file):
            format_train_file = SVM_format(model_type, input_train_file, yelp_round, para_lambda,
                                           para_r, para_path, para_path_length, para_p, para_q, para_alpha, para_bias_weight)
            command = '../liblinear/train -n %d -c %f %s %s > /dev/null' % (
                num_cores, para_c, format_train_file, model_file)
            print(command)
            os.system(command)
            os.remove(format_train_file)
        format_test_file = SVM_format(model_type, input_test_file, yelp_round, para_lambda,
                                      para_r, para_path, para_path_length, para_p, para_q, para_alpha, para_bias_weight)
        command = '../liblinear/predict %s %s /dev/null | tee %s' % (
            format_test_file, model_file, result_file)
        print(command)
        os.system(command)
        os.remove(format_test_file)
    accuracy = 0.0
    with open(result_file, 'r') as f:
        accuracy = f.readline().split()[2]
    return accuracy


def print_table(yelp_round, para_lambda, para_r, para_path, para_path_length, para_p, para_q, para_alpha, para_bias_weight, best_c_array, dev_accuracy_array, test_accuracy_array):
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

    table_file = os.path.join(result_path, 'sentiment_result.txt')
    table = PrettyTable()
    table.add_column('model_types', model_types)
    table.add_column('best_c', best_c_array)
    table.add_column('dev_accuracy', dev_accuracy_array)
    table.add_column('test_accuracy', test_accuracy_array)
    print('yelp_round: %d\tlambda: %f\tr: %f\tpath: %d\tpath_length: %d\tp: %f\tq: %f\talpha: %f\tbias_weight: %f\n' % (
        yelp_round, para_lambda, para_r, para_path, para_path_length, para_p, para_q, para_alpha, para_bias_weight))
    print(table)

    with open(table_file, 'a+') as f:
        f.write('yelp_round: %d\tlambda: %f\tr: %f\tpath: %d\tpath_length: %d\tp: %f\tq: %f\talpha: %f\tbias_weight: %f\n' % (
            yelp_round, para_lambda, para_r, para_path, para_path_length, para_p, para_q, para_alpha, para_bias_weight))
        f.write(table.get_string())
        f.write('\n\n\n')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Sentiment Classification on users using liblinear')

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

    test_accuracy_array = []
    dev_accuracy_array = []
    best_c_array = []
    for model_type in model_types:
        c_file = tune_para_SVM(model_type, args.yelp_round, args.para_lambda, args.para_r, args.para_path,
                               args.para_path_length, args.para_p, args.para_q, args.para_alpha, args.para_bias_weight)
        with open(c_file, 'r') as f:
            l = f.readlines()[-1].split()
            best_c_array.append(float(l[3]))
            dev_accuracy_array.append(l[-1])
        test_accuracy_array.append(test(model_type, args.yelp_round, args.para_lambda, args.para_r, args.para_path,
                                        args.para_path_length, args.para_p, args.para_q, args.para_alpha, args.para_bias_weight))
    print_table(args.yelp_round, args.para_lambda, args.para_r, args.para_path, args.para_path_length, args.para_p,
                args.para_q, args.para_alpha, args.para_bias_weight, best_c_array, dev_accuracy_array, test_accuracy_array)
