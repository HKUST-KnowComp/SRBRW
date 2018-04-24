import sys
import os
import argparse
from multiprocessing import cpu_count
num_cores = cpu_count()

# model_types = ['w2v', 'swe', 'swe_with_randomwalk', 'swe_with_2nd_randomwalk', 'swe_with_bias_randomwalk', 'swe_with_deepwalk', 'swe_with_node2vec']
model_types = ['swe_with_bias_randomwalk']

input_path = '../data'
output_path = '../embs'


def train(model_type, yelp_round, para_lambda, para_r, para_path, para_path_length, para_p, para_q, para_alpha, para_bias_weight):
    if model_type not in model_types:
        print('Error: not support %s' % (model_type))
        return
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
        if not os.path.isfile('./w2v'):
            command = 'gcc w2v.c -o w2v -lm -pthread -O3 -march=native -Wall -funroll-loops -Wno-unused-result'
            print(command)
            os.system(command)
        train_file = os.path.join(
            input_path, 'w2v_train_rd%d.txt' % (yelp_round))
        word_vec_file = os.path.join(
            input_path, '%s_word_vec_rd%d.txt' % (model_type, yelp_round))
        context_vec_file = os.path.join(
            input_path, '%s_context_vec_rd%d.txt' % (model_type, yelp_round))

        if os.path.isfile(word_vec_file) and os.path.isfile(context_vec_file):
            print('%s and %s already exist. If you want to re-train word to vector, please delete them first.' %
                  (word_vec_file, context_vec_file))
            return

        print('Train Word to Vector:\n')
        command = './w2v -train %s -output %s -save-context %s -size 100 -window 5 -cbow 1 -hs 1 -negative 5 -threads %d -iter 5 -sample 1e-4' % (
            train_file, word_vec_file, context_vec_file, num_cores)
        print(command)
        os.system(command)
    elif model_type == 'swe':
        if not os.path.isfile('./%s' % (model_type)):
            command = 'gcc %s.c -o %s -lm -pthread -O3 -march=native -Wall -funroll-loops -Wno-unused-result' % (
                model_type, model_type)
            print(command)
            os.system(command)
        train_file = os.path.join(
            input_path, 'swe_train_rd%d.txt' % (yelp_round))
        user_file = os.path.join(
            input_path, 'user_file_rd%d.txt' % (yelp_round))
        user_graph_file = os.path.join(
            input_path, 'user_graph_rd%d.txt' % (yelp_round))
        word_vec_file = os.path.join(output_path, '%s_word_vec_rd%d_l%s_r%s.txt' % (
            model_type, yelp_round, lambda_str, r_str))
        user_vec_file = os.path.join(output_path, '%s_user_vec_rd%d_l%s_r%s.txt' % (
            model_type, yelp_round, lambda_str, r_str))
        context_vec_file = os.path.join(output_path, '%s_context_vec_rd%d_l%s_r%s.txt' % (
            model_type, yelp_round, lambda_str, r_str))

        if os.path.isfile(word_vec_file) and os.path.isfile(context_vec_file):
            print('%s ,%s and %s already exist. If you want to re-train word to vector, please delete them first.' %
                  (word_vec_file, context_vec_file, user_vec_file))
            return

        print('Train Socialized Word Embeddings:\n')
        command = './%s -train %s -user %s -user-graph %s -output %s -save-user %s -save-context %s -size 100 -window 5 -cbow 1 -hs 1 -negative 5 -lambda %.5f -r %.5f -threads %d -iter 5 -sample 1e-4' % (
            model_type, train_file, user_file, user_graph_file, word_vec_file, user_vec_file, context_vec_file, para_lambda, para_r, num_cores)
        print(command)
        os.system(command)
    elif model_type.startswith('swe_with_randomwalk') or model_type.startswith('swe_with_deepwalk'):
        if not os.path.isfile('./%s' % (model_type)):
            command = 'gcc %s.c -o %s -lm -pthread -O3 -march=native -Wall -funroll-loops -Wno-unused-result' % (
                model_type, model_type)
            print(command)
            os.system(command)
        train_file = os.path.join(
            input_path, 'swe_train_rd%d.txt' % (yelp_round))
        user_file = os.path.join(
            input_path, 'user_file_rd%d.txt' % (yelp_round))
        user_graph_file = os.path.join(
            input_path, 'user_graph_rd%d.txt' % (yelp_round))
        word_vec_file = os.path.join(output_path, '%s_word_vec_rd%d_l%s_r%s_ph%d_pl%d.txt' % (
            model_type, yelp_round, lambda_str, r_str, para_path, para_path_length))
        user_vec_file = os.path.join(output_path, '%s_user_vec_rd%d_l%s_r%s_ph%d_pl%d.txt' % (
            model_type, yelp_round, lambda_str, r_str, para_path, para_path_length))
        context_vec_file = os.path.join(output_path, '%s_context_vec_rd%d_l%s_r%s_ph%d_pl%d.txt' % (
            model_type, yelp_round, lambda_str, r_str, para_path, para_path_length))

        if os.path.isfile(word_vec_file) and os.path.isfile(context_vec_file):
            print('%s ,%s and %s already exist. If you want to re-train word to vector, please delete them first.' %
                  (word_vec_file, context_vec_file, user_vec_file))
            return

        print('Train Socialized Word Embeddings:\n')
        command = './%s -train %s -user %s -user-graph %s -output %s -save-user %s -save-context %s -size 100 -window 5 -cbow 1 -hs 1 -negative 5 -lambda %.5f -r %.5f -paths %d -path-length %d -threads %d -iter 5 -sample 1e-4' % (
            model_type, train_file, user_file, user_graph_file, word_vec_file, user_vec_file, context_vec_file, para_lambda, para_r, para_path, para_path_length, num_cores)
        print(command)
        os.system(command)
    elif model_type.startswith('swe_with_node2vec') or model_type.startswith('swe_with_2nd_randomwalk'):
        if not os.path.isfile('./%s' % (model_type)):
            command = 'gcc %s.c -o %s -lm -pthread -O3 -march=native -Wall -funroll-loops -Wno-unused-result' % (
                model_type, model_type)
            print(command)
            os.system(command)
        train_file = os.path.join(
            input_path, 'swe_train_rd%d.txt' % (yelp_round))
        user_file = os.path.join(
            input_path, 'user_file_rd%d.txt' % (yelp_round))
        user_graph_file = os.path.join(
            input_path, 'user_graph_rd%d.txt' % (yelp_round))
        word_vec_file = os.path.join(output_path, '%s_word_vec_rd%d_l%s_r%s_ph%d_pl%d_p%s_q%s.txt' % (
            model_type, yelp_round, lambda_str, r_str, para_path, para_path_length, p_str, q_str))
        user_vec_file = os.path.join(output_path, '%s_user_vec_rd%d_l%s_r%s_ph%d_pl%d_p%s_q%s.txt' % (
            model_type, yelp_round, lambda_str, r_str, para_path, para_path_length, p_str, q_str))
        context_vec_file = os.path.join(output_path, '%s_context_vec_rd%d_l%s_r%s_ph%d_pl%d_p%s_q%s.txt' % (
            model_type, yelp_round, lambda_str, r_str, para_path, para_path_length, p_str, q_str))

        if os.path.isfile(word_vec_file) and os.path.isfile(context_vec_file):
            print('%s ,%s and %s already exist. If you want to re-train word to vector, please delete them first.' %
                  (word_vec_file, context_vec_file, user_vec_file))
            return

        print('Train Socialized Word Embeddings:\n')
        command = './%s -train %s -user %s -user-graph %s -output %s -save-user %s -save-context %s -size 100 -window 5 -cbow 1 -hs 1 -negative 5 -lambda %.5f -r %.5f -paths %d -path-length %d -p %.5f -q %.5f -threads %d -iter 5 -sample 1e-4' % (
            model_type, train_file, user_file, user_graph_file, word_vec_file, user_vec_file, context_vec_file, para_lambda, para_r, para_path, para_path_length, para_p, para_q, num_cores)
        print(command)
        os.system(command)
    elif model_type.startswith('swe_with_bias_randomwalk'):
        if not os.path.isfile('./%s' % (model_type)):
            command = 'gcc %s.c -o %s -lm -pthread -O3 -march=native -Wall -funroll-loops -Wno-unused-result' % (
                model_type, model_type)
            print(command)
            os.system(command)
        train_file = os.path.join(
            input_path, 'swe_train_rd%d.txt' % (yelp_round))
        user_file = os.path.join(
            input_path, 'user_file_rd%d.txt' % (yelp_round))
        user_graph_file = os.path.join(
            input_path, 'user_graph_rd%d.txt' % (yelp_round))
        word_vec_file = os.path.join(output_path, '%s_word_vec_rd%d_l%s_r%s_ph%d_pl%d_p%s_q%s_a%s_bw%s.txt' % (
            model_type, yelp_round, lambda_str, r_str, para_path, para_path_length, p_str, q_str, alpha_str, bias_weight_str))
        user_vec_file = os.path.join(output_path, '%s_user_vec_rd%d_l%s_r%s_ph%d_pl%d_p%s_q%s_a%s_bw%s.txt' % (
            model_type, yelp_round, lambda_str, r_str, para_path, para_path_length, p_str, q_str, alpha_str, bias_weight_str))
        context_vec_file = os.path.join(output_path, '%s_context_vec_rd%d_l%s_r%s_ph%d_pl%d_p%s_q%s_a%s_bw%s.txt' % (
            model_type, yelp_round, lambda_str, r_str, para_path, para_path_length, p_str, q_str, alpha_str, bias_weight_str))

        if os.path.isfile(word_vec_file) and os.path.isfile(context_vec_file):
            print('%s ,%s and %s already exist. If you want to re-train word to vector, please delete them first.' %
                  (word_vec_file, context_vec_file, user_vec_file))
            return

        print('Train Socialized Word Embeddings:\n')
        command = './%s -train %s -user %s -user-graph %s -output %s -save-user %s -save-context %s -size 100 -window 5 -cbow 1 -hs 1 -negative 5 -lambda %.5f -r %.5f -paths %d -path-length %d -p %.5f -q %.5f -restart-alpha %.5f -bias-weight %.5f -threads %d -iter 5 -sample 1e-4' % (
            model_type, train_file, user_file, user_graph_file, word_vec_file, user_vec_file, context_vec_file, para_lambda, para_r, para_path, para_path_length, para_p, para_q, para_alpha, para_bias_weight, num_cores)
        print(command)
        os.system(command)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Train Socialized Word Embeddings and Word to Vector')
        
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

    for model_type in model_types:
        train(model_type, args.yelp_round, args.para_lambda, args.para_r, args.para_path,
              args.para_path_length, args.para_p, args.para_q, args.para_alpha, args.para_bias_weight)
