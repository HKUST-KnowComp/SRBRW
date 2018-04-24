import os
import math

para_lambdas = [8.0]
para_rs = [0.25]
para_paths = [10]
para_path_lengths = [60]
para_ps = [0.5]
para_qs = [1]
para_alphas = [0.12]
para_bias_weights = [1.5]

embs_path = '../embs'

if not os.path.isdir(embs_path):
    os.mkdir(embs_path)

for para_lambda in para_lambdas:
    for para_r in para_rs:
        for para_path in para_paths:
            for para_path_length in para_path_lengths:
                for para_p in para_ps:
                    for para_q in para_qs:
                        for para_alpha in para_alphas:
                            for para_bias_weight in para_bias_weights:
                                command = 'python training.py --para_lambda %f --para_r %f --para_path %d --para_path_length %d --para_p %f --para_q %f --para_alpha %f --para_bias_weight %f --yelp_round 10' % (
                                    para_lambda, para_r, para_path, para_path_length, para_p, para_q, para_alpha, para_bias_weight)
                                print(command)
                                os.system(command)
