# Biased Random Walk based Social Regularization for Word Embeddings

[Socialized word embeddings](https://github.com/HKUST-KnowComp/SocializedWordEmbeddings) (SWE) has been proposed to deal with two phenomena of language use: 

* everyone has his/her own personal characteristics of language use,
* and socially connected users are likely to use language in similar ways.

We observe that the spread of language use is **transitive**, namely, one user can affect his/her friends, and the friends can also affect their friends.  However, SWE modeled transitivity implicitly. The social regularization in SWE only applies to one-hop friends, and thus users outside the one-hop social circle will not be affected directly. 

In this work, we adopt **random walk methods** to generate paths on the social graph to model the transitivity explicitly.  Each user on a path will be affected by his/her adjacent user(s) on the path. Moreover, according to the update mechanism of SWE, fewer friends a user has, fewer update opportunities he/she can get.  Hence, we propose a **biased random walk method** to provide these users with more update opportunities.  

**[Experiments](EXPERIMENTS.md) show that our random walk based social regularizations perform better on sentiment classification task**. 

### Preparation

You need to download the dataset:
* Download [Yelp dataset](https://www.yelp.com/dataset_challenge/dataset)

* Convert following datasets from json format to csv format by using `json_to_csv_converter.py` in `preprocess` and get two `.json` files:

  * `yelp_academic_dataset_review.json`
  * `yelp_academic_dataset_user.json`

  And put them in the `data` directory.

* Download [LIBLINEAR](https://github.com/cjlin1/liblinear) or [Multi-core LIBLINEAR](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/multicore-liblinear) and put the `liblinear` under the root directory of the repo. And you can install LIBLINEAR according to its *Installation*

### Preprocessing
```bash
cd preprocess
```


Modify `run.py` by specifying `--input` (Path to yelp dataset).

```bash
python run.py
```

### Training
```bash
cd train
```

You may modify the following arguments in `run.py`:

* `--yelp_round`                           The round number of yelp data, e.g. {9, 10}
* `--para_lambda`                         The trade off parameter between log-likelihood and regularization term
* `--para_r`                                   The constraint of the L2-norm
* `--para_path`                             The number of random walk paths for every review
* `--para_path_length`               The length of random walk paths for every review
* `--path_p`                                   The return parameter for the second-order random walk
* `--path_q`                                   The in-out parameter for the second-order random walk
* `--para_alpha`                           The restart parameter for the bias random walk
* `--para_bias_weight`               The bias parameter for the bias random walk

Because there are a number of ways to train models, you also need to specify `model_types` in `training.py`.

Then begin to train models:

```bash
python run.py
```


### Sentiment Classification
```bash
cd sentiment
```

You may modify the following arguments in `run.py`, arguments are the same as above. 

It will run two `.py` files, where `sentiment.py` is about sentiment classification on all users while `head_tail.py` is about sentiment classification on head users and tail users. You can choose one or both.

Because there are a number of ways to train models, you also need to specify `model_types` in `sentiment.py` and `head_tail.py`.

Then begin the classification:

```bash
python run.py
```

### User vectors for attention

We thank Tao Lei as our code is developed based on [his code](https://github.com/taolei87/rcnn/tree/master/code).

```bash
cd attention
```

You can simply re-implement our results of different settings by modifying the `run.sh`: 

1. add user and word embeddings by specifying `--user_embs` and `--embedding`.
2. add train/dev/test files by specifying `--train`, `--dev`, and `--test` respectively.
3. choose the type of layers by specifying `--layer`, *cnn* or *lstm*.
4. three settings for our experiments could be achieved by specifying `--user_atten` and `--user_atten_base`:
    * setting `--user_atten 0` for *Without attention*.
    * setting `--user_atten 1 --user_atten_base 1` for *Trained attention*.
    * setting `--user_atten 1 --user_atten_base 0` for *Fixed user vector as attention*.

Then begin to train attention models:

```bash
bash run.sh
```

### Pretrained embeddings and models

You can download pretrained embeddings and models in [release](https://github.com/HKUST-KnowComp/SRBRW/releases). 
* Put `embs/*` in the `embs/` directory.
* Put `sentiment_models/*` in the `sentiment/models/` directory.
* Put `attention_models/*` in the `attention/models/` directory.

### Directory Structure

```
.
├── attention
│   ├── models
│   ├── nn
│   │   ├── __init__.py
│   │   ├── advanced.py
│   │   ├── basic.py
│   │   ├── evaluation.py
│   │   ├── initialization.py
│   │   └── optimization.py
│   ├── utils
│   │   └── __init__.py
│   ├── dc.py
│   └── run.sh
├── data
├── embs
│   ├── users_sample.txt
│   └── words_sample.txt
├── liblinear
├── preprocess
│   ├── english.pickle
│   ├── english_stop.txt
│   ├── json_to_csv_converter.py
│   ├── preprocess.py
│   └── run.py
├── sentiment
│   ├── format_data
│   ├── models
│   ├── results
│   ├── get_SVM_format_swe.c
│   ├── get_SVM_format_w2v.c
│   ├── head_tail.py
│   ├── run.py
│   └── sentiment.py
├── train
│   ├── run.py
│   ├── swe.c
│   ├── swe_with_2nd_randomwalk.c
│   ├── swe_with_bias_randomwalk.c
│   ├── swe_with_deepwalk.c
│   ├── swe_with_node2vec.c
│   ├── swe_with_randomwalk.c
│   ├── training.py
│   └── w2v.c
├── LICENSE
└── README.md
```

### OS and Dependencies

* *nix operating systems or [WSL](https://docs.microsoft.com/en-us/windows/wsl/about)
* Python 2.7 
* gcc
* Theano (>= 0.7 but < 1.0)
* NumPy
* gensim
* PrettyTable
* Pandas
* ujson
* NLTK

### Citation

If you use this code, then please cite our IJCAI 2018 paper:
```
@inproceedings{ijcai2018-634,
  title     = {Biased Random Walk based Social Regularization for Word Embeddings},
  author    = {Ziqian Zeng and Xin Liu and Yangqiu Song},
  booktitle = {Proceedings of the Twenty-Seventh International Joint Conference on
               Artificial Intelligence, {IJCAI-18}},
  publisher = {International Joint Conferences on Artificial Intelligence Organization},             
  pages     = {4560--4566},
  year      = {2018},
  month     = {7},
  doi       = {10.24963/ijcai.2018/634},
  url       = {https://doi.org/10.24963/ijcai.2018/634},
}

```

### License

Copyright (c) 2018 HKUST-KnowComp. All rights reserved.

Licensed under the MIT License.