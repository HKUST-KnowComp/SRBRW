THEANO_FLAGS="floatX=float32,device=cuda0,cuda.root=/usr/local/cuda,on_unused_input=ignore,optimizer=fast_compile" python dc.py \
--train data/NN_train_one_fifth_rd10.txt \
--dev data/NN_dev_one_fifth_rd10.txt \
--test data/NN_test_one_fifth_rd10.txt \
--save model/bias_rd10_hlstm_fix.pkl.gz \
--embedding embs/swe_with_bias_randomwalk_word_vec_rd10_l8p0_r0p25_ph10_pl60_p0p5_q1p0_a0p12_bw2p0.txt \
--user_embs embs/swe_with_bias_randomwalk_user_vec_rd10_l8p0_r0p25_ph10_pl60_p0p5_q1p0_a0p12_bw2p0.txt \
--layer lstm \
--max_epochs 30 \
--hierarchical 1 --user_atten 1 --user_atten_base 0