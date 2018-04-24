import os
##################################
# Processing data
##################################

command = 'python preprocess.py --input ../data/yelp_dataset_challenge_round10 --yelp_round 10'
print(command)
os.system(command)
