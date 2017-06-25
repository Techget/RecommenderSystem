import pandas as pd
import numpy as np
from scipy.sparse.linalg import lsqr
from sklearn.metrics import mean_squared_error
import scipy.sparse as sps
import math

############################### Function definitions ################################

def rmse(prediction, actual_value):
    '''calcute MSE'''
    prediction = prediction[actual_value.nonzero()].flatten()
    actual_value = actual_value[actual_value.nonzero()].flatten()
    return np.sqrt(mean_squared_error(prediction, actual_value))

def predict_baseline(user, movie, timestamp):
    '''predict with improved baseline predictor'''
    time_slot = (int)((timestamp - min_time) / TIME_RANGE_LENGTH)
    prediction = all_mean_list[time_slot] + bu_and_bi_list[time_slot][user] + bu_and_bi_list[time_slot][n_users -1 + movie]
    return prediction

def cos_distance(ratings, type, smoothing=1e-9):
    '''calculate cosine distance, use smoothing to smooth the division'''
    if type == 'user':
        sim = ratings.dot(ratings.T) + smoothing
    elif type == 'movie':
        sim = ratings.T.dot(ratings) + smoothing
    norms = np.array([np.sqrt(np.diagonal(sim))])
    return (sim / norms / norms.T)

def predict_topkCF_moviepair(user, movie, timestamp, k=41):
    '''incorporate top-k neighbor, and improved baseline predictor'''
    nzero = ratings[user].nonzero()[0]
    time_slot = (int)((timestamp - min_time) / TIME_RANGE_LENGTH)
    
    choices = nzero[movie_similarity[movie, nzero].argsort()[::-1][:k]]
    
    distance_sum = 0
    for choice in choices:
        distance_sum += math.fabs(movie_similarity[movie, choice])
        
    neighbor_contribution = 0
    for choice in choices:
        neighbor_contribution += (movie_similarity[movie, choice]/distance_sum)*ratings[user, choice]
    
    movie_baseline = all_mean_list[time_slot] + bu_and_bi_list[time_slot][user] + \
        bu_and_bi_list[time_slot][n_users -1 + movie]
        
    prediction = movie_baseline + neighbor_contribution
    
    if prediction > 5: prediction = 5
    if prediction < 0.5: prediction = 0.5
    return prediction 

def predict_topkCF_userpair(user, movie, timestamp, k=41):
    '''incorporate top-k neighbor, with improved baseline predictor'''
    nzero = ratings[:,movie].nonzero()[0]
    time_slot = (int)((timestamp - min_time) / TIME_RANGE_LENGTH)  
    
    choices = nzero[user_similarity[user, nzero].argsort()[::-1][:k]]
    
    distance_sum = 0
    for choice in choices:
        distance_sum += math.fabs(user_similarity[user, choice])
        
    neighbor_contribution = 0
    for choice in choices:
        neighbor_contribution += (user_similarity[user, choice]/distance_sum)*ratings[choice, movie]
    
    user_baseline = all_mean_list[time_slot] + bu_and_bi_list[time_slot][user] + \
        bu_and_bi_list[time_slot][n_users -1 + movie]
        
    prediction = user_baseline + neighbor_contribution
    
    if prediction > 5: prediction = 5
    if prediction < 0.5: prediction = 0.5
    return prediction

############################# Start execution ######################################

names = ['user_id', 'movie_id', 'rating', 'timestamp']
trainingset_file = 'ml-100k/u2.base'
testset_file= 'ml-100k/u2.test'

print("training file: ", trainingset_file)
print("test file:", testset_file)

df = pd.read_csv(trainingset_file, sep='\t', names=names)

## Define how many time slot we need
TIME_RANGE_DIVIDEND = 3

n_users = None
n_movies = None
min_time = None
max_time = None
## Loop through dataset get indicator variables
for row in df.itertuples():
    if min_time == None:
        min_time = row[4]
        max_time = row[4]
    
    if row[4] < min_time:
        min_time = row[4]
    
    if row[4] > max_time:
        max_time = row[4]

    if n_users == None:
    	n_users = row[1]
    	n_movies = row[2]
    
    if row[1] > n_users:
    	n_users = row[1]
    if row[2] > n_movies:
    	n_movies = row[2]

## Need to plus for edge case, since for the max_time - min_time / 5 should be less than 5
TIME_RANGE_LENGTH = (int)((max_time - min_time) / TIME_RANGE_DIVIDEND) + 1

ratings = np.zeros((n_users, n_movies))
vector_b = [[] for y in range(TIME_RANGE_DIVIDEND)]
row_index = [[] for y in range(TIME_RANGE_DIVIDEND)]
col_index = [[] for y in range(TIME_RANGE_DIVIDEND)]
rate_list = [[] for y in range(TIME_RANGE_DIVIDEND)]
## Preserve the data base on timestamp
data_list = [[] for y in range(TIME_RANGE_DIVIDEND)]
## Used record index temporarily
i = [0 for y in range(TIME_RANGE_DIVIDEND)]
for row in df.itertuples():
    ratings[row[1]-1, row[2]-1] = row[3]
    time_slot = (int)((row[4] - min_time) / TIME_RANGE_LENGTH)
    
    data_list[time_slot].append(row[1:4])
    
    row_index[time_slot].append(i[time_slot])
    col_index[time_slot].append(row[1] -1)
    rate_list[time_slot].append(1)
    row_index[time_slot].append(i[time_slot])
    col_index[time_slot].append(row[2]-1 + n_users)
    rate_list[time_slot].append(1)
    vector_b[time_slot].append(row[3])
    
    i[time_slot] += 1

## Generate sparse matrixes
matrix_A_list = []
for j in range(0, TIME_RANGE_DIVIDEND):
    row = np.array(row_index[j])
    col = np.array(col_index[j])
    rate = np.array(rate_list[j])
    matrix_A_list.append(sps.coo_matrix((rate, (row, col)), shape=(len(row)/2,n_users+n_movies)))

## Clear temporary data structure
del row_index[:]
del row_index
del col_index[:]
del col_index
del rate_list[:]
del rate_list

## There should be corresponding mean value for each time slot
all_mean_list = []

for i in range(0, TIME_RANGE_DIVIDEND): 
    temp_list = []
    for row in data_list[i]:
        temp_list.append(row[2])
    temp_np_arr = np.array(temp_list)
    all_mean_list.append(np.mean(temp_np_arr[temp_np_arr!=0]))
    vector_b[i][:] = [x - all_mean_list[i] for x in vector_b[i]]


## Get moive bias and user bias respectively
bu_and_bi_list = []

print("To get user bias and movie bias, calculate least square with regularization...")
for i in range(0, TIME_RANGE_DIVIDEND):
    return_values = lsqr(matrix_A_list[i], vector_b[i], damp = 4.1)
    bu_and_bi = return_values[0]
    bu_and_bi_list.append(bu_and_bi)

## Predict and get result
print('Load test set...')
test_df = pd.read_csv(testset_file, sep='\t', names=names)
predictions = []
targets = []
print('Predict with naive base-line predictor...')
for row in test_df.itertuples():
    user, movie, actual, timestamp = row[1]-1, row[2]-1, row[3], row[4]
    predictions.append(predict_baseline(user, movie, timestamp))
    targets.append(actual)

print('Result of RMSE is: %.4f' % rmse(np.array(predictions), np.array(targets)))

## Rating to rating_delta, need to do the shift, preparation for the top-k neighbor
print('Do shift on ratings matrix...')
for row in df.itertuples():
    time_slot = (int)((row[4] - min_time) / TIME_RANGE_LENGTH)
    ratings[row[1]-1, row[2]-1] = row[3] - \
        (all_mean_list[time_slot] + bu_and_bi_list[time_slot][row[1]-1] + bu_and_bi_list[time_slot][n_users -1 + row[2]-1])

## Calculate cos distance of movie-movie pair
print('Calculate cos_distance of each movie-pair...')
movie_similarity = cos_distance(ratings, 'movie')

print('Load test set...')
test_df = pd.read_csv(testset_file, sep='\t', names=names)
predictions = []
targets = []
print('Predictor incorporates temporal parameters, regularization of biases, top-k neighbor(movie-movie pair)...')
for row in test_df.itertuples():
    user, movie, actual, timestamp = row[1]-1, row[2]-1, row[3], row[4]
    predictions.append(predict_topkCF_moviepair(user, movie, timestamp))
    targets.append(actual)

print('Result of RMSE is: %.4f' % rmse(np.array(predictions), np.array(targets)))
print()

## Calculate cos distance of user-user pair
print('Calculate cos_distance of each user-pair...')
user_similarity = cos_distance(ratings, 'user')

print('Load test set...')
test_df = pd.read_csv(testset_file, sep='\t', names=names)
predictions = []
targets = []
print('Predictor incorporates temporal parameters, regularization of biases, top-k neighbor(user-user pair)...')
for row in test_df.itertuples():
    user, movie, actual, timestamp = row[1]-1, row[2]-1, row[3], row[4]
    predictions.append(predict_topkCF_userpair(user, movie, timestamp))
    targets.append(actual)

print('Result of RMSE is: %.4f' % rmse(np.array(predictions), np.array(targets)))

