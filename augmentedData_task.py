from pyspark import SparkContext
import json
import os
import numpy as np
import time
import pandas as pd
import sys
import xgboost as xgb

os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

sc = SparkContext('local[*]','Final_Project')
sc.setLogLevel("ERROR")


start_time = time.time()

all_data_files = r"C:\Users\ankus\PycharmProjects\dsci_553_competitive_project\publicdata"
input_test_path = r"C:\Users\ankus\PycharmProjects\dsci_553_competitive_project\publicdata\yelp_val.csv"
output_path_val = r"temp_output_task3.csv"

# all_data_files = sys.argv[1]
# input_test_path = sys.argv[2]
# output_path_val = sys.argv[3]


training_RDD = sc.textFile(all_data_files+'/yelp_train.csv', minPartitions=2)
header_train=training_RDD.collect()[0]
training_RDD = training_RDD.filter(lambda x: x != header_train).map(lambda x: x.split(',')).map(lambda x: (x[0], x[1], float(x[2])))
testing_RDD = sc.textFile(input_test_path, minPartitions=2).filter(lambda x: x != "user_id,business_id,stars").map(lambda x: x.split(',')).map(lambda x: (x[0], x[1])).persist()



### Calculate mean
# mean_for_business_i = training_RDD.map(lambda x: (x[1], [x[2]])).reduceByKey(lambda acc, n: acc + n).map(lambda x: (x[0], sum(x[1]) / len(x[1]))).collectAsMap()
mean_for_business_i = training_RDD.map(lambda x: (x[1], x[2])).groupByKey().map(lambda x:(x[0],float(sum(x[1])/len(x[1])))).collectAsMap()
business_user_rating_dict = training_RDD.map(lambda x: (x[1], {x[0]: x[2]})).collectAsMap()
user_business_rating_dict = training_RDD.map(lambda x: (x[0], [(x[1], x[2])])).reduceByKey(lambda acc, n: acc + n).collectAsMap()
user_business_pair_RDD_ = training_RDD.map(lambda x:(x[0],x[1])).groupByKey().mapValues(set)    #(user,{business_ids})
user_business_pair_RDD_=user_business_pair_RDD_.collectAsMap()      #user:{business_ids}
business_user_pair_RDD_ = training_RDD.map(lambda x:(x[1],x[0])).groupByKey().mapValues(set)    #(business,{user})
business_user_pair_RDD_ =business_user_pair_RDD_.collectAsMap()     #business:{user_id}

user_sum_len_RDD = training_RDD.map(lambda x:(x[0],float(x[2]))).groupByKey().mapValues(lambda x: (sum(x),len(x))).collectAsMap()           #user_id:(sum(rating),count_raters)
business_user_rating_group_RDD = training_RDD.map(lambda x:((x[1],x[0]),float(x[2]))).collectAsMap()                #(business_id,user_id):rating
business_sum_len_RDD = training_RDD.map(lambda x:(x[1],float(x[2]))).groupByKey().mapValues(lambda x: (sum(x),len(x))).collectAsMap()       #business_id:(sum(rating),count_raters)

all_rating_average=training_RDD.map(lambda x:(1,float(x[2]))).groupByKey().mapValues(lambda x:(sum(x),len(x))).collect()
default_rating_set=float(all_rating_average[0][1][0]/all_rating_average[0][1][1])


#Predict function for Item_based
def item_prediction_func(x):
    user_id=x[0]
    business_id=x[1]
    num_weight_pred = 0
    sum_weight = 0
    if user_id not in user_business_pair_RDD_ and business_id in business_user_pair_RDD_:
        pred_rating_bus=float(business_sum_len_RDD[business_id][0]/business_sum_len_RDD[business_id][1])        #new user but we can use past business data for pred_ratings
        # return tuple([user_id, business_id, 3.0])
        return tuple([user_id, business_id,pred_rating_bus])
    elif user_id not in user_business_pair_RDD_ and business_id not in business_user_pair_RDD_:
        # return tuple([user_id, business_id, 3.0])
        return tuple([user_id, business_id, default_rating_set])  # For complete coldstart, no user data and business data
    elif user_id in user_business_pair_RDD_ and business_id not in business_user_pair_RDD_:
        pred_rating_user=float(user_sum_len_RDD[user_id][0]/user_sum_len_RDD[user_id][1])         #new business but we can use past user data for pred_ratings
        # return tuple([user_id, business_id, 3.0])
        return tuple([user_id, business_id, pred_rating_user])

    # if user_id not in user_business_rating_dict and business_id not in business_user_rating_dict:
    #     return [user_id, business_id, 3.0]
    # elif user_id not in user_business_rating_dict or business_id not in business_user_rating_dict:
    #     return [user_id, business_id, 3.0]
    for business_var_i in user_business_rating_dict[user_id]:
        users_business_1 = business_user_rating_dict[business_id]
        users_business_2 = business_user_rating_dict[business_var_i[0]]
        corated_users = set(users_business_1.keys()).intersection(set(users_business_2.keys()))
        rating_avg_business_1 = 0
        rating_avg_business_2 = 0
        num_i_j = 0
        sqrt_sum_sq_i = 0
        sqrt_sum_sq_j = 0
        if len(corated_users)==0:
            similarity_w= 0.1
        else:
            for user_temp_i in corated_users:
                rating_avg_business_1 += users_business_1[user_temp_i]
                rating_avg_business_2 += users_business_2[user_temp_i]
            rating_avg_business_1/= len(corated_users)
            rating_avg_business_2/= len(corated_users)
            for user_temp_i in corated_users:
                var_1=(users_business_1[user_temp_i] - rating_avg_business_1)
                var_2=(users_business_2[user_temp_i] - rating_avg_business_2)
                num_i_j += (var_1 * var_2)
                sqrt_sum_sq_i += pow((var_1),2)
                sqrt_sum_sq_j += pow((var_2),2)
            if num_i_j == 0:
                similarity_w= 0.1
            else:
                similarity_w= (num_i_j / (pow(sqrt_sum_sq_i, 0.5) * pow(sqrt_sum_sq_j, 0.5)))
        num_weight_pred += ((business_var_i[1] - mean_for_business_i[business_var_i[0]]) * similarity_w)
        sum_weight += abs(similarity_w)
    return [user_id, business_id, float(num_weight_pred/sum_weight + mean_for_business_i[business_id])]



### Predict test set
test_pred_RDD = testing_RDD.map(lambda x: item_prediction_func(x))#.map(lambda x: [x[0],x[1],x[2]])
# print(test_pred_RDD.collect()[:3])
user_based_DF = pd.DataFrame(test_pred_RDD.collect())
user_based_DF.columns = ['user_id', 'business_id', 'user_based_Prediction']
# print(user_based_DF['user_id'],user_based_DF['business_id'])
user_based_DF['user_business'] = user_based_DF['user_id'] + user_based_DF['business_id']

#Model_based function
def model_based_preparation(X_train_data, test_data):
    with open(all_data_files + '/user.json', encoding='utf-8') as json_file:
        user_json = json_file.readlines()
        user_json_data = list(map(json.loads, user_json))
    User_pd_DF_data = pd.DataFrame(user_json_data)[['user_id', 'average_stars', 'review_count', 'useful', 'fans', 'compliment_hot', 'compliment_more',
         'compliment_note']]
    User_pd_DF_data.columns = ['user_id', 'user_stars', 'user_review_count', 'user_useful', 'user_fans', 'user_hot',
                               'user_more', 'user_note']

    with open(all_data_files + '/business.json', encoding='utf-8') as json_file:
        business_json = json_file.readlines()
        business_json_data2 = list(map(json.loads, business_json))
        business_json_data = pd.json_normalize(business_json_data2)
    business_pd_DF_data = pd.DataFrame(business_json_data)[['business_id', 'stars', 'review_count', 'is_open','attributes.BusinessAcceptsCreditCards','attributes.BusinessAcceptsBitcoin','attributes.AcceptsInsurance','attributes.RestaurantsTakeOut','attributes.RestaurantsDelivery','attributes.DriveThru','attributes.NoiseLevel','attributes.OutdoorSeating']]
    business_pd_DF_data.columns =  ['business_id', 'business_stars', 'business_review_count', 'business_open','Payment_CreditCard','Payment_Bitcoin','Payment_Insurance','RestaurantsTakeOut','RestaurantsDelivery','DriveThru','Noise_Level','Outdoor_seating']

    # business_pd_DF_data = pd.DataFrame(business_json_data)[
    #     ['business_id', 'stars', 'review_count', 'is_open', 'attributes.BusinessAcceptsCreditCards',
    #      'attributes.BusinessAcceptsBitcoin', 'attributes.AcceptsInsurance', 'attributes.RestaurantsTakeOut',
    #      'attributes.RestaurantsDelivery', 'attributes.DriveThru']]
    # business_pd_DF_data.columns = ['business_id', 'business_stars', 'business_review_count', 'business_open',
    #                                'Payment_CreditCard', 'Payment_Bitcoin', 'Payment_Insurance', 'RestaurantsTakeOut',
    #                                'RestaurantsDelivery', 'DriveThru']

    #Processing////////////////////////////////////////////////
    business_pd_DF_data['Payment_CreditCard'] = business_pd_DF_data['Payment_CreditCard'].fillna(False)
    business_pd_DF_data['Payment_Bitcoin'] = business_pd_DF_data['Payment_Bitcoin'].fillna(False)
    business_pd_DF_data['Payment_Insurance'] = business_pd_DF_data['Payment_Insurance'].fillna(False)

    business_pd_DF_data['Payment_CreditCard'].replace({'False': False, 'True': True}, inplace=True)
    business_pd_DF_data['Payment_Bitcoin'].replace({'False': False, 'True': True}, inplace=True)
    business_pd_DF_data['Payment_Insurance'].replace({'False': False, 'True': True}, inplace=True)

    business_pd_DF_data['Payment_CreditCard'] = business_pd_DF_data['Payment_CreditCard'].astype(int)
    business_pd_DF_data['Payment_Bitcoin'] = business_pd_DF_data['Payment_Bitcoin'].astype(int)
    business_pd_DF_data['Payment_Insurance'] = business_pd_DF_data['Payment_Insurance'].astype(int)

    business_pd_DF_data['Payments_CreditCard_Bitcoin_Insurance'] = business_pd_DF_data['Payment_CreditCard'] | \
                                                                   business_pd_DF_data['Payment_Bitcoin'] | \
                                                                   business_pd_DF_data['Payment_Insurance']

    business_pd_DF_data['RestaurantsTakeOut'] = business_pd_DF_data['RestaurantsTakeOut'].fillna(False)
    business_pd_DF_data['RestaurantsDelivery'] = business_pd_DF_data['RestaurantsDelivery'].fillna(False)
    business_pd_DF_data['DriveThru'] = business_pd_DF_data['DriveThru'].fillna(False)

    business_pd_DF_data['RestaurantsTakeOut'].replace({'False': False, 'True': True}, inplace=True)
    business_pd_DF_data['RestaurantsDelivery'].replace({'False': False, 'True': True}, inplace=True)
    business_pd_DF_data['DriveThru'].replace({'False': False, 'True': True}, inplace=True)

    business_pd_DF_data['RestaurantsTakeOut'] = business_pd_DF_data['RestaurantsTakeOut'].astype(int)
    business_pd_DF_data['RestaurantsDelivery'] = business_pd_DF_data['RestaurantsDelivery'].astype(int)
    business_pd_DF_data['DriveThru'] = business_pd_DF_data['DriveThru'].astype(int)

    business_pd_DF_data['Eating_options'] = business_pd_DF_data['RestaurantsTakeOut'] | business_pd_DF_data['RestaurantsDelivery'] | business_pd_DF_data['DriveThru']

    business_pd_DF_data = business_pd_DF_data.drop(['Payment_CreditCard', 'Payment_Bitcoin', 'Payment_Insurance'],axis=1)
    business_pd_DF_data = business_pd_DF_data.drop(['RestaurantsTakeOut', 'RestaurantsDelivery', 'DriveThru'], axis=1)

    business_pd_DF_data['Outdoor_seating'] = business_pd_DF_data['Outdoor_seating'].fillna(False)
    business_pd_DF_data['Outdoor_seating'].replace({'False': False, 'True': True}, inplace=True)
    business_pd_DF_data['Outdoor_seating'] = business_pd_DF_data['Outdoor_seating'].astype(int)

    # from sklearn import preprocessing
    # le = preprocessing.LabelEncoder()

    business_pd_DF_data['Noise_Level'] = business_pd_DF_data['Noise_Level'].fillna("average")
    # business_pd_DF_data["Noise_Level"] = le.fit_transform(business_pd_DF_data["Noise_Level"])

    # list2 = []
    # for i in range(0, len(business_pd_DF_data)):
    #     if int(business_pd_DF_data["Noise_Level"][i]) in [1, 3]:
    #         list1 = 1
    #         list2.append(list1)
    #     else:
    #         list1 = 0
    #         list2.append(list1)
    list2 = []
    for i in range(0, len(business_pd_DF_data)):
        if business_pd_DF_data["Noise_Level"][i] in ['average', 'quiet']:
            list1 = 0
            list2.append(list1)
            if business_pd_DF_data["Noise_Level"][i] == 'quiet':
                business_pd_DF_data.at[i, 'Noise_Level'] = 0
            elif business_pd_DF_data["Noise_Level"][i] == 'average':
                business_pd_DF_data.at[i, 'Noise_Level'] = 1
        else:
            list1 = 1
            list2.append(list1)
            if business_pd_DF_data["Noise_Level"][i] == 'loud':
                business_pd_DF_data.at[i, 'Noise_Level'] = 2
            if business_pd_DF_data["Noise_Level"][i] == 'very_loud':
                business_pd_DF_data.at[i, 'Noise_Level'] = 3

    business_pd_DF_data["noise_avg"] = list2

    business_pd_DF_data["outside_env"] = business_pd_DF_data["noise_avg"] ^ business_pd_DF_data["Outdoor_seating"]

    business_pd_DF_data = business_pd_DF_data.drop(['noise_avg'], axis=1)


    #/////////////////////////////////////////////////////


    train_cols = pd.merge(X_train_data, User_pd_DF_data, on='user_id', how='left')
    train_X = pd.merge(train_cols, business_pd_DF_data, on='business_id', how='left')

    train_y = train_X.stars.values
    train_dropped_X = train_X.drop(["stars"], axis=1)
    train_dropped_X = train_dropped_X.drop(["user_id"], axis=1)
    train_dropped_X = train_dropped_X.drop(["business_id"], axis=1)
    train_dropped_X = train_dropped_X.values

    # Fitting XGB regressor
    model = xgb.XGBRegressor(max_depth=6,
                             min_child_weight=1,
                             subsample=0.7,
                             colsample_bytree=0.6,
                             gamma=0,
                             reg_alpha=1,
                             reg_lambda=0,
                             learning_rate=0.05,
                             n_estimators=800)

    """
     0.9789664772005493
    (max_depth=6,
        min_child_weight=1,
        subsample=0.6,
        colsample_bytree=0.6,
        gamma=0,
        reg_alpha=1,
        reg_lambda=0,
        learning_rate=0.05,
        n_estimators=800)

    0.9788821639434271
    (max_depth=6,
        min_child_weight=1,
        subsample=0.7,
        colsample_bytree=0.6,
        gamma=0,
        reg_alpha=1,
        reg_lambda=0,
        learning_rate=0.05,
        n_estimators=800)

    0.9789009283624304
    (max_depth=6,
        min_child_weight=1,
        subsample=0.7,
        colsample_bytree=0.7,
        gamma=0,
        reg_alpha=1,
        reg_lambda=0,
        learning_rate=0.05,
        n_estimators=800)
    """

    model.fit(train_dropped_X, train_y)
    # print(model)

    user_id_test_data = test_data.user_id.values
    business_id_test_data = test_data.business_id.values
    test_data_DF = pd.merge(test_data, User_pd_DF_data, on='user_id', how='left')
    test_data_DF = pd.merge(test_data_DF, business_pd_DF_data, on='business_id', how='left')

    final_test_DF = test_data_DF.drop(["stars"], axis=1)
    final_test_DF = final_test_DF.drop(["user_id"], axis=1)
    final_test_DF = final_test_DF.drop(["business_id"], axis=1)
    final_test_DF = final_test_DF.values

    # test prediction
    output_pred = model.predict(final_test_DF)
    model_output_df = pd.DataFrame()
    model_output_df["user_id"] = user_id_test_data
    model_output_df["business_id"] = business_id_test_data
    model_output_df["prediction"] = output_pred

    return model_output_df
# Model_based
model_data_X = pd.DataFrame(training_RDD.collect())
model_data_X.columns = ['user_id', 'business_id', 'stars']
X_test_data = pd.read_csv(input_test_path, sep=',')
model_prediction_DF = model_based_preparation(model_data_X,X_test_data)
model_prediction_DF['user_business'] = model_prediction_DF['user_id']+model_prediction_DF['business_id']

alpha=0.8
hybrid_df = pd.merge(model_prediction_DF, user_based_DF, on='user_business', how='left')
hybrid_df['avg_prediction'] = ((alpha*pd.to_numeric(hybrid_df['prediction'])) + ((1-alpha)*pd.to_numeric(hybrid_df['user_based_Prediction'])))
# print(hybrid_df.columns)
# hybrid_df = hybrid_df.drop(["prediction", "user_business", "user_id_y", "business_id_y", "user_based_Prediction"], axis=1)

predictions_cols=X_test_data.stars.values
count_01=0
count_12=0
count_23=0
count_34=0
count_45=0


for i in predictions_cols:
    if i>=0 and i<1:
        count_01+=1
    elif i>=1 and i<2:
        count_12+=1
    elif i>=2 and i<3:
        count_23+=1
    elif i>=3 and i<4:
        count_34+=1
    elif i>=4 and i<5:
        count_45+=1

print("Error Distribution: ")
print(">=0 and i<1: ",count_01)
print(">=1 and i<2: ",count_12)
print(">=2   and i<3: ",count_23)
print(">=3 and i<4: ",count_34)
print(">=4: ",count_45)


#RMSE
test_len = X_test_data.shape[0]
assert test_len == hybrid_df.shape[0]
RMSE = 0
for i in range(test_len):
    RMSE += (hybrid_df['avg_prediction'][i] - X_test_data['stars'][i]) ** 2
RMSE = np.sqrt(RMSE / test_len)
print('RMSE: ', RMSE)

#Output_file
output = 'user_id, business_id, prediction' + '\n'
with open(output_path_val, 'w+', encoding='utf-8') as outputFile:
    for i in range(hybrid_df.shape[0]):
        output += "{},{},{}\n".format(hybrid_df['user_id_x'][i], hybrid_df['business_id_x'][i], hybrid_df['avg_prediction'][i])
    outputFile.write(output)

print('time: ', time.time() - start_time)