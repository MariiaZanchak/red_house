import scipy.io
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

#loading the data
train = scipy.io.loadmat('redhouseTrain.mat')
test1 = scipy.io.loadmat('redhouseTest1.mat')
test2 = scipy.io.loadmat('redhouseTest2.mat')

#choose room
room_num=0
while room_num not in [1,2,3,4,5,6,7,8,9,10]:
    print("Enter the room number (1,...,10):")
    room_num=int(input())
    if room_num not in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        print("Error: Wrong room number")

#making X and Y
def preprocess (train,room_num):

    disturbaces = train['d']
    inputs = train['u']
    states = train['x']
    outputs = train['y']
    schedule = train['proxy']

    disturbaces_df = pd.DataFrame(disturbaces)
    inputs_df = pd.DataFrame(inputs)
    states_df = pd.DataFrame(states)
    outputs_df = pd.DataFrame(outputs)
    schedule_df = pd.DataFrame(schedule)
    schedule_df = schedule_df.iloc[[0,1,2]]

    #choosing data for one room
    states_df = states_df.iloc[[room_num+2]]
    inputs_df=inputs_df.iloc[[room_num-1]]

    #X
    for_prediction = inputs_df
    for_prediction = for_prediction.append(states_df, ignore_index=True)
    for_prediction = for_prediction.append(schedule_df, ignore_index=True)
    for_prediction = for_prediction.append(disturbaces_df, ignore_index=True)
    for_prediction = for_prediction.T

    #Y
    outputs_df = outputs_df.iloc[[room_num+2]]
    outputs_df = outputs_df.T

    #choosing rows with NA
    xna = np.where(for_prediction.isna())[0]
    yna = np.where(outputs_df.isna())[0]
    na = np.concatenate((xna, yna))
    na = np.unique(na)

    #deleting rows with NA
    outputs_df = outputs_df.drop(na)
    for_prediction = for_prediction.drop(na)

    return for_prediction,outputs_df


X_train,Y_train=preprocess(train,room_num)
X_test1,Y_test1=preprocess(test1,room_num)
X_test2,Y_test2=preprocess(test2,room_num)

# define LinearRegression model
model_lr= LinearRegression()
# fit model
model_lr.fit(X_train, Y_train)

def summary(model):
    print("Accuracy:")
    print("On training set: ",round(model.score(X_train, Y_train) * 100,2), '%')
    print("On testing_1 set: ",round(model.score(X_test1, Y_test1) * 100,2), '%')
    print("On testing_2 set: ",round(model.score(X_test2, Y_test2) * 100,2), '%')

    Y_1 = model.predict(X_test1)
    Y_2 = model.predict(X_test2)

    # MSE
    print("MSE")

    print("On testing_1 set: ", round(mean_squared_error(Y_test1,Y_1),3))
    print("On testing_2 set: ", round(mean_squared_error(Y_test2,Y_2),3))
    return Y_1 ,Y_2


print("Multiple linear regression:")
pred_lr_1,pred_lr_2=summary(model_lr)

# Train Decision Tree Regressoin
model_dt = DecisionTreeRegressor(max_depth=5)
model_dt.fit(X_train,Y_train)

print("Decision Tree Regressoin:")
pred_dt_1,pred_dt_2=summary(model_dt)