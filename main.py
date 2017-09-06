import numpy as np
import csv
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

#########################################################

#INPUT: non-float responses
#OUTPUT: float corresponding to response
def change_str_to_float(str_arg):
    vals_1 = ("never smoked", "never", "i am often early","no time at all","less than an hour a day")
    vals_3 = ("tried smoking", "former smoker", "social drinker" ,"only to avoid hurting someone", "sometimes","i am always on time", "few hours a day")
    vals_5 = ("current smoker","drink a lot","everytime it suits me","i am often running late","most of the day")
    if str_arg in vals_1:
        return float(1)
    elif str_arg in vals_3:
        return float(3)
    elif str_arg in vals_5:
        return float(5)
    else:
        return float(-2)

######################################################
#INPUT: list_old is a csv.reader file
#OUTPUT: X values, true y values
def clean_data(csv_list, num_n, num_t):

    X_matrix = np.zeros([num_t,num_n], float)
    Y_vals = []     #num of responses, minus title row
    csv_list.__next__() #skip first row, just label

    for j,row in enumerate(csv_list):
        current_row = []
        for i in row:
            if isinstance(i, float):
                current_row.append(i)
            elif i == "city":
                current_row.append(-1)
                Y_vals.append(1)
            elif i == "village":
                current_row.append(-1)
                Y_vals.append(0)
            elif isinstance(i, str) and i != "" :
                current_row.append( change_str_to_float(i) )
            else:
                #currently, no response replaced with avg, 3
                current_row.append(3)
        #only add sample if gender provided and index valid
        if -1 in current_row and j < X_matrix.shape[0]:
            X_matrix[j,:] = current_row
        elif j < X_matrix.shape[0]:
            #delete a reserved row if missing gender
            #X_matrix = np.delete(X_matrix, num_t - 1 , 0)
            #print("VAL:", num_t-1, '\n')
            num_t = num_t - 1
            #print("matrix of X:", X_matrix.shape, X_matrix)

    #delete demographics columns
    X_matrix = np.delete(X_matrix,np.arange(140, num_n), 1)
    #print("CLEANING X_ shape:", X_matrix.shape, "y shape:", len(Y_vals))
    X_no0 = X_matrix[~(X_matrix==0).all(1)] #remove 0's
    return X_no0, Y_vals

#########################################################

def use_PCA(X):
    pca=PCA(n_components=10)
    pca.fit(X_tr)
    X_trans = pca.fit_transform(X)
    #print(pca.explained_variance_ratio_, "\nX:\n", X_trans)
    return X_trans

def report(model, correct_te, correct_tr, t_te, t_tr):

    fraction_tr = (correct_tr / t_tr)
    fraction_te = (correct_te / t_te)
    print(model)
    print("Train:", correct_tr, "/", t_tr, "=", "%1.4f"% (fraction_tr))
    print("Test: ", correct_te, "/", t_te, "=", "%1.4f"% (fraction_te))


def accuracy(yhat, ytrue):
    correct = 0
    #print("EQUAL? accuracy y's",len(yhat), len(ytrue))
    for i,pred in enumerate(yhat):
        if pred == ytrue[i]:
            correct = correct + 1
    return correct

def split_data(X,y,num_t, num_tr_t):
    X_tr, X_te = np.vsplit(X,[num_tr_t])
    #print("X_tr shape:", X_tr.shape, "X_te shape:", X_te.shape)
    y_tr = y[0:num_tr_t]
    y_te = y[num_tr_t:num_t]
    #print("split:", X_te.shape, len(y_te))
    #print("tey",len(y_te), "try", len(y_tr))
    return X_tr, y_tr, X_te, y_te

def SVM_model(X_tr,y_tr, X_te, y_te):
    classify = svm.SVC()
    #print("SVMmodel :",X_tr.shape, len(y_tr))
    classify.fit(X_tr,y_tr)
    #print("SVMmodel tr,te shapeX",X_tr.shape, X_te.shape)
    yhat_tr = classify.predict(X_tr) #make predictions
    yhat_te = classify.predict(X_te) #make predictions

    return accuracy(yhat_tr, y_tr), accuracy(yhat_te, y_te) 

def MLP_model(X_tr,y_tr, X_te, y_te):
    classify = MLPClassifier(solver='lbfgs', activation='logistic')
    classify.fit(X_tr,y_tr)

    yhat_te = classify.predict(X_te) #make predictions
    yhat_tr = classify.predict(X_tr) #make predictions

    return accuracy(yhat_tr, y_tr), accuracy(yhat_te, y_te) 


def GPC_model(X_tr,y_tr, X_te, y_te):
    classify = GaussianProcessClassifier()
    classify.fit(X_tr,y_tr)
    yhat_te = classify.predict(X_te) #make predictions
    yhat_tr = classify.predict(X_tr) #make predictions
    return accuracy(yhat_tr, y_tr), accuracy(yhat_te, y_te) 


def LR_model(X_tr,y_tr, X_te, y_te):
    classify = LogisticRegression()
    classify.fit(X_tr,y_tr)
    yhat_tr = classify.predict(X_tr) #make predictions
    yhat_te = classify.predict(X_te) #make predictions
    return accuracy(yhat_tr, y_tr), accuracy(yhat_te, y_te)  


#######################

#open training data
file = open('D:/schoolwork/CS142proj/responses_data.csv')
csv_f = csv.reader(file,quoting=csv.QUOTE_NONNUMERIC)

#clean training data: convert strings, separate X's and Y's
n=150
t=1010
X, y = clean_data(csv_f, n, t)

#after cleaning, some rows/cols deleted, update t,n
t, n = X.shape
#print("t,n, 1st",t, n)
#print("X,y", X, y)

#SVM model prediction
for x in range(2,6):
    t_tr=x*100
    t_te=t-t_tr
    X_tr, y_tr, X_te, y_te = split_data(X,y,t,t_tr)
    correct_tr, correct_te = SVM_model(X_tr, y_tr, X_te, y_te)
    report("SVM", correct_te, correct_tr, t_te, t_tr)
    
print("____________________________")

#MLP
for x in range(2,6):
    t_tr=x*100
    t_te=t-t_tr
    X_tr, y_tr, X_te, y_te = split_data(X,y,t,t_tr)
    correct_tr, correct_te = MLP_model(X_tr, y_tr, X_te, y_te)
    report("MLP", correct_te, correct_tr, t_te, t_tr)
        
print("____________________________")

#GPC model prediction
for x in range(2,6):
    t_tr=x*100
    t_te=t-t_tr
    X_tr, y_tr, X_te, y_te = split_data(X,y,t,t_tr)
    correct_tr, correct_te = GPC_model(X_tr, y_tr, X_te, y_te)
    report("GPC", correct_te, correct_tr, t_te, t_tr)
    
print("____________________________")

#LR model prediction
for x in range(2,6):
    t_tr=x*100
    t_te=t-t_tr
    X_tr, y_tr, X_te, y_te = split_data(X,y,t,t_tr)
    correct_tr, correct_te = LR_model(X_tr, y_tr, X_te, y_te)
    report("LR", correct_te, correct_tr, t_te, t_tr)
    
print("____________________________")

print("")
print("")
#print("X,",X.shape, X)
#print('y,',len(y),y)
#use_PCA(X_tr)

