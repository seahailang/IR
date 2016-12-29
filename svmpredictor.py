'''
#using svm to predict
# preprocesing:
# using chi_square to selecte 5000 relevant feature
# using SVD to decomposition the feature to 100 dimantion
# predictor
'''

from sklearn.svm import SVC
from sklearn.decomposition import TruncatedSVD,FastICA,LatentDirichletAllocation,RandomizedPCA
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2,f_classif,mutual_info_classif
from sklearn.externals import joblib
from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

import numpy as np
from scipy.sparse import vstack
import scipy.sparse as sp

import pickle
import os
import winsound
#from randomsvm import randomsvm

from multiprocessing import Process
from datetime import datetime

from readfile import *

trainfile = os.path.join(BasePath,'temp\csrtrain_pickle')
testfile = os.path.join(BasePath,'temp\csrtest_pickle')
labelfile = os.path.join(BasePath,'temp/train_info')
test_info = os.path.join(BasePath,'temp/test_info')
resultfile = os.path.join(BasePath,'data/result.csv')

################################################################
######调参看这里########

#特征选择的参数

selector = [
            SelectKBest(chi2,k=300000),
            SelectKBest(chi2,k=300000),
            SelectKBest(chi2,k=300000)
            ]

# selector = [
#             SelectKBest(mutual_info_classif,k=450),
#             SelectKBest(mutual_info_classif,k=450),
#             SelectKBest(mutual_info_classif,k=450)
#             ]



# #支持向量机的参数
clf= [
        SVC(1),
        SVC(1),
        SVC(1)
        ]

svd = [
        TruncatedSVD(500),
        TruncatedSVD(500),
        TruncatedSVD(500)
        ]
# clf= [
#         LogisticRegression(penalty='l1',C=0.08),
#         LogisticRegression(penalty='l1',C=0.08),
#         LogisticRegression(penalty='l1',C=0.08)
#         ]

# clf = [
#         RandomForestClassifier(200),
#         RandomForestClassifier(200),
#         RandomForestClassifier(200)

#         ]

###############################################################

def getdata(filename = trainfile):
    with open(filename,'rb') as file:
        data = pickle.load(file)
    data.format = 'csr'
    # data = data.transpose().tocsr()
    return data

def getlabel(filename = labelfile):
    with open(filename,'rb') as file:
        label = np.array(pickle.load(file))
    age_label = label[:,1]
    gender_label = label[:,2]
    edu_label = label[:,3]
    return age_label,gender_label,edu_label


def worker(selector,svd,clf,data,label,lid,testdata,i):
    print(datetime.now())
    print(data.shape)
    mdata = data[lid[i]]
    mlabel = label[i][lid[i]]
    mdata = selector[i].fit_transform(mdata,mlabel)
    mtest = selector[i].transform(testdata)
    rowflag = mdata.shape[0]
    mdataset = vstack([mdata,mtest],format = 'csr')
    mdataset = svd[i].fit_transform(mdataset)
    mdata = mdataset[:rowflag,:]
    mtest = mdataset[rowflag:,:]
    del(mdataset)


    clf[i].fit(mdata,mlabel)
    temp = clf[i].predict(mtest)
    with open(BasePath+'/temp/%d'%(i),'wb') as file:
        pickle.dump(temp,file)
    print(type(result[i]))
    print(len(result[i]))
    print(datetime.now())



if __name__=='__main__':
    #label
    age,gender,edu = getlabel()
    print(1)
    #load the data
    data = getdata()



    age = np.array(age)
    gender = np.array(gender)
    edu = np.array(edu)

    label = [age,gender,edu]

    #ignore the unlabeled data
    aid = np.array([False if i =='0' else True for i in age]) 
    gid = np.array([False if i =='0' else True for i in gender]) 
    eid = np.array([False if i =='0' else True for i in edu])

    lid = [aid,gid,eid]




    # #load test data
    with open(testfile,'rb') as file:
        testdata = pickle.load(file)
        testdata.format ='csr' 


    


    # #using svd to decomposition the train and test data
    # svd = TruncatedSVD(300)
    # # # # svd = RandomizedPCA(100)
    # rowflag = data.shape[0]

    # with open(BasePath+'/temp/data','rb') as file:
    #     mdataset=pickle.load(file)

    # mdataset = vstack([data,testdata],format = 'csr')
    # mdataset = svd.fit_transform(mdataset)
    # data = mdataset[:rowflag,:]
    # testdata = mdataset[rowflag:,:]
    # # with open(BasePath+'/temp/data','wb') as file:
    # #     pickle.dump(mdataset,file)
    # testdata = testdata[:]
    # del(mdataset)


    



    # p1 = Process(target = worker,args=(selector,svd,clf,data,label,lid,testdata,0))
    # p2 = Process(target = worker,args=(selector,svd,clf,data,label,lid,testdata,1))
    # p3 = Process(target = worker,args=(selector,svd,clf,data,label,lid,testdata,2))

    # p1.start()
    # p2.start()
    # p3.start()

    # p1.join()
    # p2.join()
    # p3.join()
    # print(result)





    result = []
    for i in range(3):
        mdata = data[lid[i]]
        mtest = testdata[:]
        mlabel = label[i][lid[i]]

        #using chi_square to select feature in train and test
        mdata = selector[i].fit_transform(mdata,mlabel)
        mtest = selector[i].transform(mtest)


        using svd to decomposition the train and test data
        svd = TruncatedSVD(500)
        # # # svd = RandomizedPCA(100)
        rowflag = mdata.shape[0]
        mdataset = vstack([mdata,mtest],format = 'csr')
        mdataset = svd.fit_transform(mdataset)
        mdata = mdataset[:rowflag,:]
        mtest = mdataset[rowflag:,:]

        delidx=[]
        for j in range(5):
            mean = mdata[:,j].mean()
            var = mdata[:,j].var()

            for k in range(len(mdata)):
            
                if mdata[k,j]>mean+6*var or mdata[k,j]<mean-6*var:
                    delidx.append(k)
        idx = np.array([k for k in range(len(mlabel)) if k not in delidx])

        mdata = mdata[idx]
        mlabel = mlabel[idx]
        print(i)


        # cutclf=LogisticRegression(penalty='l1',C=0.08)
        # idx = np.random.randint(len(mlabel),size=int(len(mlabel)/2))

        # idx2 = np.array([i for i in range(len(mlabel)) if i not in idx ])

        # cutclf.fit(mdata[idx],mlabel[idx])

        # rlt = cutclf.predict(mdata[idx2])
        # cut1 = np.array([idx2[i]  for i in range(len(rlt)) if mlabel[idx2[i]]==rlt[i]])

        # cutclf.fit(mdata[cut1],mlabel[cut1])
        # rlt = cutclf.predict(mdata[idx])
        # cut2 = np.array([idx[i] for i in range(len(rlt)) if mlabel[idx[i]]==rlt[i]])

        # cut = np.array(list(cut1)+list(cut2))

        # mdata = mdata[cut]
        # mlabel = mlabel[cut]

        # ###if we want cross validation
        # ####clf 是要调节的参数之一
        # cv_score.append(cross_val_score(clf[i],mdata,mlabel,cv=3,n_jobs=-1))
        # print(i,i)


        # ##################################################################
        # 下面的代码在CV的时候不需要
        # using svm to classify the test
        clf[i].fit(mdata,mlabel)

        mpredict = clf[i].predict(mtest)

        result.append(mpredict)
        del(mdata)
        del(mtest)

    # score = np.mean([np.mean(cv) for cv in cv_score])
    # print(U'预期的得分是')
    # print(score)
    # print(U'每一项的得分是')
    # for cv in cv_score:
    #     print(cv)
    # winsound.Beep(600,2000)
        

#############################################################################################
    #下面的代码是预测用的，cv的时候没用
    #write the result to file
    with open(test_info,'rb') as file:
        test = pickle.load(file)

    # for i in range(3):
    #     with open(BasePath+'/temp/%d'%(i),'rb') as file:
    #             result[i]=pickle.load(file)

    with open(resultfile,'w') as file:
        print(len(test))
        for i in range(len(test)):
            file.write(test[i][0]+' '+result[0][i]+' '+result[1][i]+' '+result[2][i]+'\n')
            # print(test[i][0]+' '+result[0][i]+' '+result[1][i]+' '+result[2][i]+'\n')








