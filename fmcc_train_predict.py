# 판다랑 넘파이
import pandas as pd
import numpy as np

# 분류기
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# 플롯용 및 평가용
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support

# 모델 저장 및 불러오기용
import joblib

# 스케일러와 학습/테스트 분리기
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 학습용 csv 갖다가 학습하기
def train_set(train_csv):
    #파일명 비어있을 경우
    if(len(train_csv)==0):
        print("파일명이 비어있으므로 voice_train.csv 으로 진행")
        train_csv="voice_train.csv"

    #R에서 뽑은 학습용 CSV 불러오기
    train_data = pd.read_csv(train_csv)
    train_data.head()
    train_data.groupby("label").count()

    # 라벨 칼럼 인코딩하기, Female 은 0, male 은 1
    class_mapping = {label: idx for idx, label in enumerate(np.unique(train_data['label']))}
    class_mapping

    # 클래스 라벨의 strings을 integers로 변환
    train_data['label'] = train_data['label'].map(class_mapping)

    # X,y를 생성하고 데이터셋을 학습용과 평가용으로 나누기
    X, y = train_data.iloc[:, :-1].values, train_data.iloc[:, -1].values

    X_train, X_test, y_train, y_test =\
        train_test_split(X, y, 
                        test_size=0.35,
                        random_state=0, 
                        stratify=y)

    stdsc = StandardScaler()
    X_train_std = stdsc.fit_transform(X_train)
    X_test_std = stdsc.transform(X_test)


    # Train support vector machine model
    # 서포트 벡터 머신 모델 학습
    svm = SVC()
    print("Train started")
    svm.fit(X_train_std, y_train)
    joblib.dump(svm, './trained/svm.pkl')
    if(svm.fit_status_==0):
        print("정상 fitted")
    else:
        print("fit 문제 있음")
    print("특징 수 :  ",svm.n_features_in_)
    print("옵션 : ",svm.get_params())

    print("Support Vector Machine")
    print("Accuracy on training set: {:.3f}".format(svm.score(X_train_std, y_train)))
    print("Accuracy on test set: {:.3f}".format(svm.score(X_test_std, y_test)))

    y_pred_sm = svm.predict(X_test_std)
    print("Predicted value: ",y_pred_sm)

    precision, recall, fscore, support = precision_recall_fscore_support(y_test, y_pred_sm, average='micro')
    print("Precision, Recall and fscore:",precision, recall, fscore,)


# 평가용 csv 갖다가 평가하기
def predict_set(test_csv):
    #파일명 비어있을 경우
    if(len(test_csv)==0):
        print("파일명이 비어있으므로 voice_test.csv 으로 진행")
        test_csv="voice_test.csv"

    #R에서 뽑은 테스트용 CSV 불러오기
    test_data = pd.read_csv(test_csv)
    test_data.head()


    svm= joblib.load('./trained/svm.pkl')

    #X, Y 생성
    X1, y1 = test_data.iloc[:, :-1].values, test_data.iloc[:, -1].values
    y1

    # 특징들을 정규화
    stdsc = StandardScaler()
    X1_std = stdsc.fit_transform(X1)
   
    #Predicting the target variable using SVM
    y1_pred_svm = svm.predict(X1_std)
    #y1_pred_forest = forest.predict(X1_std)
    

    # 학습할 파일명들 저장된 ctl 파일 읽기
    train_path = "./fmcc_test900.ctl"

    # 읽어서 trainNames에 저장
    with open(train_path) as f:
        trainNames = f.read().splitlines()

    # 결과 작성
    with open("./유하_test_results.txt", 'w+t') as f:
        for i in range(len(trainNames)):
            f.write(trainNames[i])
            f.write(" ")
            if(y1_pred_svm[i].item()==0):
                f.write("feml")
            elif(y1_pred_svm[i].item()==1):
                f.write("male") 
            f.write("\n")


