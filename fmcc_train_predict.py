import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support

# 모델 저장 및 불러오기용
import joblib


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# 일단 불러오기



# 학습용 csv 갖다가 학습하기
def train_set(train_csv):
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
                        test_size=0.32,
                        random_state=0, 
                        stratify=y)

    stdsc = StandardScaler()
    X_train_std = stdsc.fit_transform(X_train)
    X_test_std = stdsc.transform(X_test)


    #Train support vector machine model
    svm = SVC()
    print("Train started")
    svm.fit(X_train_std, y_train)
    joblib.dump(svm, 'trained.pkl') 

    print("Support Vector Machine")
    print("Accuracy on training set: {:.3f}".format(svm.score(X_train_std, y_train)))
    print("Accuracy on test set: {:.3f}".format(svm.score(X_test_std, y_test)))

    y_pred_sm = svm.predict(X_test_std)
    print("Predicted value: ",y_pred_sm)

    precision, recall, fscore, support = precision_recall_fscore_support(y_test, y_pred_sm, average='micro')
    print("Precision, Recall and fscore:",precision, recall, fscore,)


    #Train random forest model
    forest = RandomForestClassifier(n_estimators=5, random_state=0)
    forest.fit(X_train_std, y_train)

    print("Random Forest")
    print("Accuracy on training set: {:.3f}".format(forest.score(X_train_std, y_train)))
    print("Accuracy on test set: {:.3f}".format(forest.score(X_test_std, y_test)))

    y_pred_forest = forest.predict(X_test_std)
    print("Predicted value: ",y_pred_forest)

    precision, recall, fscore, support = precision_recall_fscore_support(y_test, y_pred_forest, average='micro')
    print("Precision, Recall and fscore:",precision, recall, fscore,)




    #여기서부턴 임시임(테스트용)
    #Read the file which got generated using our voice samples and using code written in R.
    print("여기서부터 테스트용 csv 비교")
    test_data = pd.read_csv("voice_test.csv")
    test_data.head()

    #X, Y 생성
    X1, y1 = test_data.iloc[:, :-1].values, test_data.iloc[:, -1].values
    y1

    # 특징들을 정규화
    stdsc = StandardScaler()
    X1_std = stdsc.fit_transform(X1)

    #Predicting the target variable using SVM
    y1_pred_svm = svm.predict(X1_std)
    y1_pred_forest = forest.predict(X1_std)
    print("SVM 예측 결과: ", y1_pred_svm)
    

    # 학습할 파일명들 저장된 ctl 파일 읽기
    train_path = "./fmcc_test.ctl"

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
    
    




train_set("voice_train.csv")













# 평가용 csv 갖다가 평가하기
# 아직 다 작성 못함
def predict_set(test_csv):
    #Read the file which got generated using our voice samples and using code written in R.
    test_data = pd.read_csv(test_csv)
    test_data.head()

    svm = joblib.load('trained.pkl')

    #X, Y 생성
    X1, y1 = test_data.iloc[:, :-1].values, test_data.iloc[:, -1].values
    y1

    # 특징들을 정규화
    stdsc = StandardScaler()
    X1_std = stdsc.fit_transform(X1)

    #Predicting the target variable using SVM
    y1_pred_svm = svm.predict(X1_std)

    print("SVM: ",y1_pred_svm)

