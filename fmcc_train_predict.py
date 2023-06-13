# 판다랑 넘파이
import pandas as pd
import numpy as np

# 분류기
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

# 플롯용 및 평가용
import matplotlib.pyplot as plt # 곧 사용하긴 해야함
from sklearn.metrics import precision_recall_fscore_support
from mpl_toolkits.mplot3d import Axes3D # 3차원 그래프

# 모델 저장 및 불러오기용
import joblib

# 스케일러와 학습/테스트 분리기
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 파일 크기 측정용
from os.path import getsize

# Grid search 통한 최적의 C, gamma 값 구하고 최적의 모델 구하기
def search_Best_Model(x_train, y_train):

    # 하이퍼파라미터 그리드 정의
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': [1, 0.1, 0.01],
        'kernel': ['rbf'] 
    }

    # Grid search
    grid_search = GridSearchCV(SVC(), param_grid, cv=5, scoring='accuracy', return_train_score=True)
    grid_search.fit(x_train, y_train)

    # 최적의 파라미터, 정확도 출력
    print('최적의 파라미터:', grid_search.best_params_)
    print('최고 정확도:', grid_search.best_score_)

    # Grid search에서 확인된 C, gamma 값과 해당 조합의 정확도 가져오기
    cv_results_df = pd.DataFrame(grid_search.cv_results_)
    C_values = cv_results_df["param_C"].tolist()
    gamma_values = cv_results_df["param_gamma"].tolist()
    accuracy_values = cv_results_df["mean_test_score"].tolist()

    print("C_values\tGamma_values\tAccuracy_values")
    for c, g, a in zip(C_values, gamma_values, accuracy_values):
        print("{}\t\t{}\t\t{}".format(c, g, a))

    # 3D 그래프 그리기
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(C_values, gamma_values, accuracy_values)

    ax.set_xlabel('C', fontsize=12)
    ax.set_ylabel('Gamma', fontsize=12)
    ax.set_zlabel('Accuracy', fontsize=12)
    ax.set_title('Hyperparameter Grid Search Results', fontsize=16)
    ax.xaxis.labelpad = 15
    ax.yaxis.labelpad = 15
    ax.zaxis.labelpad = 15

    # 최적의 하이퍼파라미터에 대한 점 표시
    best_acc, best_C, best_gamma = max(zip(accuracy_values, C_values, gamma_values))
    ax.scatter(best_C, best_gamma, best_acc, color='r', s=100, marker='*', label="Best params: C={}, gamma={}, accuracy={}".format(best_C, best_gamma, best_acc))
    ax.legend(fontsize=12)

    plt.show()

    best_params = grid_search.best_params_
    svm = SVC(C=best_params['C'], gamma=best_params['gamma'], kernel=best_params['kernel'])

    return svm

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
    svm = search_Best_Model(X_train_std, y_train) # 최적의 C, gamma가 들어간 서포트 벡터 머신 
    print("Train started")
    svm.fit(X_train_std, y_train)
    joblib.dump(svm, './trained/svm.pkl')
    if(svm.fit_status_==0):
        print("정상 fitted")
    else:
        print("fit 문제 있음")
    print("옵션 : ",svm.get_params())
    print("특징 수 :  ",svm.n_features_in_,"개")
    print("모델 사이즈 : ", getsize('./trained/svm.pkl'),"bytes")
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


train_set(train_csv="voice_train.csv")
predict_set(test_csv="voice_test.csv")