import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support
#일단 불러오기



# 학습용 csv 갖다가 학습하기
def train_set(train_csv):
    train_data = pd.read_csv(train_csv)
    train_data.head()
    train_data.groupby("label").count()

    #라벨 칼럼 인코딩하기, Female 은 0, male 은 1
    class_mapping = {label: idx for idx, label in enumerate(np.unique(train_data['label']))}
    class_mapping

# 평가용 csv 갖다가 평가하기
def predict_set(test_csv):
    test_data = pd.read_csv(test_csv)
    test_data.head()
