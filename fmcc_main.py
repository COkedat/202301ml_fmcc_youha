# 자~ 파일 읽어보기 를! 할거에요
# 학습할 파일명들 저장된 ctl 파일 읽기
file_path = "fmcc_train.ctl"

# 읽어서 filenames에 저장
with open(file_path) as f:
    filenames = f.read().splitlines()

#테스트용 출력
#print(filenames)


'''

앞으로 구현해야하는것들
1. 전처리
- 지나치게 큰잡음 제거
- 볼륨 조정(?)


2. 학습
- 전처리한 사운드들을 기반으로 학습함


3. 학습 테스트
- 물론 테스트용 데이터도 잡음이 섞여있기 때문에 전처리 후 평가해야함
- 대충 그럼

'''
