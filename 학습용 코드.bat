@echo off
 
:_MENU
cls
echo (처음일 경우 1-2-3-4-5 순서대로 실행)
echo 1. 학습용 wav로 변경 및 잡음 제거
echo 2. 테스트용 wav로 변경 및 잡음 제거
echo 3. 학습용 wav를 csv로 특징 추출 
echo 4. 테스트용 wav를 csv로 특징 추출
echo 5. SVM 학습
echo 0. 종료
choice /c:123450
 
IF %ERRORLEVEL% == 1 goto _E1
IF %ERRORLEVEL% == 2 goto _E2
IF %ERRORLEVEL% == 3 goto _E3
IF %ERRORLEVEL% == 4 goto _E4
IF %ERRORLEVEL% == 5 goto _E5
IF %ERRORLEVEL% == 0 goto _END
echo 종료..
pause
exit 

rem 1. 학습용 wav로 변경 및 잡음 제거
:_E1
start /wait python fmcc_main.py trainWav
pause
goto _MENU
 
 

rem 2. 테스트용 wav로 변경 및 잡음 제거
:_E2
start /wait python fmcc_main.py testWav
pause
goto _MENU



rem 3. 학습용 wav를 csv로 특징 추출 
:_E3
echo 학습 wav 특징 추출... (시간이 걸릴 수 있음)
start /WAIT RScript extractfeatures_from_trainWav.R
echo 학습 wav 특징 추출 완료...
pause
goto _MENU



rem 4. 테스트용 wav를 csv로 특징 추출
:_E4
echo 테스트 wav 특징 추출...
start /WAIT RScript extractfeatures_from_testWav.R
echo 테스트 wav 특징 추출 완료...
pause
goto _MENU


rem 5. SVM 학습
:_E5
echo SVM 학습...
start /wait python fmcc_main.py train
pause
goto _MENU

 
:END
exit



