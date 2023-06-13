@echo off
 
:_MENU
cls
echo (처음일 경우 1-2 순서대로 실행)
echo 1. SVM 결과예측
echo 2. eval.pl 실행
echo 0. 종료
choice /c:120
 
IF %ERRORLEVEL% == 1 goto _E1
IF %ERRORLEVEL% == 2 goto _E2
IF %ERRORLEVEL% == 0 goto _END
echo 종료..
pause
exit 

rem 1. SVM 결과예측
:_E1
echo SVM 결과예측...
start /wait python fmcc_main.py predict
pause
goto _MENU

:_E2
echo eval.pl 실행...
set /p %str1%=평가할 파일 이름 입력 (기본 : 유하_test_results.txt):
if "%str1%" == "" set str1="유하_test_results.txt"
set /p str2=원본 파일 이름 입력 (기본 : fmcc_test900_ref.txt):
if "%str2%" == "" set str2="fmcc_test900_ref.txt"
perl eval.pl %str1% %str2%
pause
goto _MENU
 
:END
exit



