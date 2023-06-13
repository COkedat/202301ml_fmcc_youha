@echo off
 
:_MENU
cls
echo (처음일 경우 1-2 순서대로 실행)
echo 1. SVM 결정
echo 2. eval.pl 실행
echo 0. 종료
choice /c:120
 
IF %ERRORLEVEL% == 1 goto _E1
IF %ERRORLEVEL% == 2 goto _E2
IF %ERRORLEVEL% == 0 goto _END
echo 종료..
pause
exit 

rem 1. SVM 결정
:_E1
echo SVM 결정...
start /wait python fmcc_main.py predict
pause
goto _MENU

:_E2
echo eval.pl 실행...
perl eval.pl 유하_test_results.txt fmcc_test900_ref.txt
pause
goto _MENU
 
:END
exit



