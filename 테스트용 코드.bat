@echo off
 
:_MENU
cls
echo (ó���� ��� 1-2 ������� ����)
echo 1. SVM ����
echo 2. eval.pl ����
echo 0. ����
choice /c:120
 
IF %ERRORLEVEL% == 1 goto _E1
IF %ERRORLEVEL% == 2 goto _E2
IF %ERRORLEVEL% == 0 goto _END
echo ����..
pause
exit 

rem 1. SVM ����
:_E1
echo SVM ����...
start /wait python fmcc_main.py predict
pause
goto _MENU

:_E2
echo eval.pl ����...
perl eval.pl ����_test_results.txt fmcc_test900_ref.txt
pause
goto _MENU
 
:END
exit



