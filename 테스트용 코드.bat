@echo off
 
:_MENU
cls
echo (ó���� ��� 1-2 ������� ����)
echo 1. SVM �������
echo 2. eval.pl ����
echo 0. ����
choice /c:120
 
IF %ERRORLEVEL% == 1 goto _E1
IF %ERRORLEVEL% == 2 goto _E2
IF %ERRORLEVEL% == 0 goto _END
echo ����..
pause
exit 

rem 1. SVM �������
:_E1
echo SVM �������...
start /wait python fmcc_main.py predict
pause
goto _MENU

:_E2
echo eval.pl ����...
set /p %str1%=���� ���� �̸� �Է� (�⺻ : ����_test_results.txt):
if "%str1%" == "" set str1="����_test_results.txt"
set /p str2=���� ���� �̸� �Է� (�⺻ : fmcc_test900_ref.txt):
if "%str2%" == "" set str2="fmcc_test900_ref.txt"
perl eval.pl %str1% %str2%
pause
goto _MENU
 
:END
exit



