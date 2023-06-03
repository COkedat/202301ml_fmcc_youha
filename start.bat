@echo off
 
:_MENU
cls
echo (ó���� ��� 1-2-3-4-5-6 ������� ����)
echo 1. �н��� wav�� ���� �� ���� ����
echo 2. �׽�Ʈ�� wav�� ���� �� ���� ����
echo 3. �н��� wav�� csv�� Ư¡ ���� 
echo 4. �׽�Ʈ�� wav�� csv�� Ư¡ ����
echo 5. SVM �н�
echo 6. SVM ����
echo 7. eval.pl ����
echo 0. ����
choice /c:12345670
 
IF %ERRORLEVEL% == 1 goto _E1
IF %ERRORLEVEL% == 2 goto _E2
IF %ERRORLEVEL% == 3 goto _E3
IF %ERRORLEVEL% == 4 goto _E4
IF %ERRORLEVEL% == 5 goto _E5
IF %ERRORLEVEL% == 6 goto _E6
IF %ERRORLEVEL% == 7 goto _E7
IF %ERRORLEVEL% == 0 goto _END
echo ����..
pause
exit 

rem 1. �н��� wav�� ���� �� ���� ����
:_E1
start /wait python fmcc_main.py trainWav
pause
goto _MENU
 
 

rem 2. �׽�Ʈ�� wav�� ���� �� ���� ����
:_E2
start /wait python fmcc_main.py testWav
pause
goto _MENU



rem 3. �н��� wav�� csv�� Ư¡ ���� 
:_E3
echo �н� wav Ư¡ ����... (�ð��� �ɸ� �� ����)
start /WAIT RScript extractfeatures_from_trainWav.R
echo �н� wav Ư¡ ���� �Ϸ�...
pause
goto _MENU



rem 4. �׽�Ʈ�� wav�� csv�� Ư¡ ����
:_E4
echo �׽�Ʈ wav Ư¡ ����...
start /WAIT RScript extractfeatures_from_testWav.R
echo �׽�Ʈ wav Ư¡ ���� �Ϸ�...
pause
goto _MENU


rem 5. SVM �н�
:_E5
echo SVM �н�...
start /wait python fmcc_main.py train
pause
goto _MENU


rem 6. SVM ����
:_E6
echo SVM ����...
start /wait python fmcc_main.py predict
pause
goto _MENU

:_E7
echo eval.pl ����...
perl eval.pl ����_test_results.txt fmcc_test900_ref.txt
pause
goto _MENU
 
:END
exit


