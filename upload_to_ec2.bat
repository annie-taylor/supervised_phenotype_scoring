@echo off
set KEY=C:\Users\Eric\.ssh\scoring-key
set EC2=ubuntu@3.22.234.176
set REMOTE=~/supervised_phenotype_scoring
set LOCAL=E:\scoring\batches

echo.
echo Pulling latest code on EC2...
ssh -i "%KEY%" %EC2% "cd %REMOTE% && git pull"

echo.
echo Uploading batches...

echo [1/5] bk37wh86_rd75wh72_20260414
scp -i "%KEY%" -r "%LOCAL%\bk37wh86_rd75wh72_20260414" %EC2%:%REMOTE%/batches/

echo [2/5] pk100bk68_ye81br444_20260414
scp -i "%KEY%" -r "%LOCAL%\pk100bk68_ye81br444_20260414" %EC2%:%REMOTE%/batches/

echo [3/5] pu53pu22_rd75wh72_20260414
scp -i "%KEY%" -r "%LOCAL%\pu53pu22_rd75wh72_20260414" %EC2%:%REMOTE%/batches/

echo [4/5] gr73gr72_pk100bk68_20260414
scp -i "%KEY%" -r "%LOCAL%\gr73gr72_pk100bk68_20260414" %EC2%:%REMOTE%/batches/

echo [5/5] gr99or87_rd75wh72_20260414
scp -i "%KEY%" -r "%LOCAL%\gr99or87_rd75wh72_20260414" %EC2%:%REMOTE%/batches/

echo.
echo All uploads complete. SSH in and run:
echo   pkill -f ranking_app.py
echo   conda activate supervised_phenotype_scoring
echo   cd %REMOTE%
echo   nohup python ranking_app.py --all-batches ^> app.log 2^>^&1 ^&
