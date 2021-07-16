@ECHO off

REM upgrade packaging tools
pip install --upgrade pip
pip install --upgrade build

REM install requirements
pip install -r dependencies.txt

REM build package
python -m build

REM install
pushd dist
for /f "delims=" %%i in ('dir /b *.whl') do set whl=%%i
popd
pip install dist/%whl%

PAUSE
