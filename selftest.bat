@echo off
chcp 65001 >nul 2>&1
cd /d "%~dp0"

:: ============================================================
::  AirControl - 标准化自测 (编译 + lint + 测试)
::  用法示例:
::    selftest.bat                一键全量自测
::    selftest.bat --cov          附带覆盖率
::    selftest.bat --lint-fix     先自动修复风格再检查
::    selftest.bat -k mouse       透传 pytest 参数
:: ============================================================

python selftest.py %*
set EXITCODE=%errorlevel%

if %EXITCODE% neq 0 (
    echo.
    echo [自测未通过] exit code: %EXITCODE%
    pause
)

exit /b %EXITCODE%
