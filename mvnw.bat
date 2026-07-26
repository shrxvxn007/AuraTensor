@rem AuraTensor minimal Maven Wrapper for Windows
@echo off
setlocal

set MAVEN_VERSION=3.9.6
set WRAPPER_DIR=%USERPROFILE%\.m2\wrapper\dists\apache-maven-%MAVEN_VERSION%
set MVN_BIN=%WRAPPER_DIR%\apache-maven-%MAVEN_VERSION%\bin\mvn.cmd

where mvn >nul 2>&1
if %ERRORLEVEL% == 0 (
  mvn %*
  exit /b %ERRORLEVEL%
)

if exist "%MVN_BIN%" (
  call "%MVN_BIN%" %*
  exit /b %ERRORLEVEL%
)

echo [mvnw] Apache Maven not found. Downloading Maven %MAVEN_VERSION%...
if not exist "%WRAPPER_DIR%" mkdir "%WRAPPER_DIR%"

set ARCHIVE=apache-maven-%MAVEN_VERSION%-bin.zip
set URL=https://archive.apache.org/dist/maven/maven-3/%MAVEN_VERSION%/binaries/%ARCHIVE%
set TMP=%TEMP%\mvnw-%RANDOM%
mkdir "%TMP%"

powershell -Command "(New-Object System.Net.WebClient).DownloadFile('%URL%', '%TMP%\%ARCHIVE%')"
powershell -Command "Expand-Archive -Path '%TMP%\%ARCHIVE%' -DestinationPath '%WRAPPER_DIR%' -Force"

call "%MVN_BIN%" %*
exit /b %ERRORLEVEL%
