@echo off
echo ========================================
echo MedResearch AI - Frontend Setup
echo ========================================
echo.

REM Check if Node.js is installed
node --version >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo ERROR: Node.js is not installed!
    echo Please install Node.js from https://nodejs.org/
    pause
    exit /b 1
)

echo Node.js version:
node --version
echo.

REM Check if .env exists
if not exist ".env" (
    echo Creating .env file from .env.example...
    copy .env.example .env
    echo.
    echo IMPORTANT: Please edit .env and add your Auth0 Client ID!
    echo.
    echo 1. Go to: https://manage.auth0.com/dashboard
    echo 2. Navigate to: Applications -^> Applications
    echo 3. Copy your Client ID
    echo 4. Edit frontend/.env and replace YOUR_CLIENT_ID_HERE
    echo.
    pause
)

echo Installing dependencies...
echo This may take 1-2 minutes...
echo.
call npm install

IF %ERRORLEVEL% NEQ 0 (
    echo.
    echo Installation failed! Please check the error above.
    pause
    exit /b 1
)

echo.
echo ========================================
echo Setup Complete!
echo ========================================
echo.
echo Before starting the server:
echo 1. Make sure backend is running on port 8000
echo 2. Edit .env with your Auth0 Client ID (if not done)
echo.
echo To start the development server, run:
echo   npm run dev
echo.
echo Then open: http://localhost:5173
echo.
pause
