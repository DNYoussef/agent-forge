@echo off
echo Starting Agent Forge Cognate System with ACT Titans Architecture
echo ============================================================echo.

echo Starting WebSocket Server (Port 8085)...
start "WebSocket Server" cmd /k "cd /d %~dp0 && python agent_forge\api\websocket_server.py"

timeout /t 2

echo Starting Python Bridge Server (Port 8001)...
start "Python Bridge" cmd /k "cd /d %~dp0 && python agent_forge\api\python_bridge_server.py"

timeout /t 2

echo Starting Dashboard (Port 3000)...
start "Dashboard" cmd /k "cd /d %~dp0\src\web\dashboard && npm run dev"

timeout /t 5

echo.
echo ============================================================echo All servers started!
echo.
echo - WebSocket Server: http://localhost:8085
echo - Python Bridge API: http://localhost:8001
echo - Dashboard: http://localhost:3000/phases/cognate
echo.
echo To test the system:
echo 1. Open http://localhost:3000/phases/cognate
echo 2. Click "Start Pretraining"
echo 3. Watch the 3 orbs ripple and change color as training progresses
echo 4. Progress bars should update in real-time
echo.
echo Press any key to open the dashboard...
pause > nul

start http://localhost:3000/phases/cognate