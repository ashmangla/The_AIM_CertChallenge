#!/bin/bash

# HandyAssist Server Restart Script
# This script kills any existing processes on ports 8000 and 3000, then restarts both servers

echo "🔧 HandyAssist Server Restart"
echo "=============================="
echo ""

# Kill backend process on port 8000
echo "Checking port 8000 (backend)..."
if lsof -ti:8000 > /dev/null 2>&1; then
    echo "  ✓ Killing existing process on port 8000"
    lsof -ti:8000 | xargs kill -9 2>/dev/null
    sleep 1
else
    echo "  ✓ Port 8000 is free"
fi

# Kill frontend process on port 3000
echo "Checking port 3000 (frontend)..."
if lsof -ti:3000 > /dev/null 2>&1; then
    echo "  ✓ Killing existing process on port 3000"
    lsof -ti:3000 | xargs kill -9 2>/dev/null
    sleep 1
else
    echo "  ✓ Port 3000 is free"
fi

echo ""
echo "Starting servers..."
echo ""

# Start backend
echo "🚀 Starting backend on port 8000..."
cd "$(dirname "$0")/api" && uvicorn app:app --reload --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

# Start frontend
echo "🚀 Starting frontend on port 3000..."
cd "$(dirname "$0")/frontend" && npm run dev &
FRONTEND_PID=$!

echo ""
echo "=============================="
echo "✅ Servers started!"
echo ""
echo "Backend PID: $BACKEND_PID"
echo "Frontend PID: $FRONTEND_PID"
echo ""
echo "Backend:  http://localhost:8000"
echo "Frontend: http://localhost:3000"
echo ""
echo "To stop servers:"
echo "  kill $BACKEND_PID $FRONTEND_PID"
echo ""
echo "Or use: lsof -ti:8000,3000 | xargs kill -9"
echo "=============================="

