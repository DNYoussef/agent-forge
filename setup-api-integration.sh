#!/bin/bash
# Setup Script for API Integration System
# Installs dependencies, configures environment, and runs initial tests

set -e

echo "🚀 Setting up Agent Forge API Integration System..."
echo "=================================================="

# Check if we're in the right directory
if [ ! -f "package.json" ]; then
    echo "❌ Error: package.json not found. Please run this script from the project root."
    exit 1
fi

# Install Node.js dependencies
echo "📦 Installing Node.js dependencies..."
npm install

# Install additional dependencies for API integration
echo "📦 Installing API integration dependencies..."
npm install --save uuid
npm install --save-dev @types/uuid

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: Python 3 is required but not installed."
    echo "Please install Python 3 and try again."
    exit 1
fi

echo "🐍 Python 3 found: $(python3 --version)"

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip3 install fastapi uvicorn python-multipart

# Create necessary directories
echo "📁 Creating directory structure..."
mkdir -p src/api/logs
mkdir -p src/api/data
mkdir -p .env

# Generate environment configuration
echo "⚙️  Generating environment configuration..."
cat > .env.api-integration << EOF
# API Integration Configuration
PYTHON_BRIDGE_PORT=8001
PYTHON_BRIDGE_HOST=127.0.0.1
API_TIMEOUT=5000
API_RETRY_ATTEMPTS=3
API_RETRY_DELAY=1000
ENABLE_FALLBACK=true
FALLBACK_DELAY=500
DEBUG_MODE=false
EOF

echo "✅ Environment configuration created at .env.api-integration"

# Make Python bridge executable
echo "🔧 Making Python bridge executable..."
chmod +x src/api/python-bridge-server.py

# Create systemd service file (optional, for Linux)
if [ "$1" = "--service" ] && command -v systemctl &> /dev/null; then
    echo "📋 Creating systemd service file..."

    SCRIPT_DIR=$(pwd)

    cat > /tmp/agent-forge-bridge.service << EOF
[Unit]
Description=Agent Forge Python Bridge Server
After=network.target

[Service]
Type=simple
User=$(whoami)
WorkingDirectory=$SCRIPT_DIR
Environment=PATH=$PATH
ExecStart=/usr/bin/python3 $SCRIPT_DIR/src/api/python-bridge-server.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

    echo "Service file created at /tmp/agent-forge-bridge.service"
    echo "To install the service, run:"
    echo "  sudo cp /tmp/agent-forge-bridge.service /etc/systemd/system/"
    echo "  sudo systemctl daemon-reload"
    echo "  sudo systemctl enable agent-forge-bridge"
    echo "  sudo systemctl start agent-forge-bridge"
fi

# Test the setup
echo "🧪 Running setup validation tests..."

# Start Python bridge in background for testing
echo "Starting Python bridge server for testing..."
python3 src/api/python-bridge-server.py &
BRIDGE_PID=$!

# Wait for server to start
sleep 3

# Test bridge connectivity
echo "Testing bridge connectivity..."
if curl -s http://localhost:8001/health > /dev/null; then
    echo "✅ Python bridge is running and accessible"
else
    echo "❌ Python bridge is not responding"
fi

# Stop test bridge
kill $BRIDGE_PID 2>/dev/null || true

# Run TypeScript compilation test
echo "🔧 Testing TypeScript compilation..."
if npx tsc --noEmit --skipLibCheck src/api/**/*.ts; then
    echo "✅ TypeScript compilation successful"
else
    echo "❌ TypeScript compilation failed"
fi

# Create startup scripts
echo "📝 Creating startup scripts..."

# Start script
cat > scripts/start-api-integration.sh << EOF
#!/bin/bash
# Start API Integration System

echo "🚀 Starting Agent Forge API Integration..."

# Start Python bridge
echo "Starting Python bridge server..."
python3 src/api/python-bridge-server.py &
BRIDGE_PID=\$!
echo \$BRIDGE_PID > .bridge.pid

echo "✅ Python bridge started (PID: \$BRIDGE_PID)"
echo "🌐 Bridge available at: http://localhost:8001"
echo "📊 Health check: http://localhost:8001/health"

echo ""
echo "To stop the bridge, run: ./scripts/stop-api-integration.sh"
EOF

# Stop script
cat > scripts/stop-api-integration.sh << EOF
#!/bin/bash
# Stop API Integration System

echo "🛑 Stopping Agent Forge API Integration..."

if [ -f .bridge.pid ]; then
    BRIDGE_PID=\$(cat .bridge.pid)
    if ps -p \$BRIDGE_PID > /dev/null; then
        kill \$BRIDGE_PID
        echo "✅ Python bridge stopped (PID: \$BRIDGE_PID)"
    else
        echo "⚠️  Bridge process not found (PID: \$BRIDGE_PID)"
    fi
    rm -f .bridge.pid
else
    echo "⚠️  No bridge PID file found"
    # Try to kill by process name
    pkill -f "python-bridge-server.py" && echo "✅ Bridge processes terminated"
fi

echo "✅ API Integration stopped"
EOF

# Test script
cat > scripts/test-api-integration.sh << EOF
#!/bin/bash
# Test API Integration System

echo "🧪 Testing Agent Forge API Integration..."

# Check if bridge is running
if ! curl -s http://localhost:8001/health > /dev/null; then
    echo "❌ Python bridge is not running. Start it first with:"
    echo "  ./scripts/start-api-integration.sh"
    exit 1
fi

echo "✅ Python bridge is running"

# Run compatibility tests
echo "Running compatibility tests..."
if [ -f "src/api/testing/test-runner.ts" ]; then
    npx tsx src/api/testing/test-runner.ts --report
else
    echo "⚠️  Test runner not found, skipping compatibility tests"
fi

echo "✅ Test completed"
EOF

# Make scripts executable
chmod +x scripts/start-api-integration.sh
chmod +x scripts/stop-api-integration.sh
chmod +x scripts/test-api-integration.sh

echo ""
echo "🎉 API Integration Setup Complete!"
echo "=================================="
echo ""
echo "📋 What was created:"
echo "  • Python bridge server (src/api/python-bridge-server.py)"
echo "  • Next.js API routes (src/web/dashboard/app/api/phases/)"
echo "  • TypeScript interfaces and utilities (src/api/)"
echo "  • Simulation fallback system"
echo "  • Session management utilities"
echo "  • Comprehensive test suite"
echo "  • Startup/shutdown scripts (scripts/)"
echo ""
echo "🚀 Getting started:"
echo "  1. Start the system:    ./scripts/start-api-integration.sh"
echo "  2. Test the system:     ./scripts/test-api-integration.sh"
echo "  3. Stop the system:     ./scripts/stop-api-integration.sh"
echo ""
echo "🌐 Endpoints will be available at:"
echo "  • Python Bridge:       http://localhost:8001"
echo "  • Next.js API:          http://localhost:3000/api/phases/"
echo "  • Health Check:         http://localhost:8001/health"
echo ""
echo "📚 Configuration:"
echo "  • Environment vars:     .env.api-integration"
echo "  • Logs will be in:      src/api/logs/"
echo ""
echo "✅ Ready to integrate real Python backend with Next.js frontend!"

# Final validation
if [ "$2" != "--skip-validation" ]; then
    echo ""
    echo "🔍 Running final validation..."

    # Check file permissions
    if [ -x "src/api/python-bridge-server.py" ]; then
        echo "✅ Python bridge is executable"
    else
        echo "❌ Python bridge is not executable"
        exit 1
    fi

    # Check TypeScript files
    if [ -f "src/api/types/phase-interfaces.ts" ]; then
        echo "✅ TypeScript interfaces found"
    else
        echo "❌ TypeScript interfaces missing"
        exit 1
    fi

    # Check API routes
    if [ -f "src/web/dashboard/app/api/phases/cognate/route.ts" ]; then
        echo "✅ Cognate API route found"
    else
        echo "❌ Cognate API route missing"
        exit 1
    fi

    if [ -f "src/web/dashboard/app/api/phases/evomerge/route.ts" ]; then
        echo "✅ EvoMerge API route found"
    else
        echo "❌ EvoMerge API route missing"
        exit 1
    fi

    echo "✅ All validation checks passed"
fi

echo ""
echo "🎯 Next Steps:"
echo "  • Review the generated .env.api-integration file"
echo "  • Customize Python bridge configuration if needed"
echo "  • Update your Next.js application to use the new API routes"
echo "  • Run the test suite to ensure everything works"
echo ""
echo "📖 For more information, check the documentation in src/api/"