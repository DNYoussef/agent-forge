#!/bin/bash
# Agent-Forge Test Runner

echo "🧪 Agent-Forge Test Suite Runner"
echo "=================================="
echo ""

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_step() {
    echo -e "${BLUE}➡️  $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

# Parse arguments
MODE=${1:-all}

case $MODE in
    unit)
        print_step "Running Unit Tests..."
        pytest tests/unit/ -v --cov=src --cov-report=term-missing
        ;;
    
    integration)
        print_step "Running Integration Tests..."
        pytest tests/integration/ -v
        ;;
    
    e2e)
        print_step "Running E2E Tests..."
        npx playwright test
        ;;
    
    performance)
        print_step "Running Performance Tests..."
        pytest tests/performance/benchmarks.py -v
        ;;
    
    coverage)
        print_step "Generating Coverage Report..."
        pytest --cov=src --cov-report=html --cov-report=term
        print_success "Coverage report: htmlcov/index.html"
        ;;
    
    quick)
        print_step "Running Quick Tests (unit only)..."
        pytest tests/unit/ -v -m "not slow"
        ;;
    
    all)
        print_step "Running All Tests..."
        
        print_step "1/4 Unit Tests..."
        pytest tests/unit/ -v --cov=src --cov-report=xml
        
        print_step "2/4 Integration Tests..."
        pytest tests/integration/ -v
        
        print_step "3/4 E2E Tests..."
        npx playwright test
        
        print_step "4/4 Performance Tests..."
        pytest tests/performance/benchmarks.py -v
        
        print_success "All tests completed!"
        print_success "Coverage report: coverage.xml"
        ;;
    
    *)
        echo "Usage: $0 {unit|integration|e2e|performance|coverage|quick|all}"
        echo ""
        echo "Options:"
        echo "  unit        - Run unit tests only"
        echo "  integration - Run integration tests only"
        echo "  e2e         - Run E2E tests only"
        echo "  performance - Run performance tests only"
        echo "  coverage    - Generate coverage report"
        echo "  quick       - Run quick tests (skip slow)"
        echo "  all         - Run all tests (default)"
        exit 1
        ;;
esac

echo ""
print_success "Done!"
