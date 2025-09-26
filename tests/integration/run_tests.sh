#!/bin/bash

# Agent Forge Integration Test Runner
# Runs complete integration test suite with various options

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print colored message
print_message() {
    local color=$1
    local message=$2
    echo -e "${color}${message}${NC}"
}

# Print header
print_header() {
    echo ""
    print_message "$BLUE" "========================================="
    print_message "$BLUE" "$1"
    print_message "$BLUE" "========================================="
    echo ""
}

# Check if services are running
check_services() {
    print_header "Checking Services"

    # Check API server
    if curl -s http://localhost:8000/api/v1/info > /dev/null 2>&1; then
        print_message "$GREEN" "✓ API server is running (port 8000)"
    else
        print_message "$YELLOW" "⚠ API server not detected (port 8000)"
        print_message "$YELLOW" "  Start with: uvicorn src.api.main:app --reload"
    fi

    # Check frontend
    if curl -s http://localhost:3000 > /dev/null 2>&1; then
        print_message "$GREEN" "✓ Frontend is running (port 3000)"
    else
        print_message "$YELLOW" "⚠ Frontend not detected (port 3000)"
        print_message "$YELLOW" "  Start with: cd src/web/dashboard && npm run dev"
    fi
}

# Run API tests
run_api_tests() {
    print_header "Running API Integration Tests"
    pytest tests/integration/api/ -v --tb=short -m "not slow"
}

# Run WebSocket tests
run_websocket_tests() {
    print_header "Running WebSocket Integration Tests"
    pytest tests/integration/websocket/ -v --tb=short
}

# Run E2E Python tests
run_e2e_python_tests() {
    print_header "Running E2E Python Tests"
    pytest tests/integration/e2e/test_phase_execution.py -v --tb=short
}

# Run E2E UI tests
run_e2e_ui_tests() {
    print_header "Running E2E UI Tests (Playwright)"
    npx playwright test tests/integration/e2e/test_ui_integration.spec.ts
}

# Run validation suite
run_validation_tests() {
    print_header "Running System Validation Tests"
    pytest tests/integration/validation/ -v --tb=short
}

# Run all tests
run_all_tests() {
    print_header "Running Complete Integration Test Suite"

    echo "1. API Tests..."
    pytest tests/integration/api/ -v --tb=short -m "not slow" || true

    echo ""
    echo "2. WebSocket Tests..."
    pytest tests/integration/websocket/ -v --tb=short || true

    echo ""
    echo "3. E2E Python Tests..."
    pytest tests/integration/e2e/test_phase_execution.py -v --tb=short || true

    echo ""
    echo "4. Validation Tests..."
    pytest tests/integration/validation/ -v --tb=short || true

    echo ""
    echo "5. UI Tests (Playwright)..."
    npx playwright test tests/integration/e2e/test_ui_integration.spec.ts || true
}

# Run with coverage
run_with_coverage() {
    print_header "Running Tests with Coverage"

    pytest tests/integration/ \
        --cov=src/api \
        --cov-report=html \
        --cov-report=term-missing \
        --tb=short \
        -v

    print_message "$GREEN" "\n✓ Coverage report generated at htmlcov/index.html"
}

# Run parallel tests
run_parallel() {
    print_header "Running Tests in Parallel"
    pytest tests/integration/ -n auto -v --tb=short
}

# Run specific test file
run_specific() {
    local test_file=$1
    print_header "Running Specific Test: $test_file"
    pytest "$test_file" -v --tb=short
}

# Generate test report
generate_report() {
    print_header "Generating Test Reports"

    # Python tests with JUnit XML
    pytest tests/integration/ \
        --junitxml=test-results/integration-results.xml \
        --html=test-results/integration-report.html \
        --self-contained-html \
        -v

    # Playwright report
    npx playwright test --reporter=html

    print_message "$GREEN" "\n✓ Reports generated:"
    print_message "$GREEN" "  - test-results/integration-report.html"
    print_message "$GREEN" "  - playwright-report/index.html"
}

# Clean test artifacts
clean_artifacts() {
    print_header "Cleaning Test Artifacts"

    rm -rf htmlcov/
    rm -rf test-results/
    rm -rf playwright-report/
    rm -rf .pytest_cache/
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

    print_message "$GREEN" "✓ Test artifacts cleaned"
}

# Show help
show_help() {
    cat << EOF
Agent Forge Integration Test Runner

Usage: ./run_tests.sh [command]

Commands:
    all             Run all integration tests (default)
    api             Run API integration tests only
    websocket       Run WebSocket tests only
    e2e             Run E2E Python tests only
    ui              Run UI tests (Playwright) only
    validation      Run validation suite only
    coverage        Run tests with coverage report
    parallel        Run tests in parallel
    check           Check if services are running
    clean           Clean test artifacts
    report          Generate comprehensive test reports
    help            Show this help message

Examples:
    ./run_tests.sh                  # Run all tests
    ./run_tests.sh api              # Run API tests only
    ./run_tests.sh coverage         # Run with coverage
    ./run_tests.sh parallel         # Run in parallel

Environment Variables:
    API_BASE_URL    API server URL (default: http://localhost:8000)
    UI_BASE_URL     Frontend URL (default: http://localhost:3000)

EOF
}

# Main script
main() {
    local command=${1:-all}

    case $command in
        all)
            check_services
            run_all_tests
            ;;
        api)
            run_api_tests
            ;;
        websocket)
            run_websocket_tests
            ;;
        e2e)
            run_e2e_python_tests
            ;;
        ui)
            run_e2e_ui_tests
            ;;
        validation)
            run_validation_tests
            ;;
        coverage)
            run_with_coverage
            ;;
        parallel)
            run_parallel
            ;;
        check)
            check_services
            ;;
        clean)
            clean_artifacts
            ;;
        report)
            generate_report
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            if [ -f "$command" ]; then
                run_specific "$command"
            else
                print_message "$RED" "Unknown command: $command"
                echo ""
                show_help
                exit 1
            fi
            ;;
    esac
}

# Run main function
main "$@"