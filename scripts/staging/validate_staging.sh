#!/bin/bash
# Agent Forge Staging Validation Script
# Version: 1.0.0
# Environment: staging

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
VALIDATION_LOG="$PROJECT_ROOT/logs/staging_validation_$(date +%Y%m%d_%H%M%S).log"
VALIDATION_REPORT="$PROJECT_ROOT/staging_validation_report_$(date +%Y%m%d_%H%M%S).json"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1" | tee -a "$VALIDATION_LOG"
}

log_error() {
    echo -e "${RED}[ERROR] $1${NC}" | tee -a "$VALIDATION_LOG"
}

log_success() {
    echo -e "${GREEN}[SUCCESS] $1${NC}" | tee -a "$VALIDATION_LOG"
}

log_warning() {
    echo -e "${YELLOW}[WARNING] $1${NC}" | tee -a "$VALIDATION_LOG"
}

log_info() {
    echo -e "${BLUE}[INFO] $1${NC}" | tee -a "$VALIDATION_LOG"
}

# Initialize validation results
VALIDATION_RESULTS='{"timestamp": "'$(date -u +%Y-%m-%dT%H:%M:%SZ)'", "environment": "staging", "tests": {}}'

# Function to update validation results
update_result() {
    local test_name="$1"
    local status="$2"
    local details="$3"
    
    VALIDATION_RESULTS=$(echo "$VALIDATION_RESULTS" | jq \
        --arg name "$test_name" \
        --arg status "$status" \
        --arg details "$details" \
        '.tests[$name] = {"status": $status, "details": $details, "timestamp": now | strftime("%Y-%m-%dT%H:%M:%SZ")}')
}

# Service connectivity tests
test_service_connectivity() {
    log_info "Testing service connectivity..."
    
    local services=(
        "api:8000:/health"
        "websocket:8001:/health"
        "swarm:8002:/health"
        "prometheus:9090/api/v1/status/config"
        "grafana:3000/api/health"
    )
    
    local connectivity_passed=true
    
    for service in "${services[@]}"; do
        local name=$(echo "$service" | cut -d: -f1)
        local port=$(echo "$service" | cut -d: -f2 | cut -d/ -f1)
        local endpoint=$(echo "$service" | cut -d/ -f2-)
        local url="http://localhost:$port/$endpoint"
        
        log_info "Testing $name service at $url..."
        
        if curl -f -s "$url" > /dev/null 2>&1; then
            log_success "$name service is accessible"
        else
            log_error "$name service is not accessible at $url"
            connectivity_passed=false
        fi
    done
    
    if [ "$connectivity_passed" = true ]; then
        update_result "service_connectivity" "PASSED" "All services are accessible"
        log_success "Service connectivity test passed"
    else
        update_result "service_connectivity" "FAILED" "One or more services are not accessible"
        log_error "Service connectivity test failed"
    fi
}

# Database connectivity and schema validation
test_database() {
    log_info "Testing database connectivity and schema..."
    
    # Test database connectivity
    if docker-compose -f "$PROJECT_ROOT/config/staging/staging_docker-compose.yml" exec -T postgres pg_isready -U postgres > /dev/null 2>&1; then
        log_success "Database is accessible"
        
        # Test schema
        local tables_count=$(docker-compose -f "$PROJECT_ROOT/config/staging/staging_docker-compose.yml" exec -T postgres \
            psql -U postgres -d agent_forge_staging -t -c "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'public';" | tr -d ' \n')
        
        if [ "$tables_count" -gt 0 ]; then
            log_success "Database schema is present ($tables_count tables found)"
            update_result "database_connectivity" "PASSED" "Database accessible with $tables_count tables"
        else
            log_warning "Database is accessible but no tables found"
            update_result "database_connectivity" "WARNING" "Database accessible but no tables found"
        fi
    else
        log_error "Database is not accessible"
        update_result "database_connectivity" "FAILED" "Database is not accessible"
    fi
}

# API functionality tests
test_api_functionality() {
    log_info "Testing API functionality..."
    
    local api_base="http://localhost:8000"
    local api_tests_passed=true
    
    # Test health endpoint
    log_info "Testing health endpoint..."
    if curl -f -s "$api_base/health" | jq -e '.status == "healthy"' > /dev/null 2>&1; then
        log_success "Health endpoint is working"
    else
        log_error "Health endpoint test failed"
        api_tests_passed=false
    fi
    
    # Test API documentation endpoint
    log_info "Testing API documentation..."
    if curl -f -s "$api_base/docs" > /dev/null 2>&1; then
        log_success "API documentation is accessible"
    else
        log_warning "API documentation might not be available"
    fi
    
    # Test metrics endpoint
    log_info "Testing metrics endpoint..."
    if curl -f -s "$api_base/metrics" > /dev/null 2>&1; then
        log_success "Metrics endpoint is working"
    else
        log_warning "Metrics endpoint might not be available"
    fi
    
    if [ "$api_tests_passed" = true ]; then
        update_result "api_functionality" "PASSED" "All API functionality tests passed"
        log_success "API functionality tests passed"
    else
        update_result "api_functionality" "FAILED" "Some API functionality tests failed"
        log_error "API functionality tests failed"
    fi
}

# WebSocket functionality tests
test_websocket_functionality() {
    log_info "Testing WebSocket functionality..."
    
    local ws_base="http://localhost:8001"
    
    # Test WebSocket health endpoint
    if curl -f -s "$ws_base/health" > /dev/null 2>&1; then
        log_success "WebSocket service health endpoint is working"
        update_result "websocket_functionality" "PASSED" "WebSocket service is operational"
    else
        log_error "WebSocket service health endpoint failed"
        update_result "websocket_functionality" "FAILED" "WebSocket service health check failed"
    fi
}

# Swarm coordinator functionality tests
test_swarm_functionality() {
    log_info "Testing Swarm coordinator functionality..."
    
    local swarm_base="http://localhost:8002"
    
    # Test swarm health endpoint
    if curl -f -s "$swarm_base/health" > /dev/null 2>&1; then
        log_success "Swarm coordinator health endpoint is working"
        
        # Test swarm status endpoint
        if curl -f -s "$swarm_base/status" > /dev/null 2>&1; then
            log_success "Swarm status endpoint is working"
            update_result "swarm_functionality" "PASSED" "Swarm coordinator is operational"
        else
            log_warning "Swarm status endpoint might not be available"
            update_result "swarm_functionality" "WARNING" "Swarm coordinator health OK but status endpoint unavailable"
        fi
    else
        log_error "Swarm coordinator health endpoint failed"
        update_result "swarm_functionality" "FAILED" "Swarm coordinator health check failed"
    fi
}

# Performance tests
test_performance() {
    log_info "Testing performance..."
    
    local api_base="http://localhost:8000"
    
    # Test API response time
    log_info "Testing API response time..."
    local response_time=$(curl -o /dev/null -s -w '%{time_total}' "$api_base/health")
    local response_time_ms=$(echo "$response_time * 1000" | bc -l | cut -d. -f1)
    
    if [ "$response_time_ms" -lt 1000 ]; then
        log_success "API response time is acceptable: ${response_time_ms}ms"
        update_result "performance" "PASSED" "API response time: ${response_time_ms}ms"
    else
        log_warning "API response time is slow: ${response_time_ms}ms"
        update_result "performance" "WARNING" "API response time slow: ${response_time_ms}ms"
    fi
}

# Security tests
test_security() {
    log_info "Testing security configurations..."
    
    local security_passed=true
    
    # Test for exposed sensitive endpoints
    log_info "Checking for exposed sensitive endpoints..."
    local sensitive_endpoints=(
        "/admin"
        "/debug"
        "/config"
        "/secrets"
    )
    
    for endpoint in "${sensitive_endpoints[@]}"; do
        if curl -f -s "http://localhost:8000$endpoint" > /dev/null 2>&1; then
            log_warning "Potentially sensitive endpoint exposed: $endpoint"
            security_passed=false
        fi
    done
    
    # Test for basic security headers
    log_info "Checking security headers..."
    local headers=$(curl -I -s "http://localhost:8000/health")
    
    if echo "$headers" | grep -i "x-frame-options" > /dev/null; then
        log_success "X-Frame-Options header is present"
    else
        log_warning "X-Frame-Options header is missing"
        security_passed=false
    fi
    
    if [ "$security_passed" = true ]; then
        update_result "security" "PASSED" "Security configurations are acceptable"
        log_success "Security tests passed"
    else
        update_result "security" "WARNING" "Some security issues detected"
        log_warning "Security tests found issues"
    fi
}

# Integration tests
test_integration() {
    log_info "Running integration tests..."
    
    cd "$PROJECT_ROOT"
    
    # Run Python integration tests if they exist
    if [ -f "test_integration_complete.py" ]; then
        log_info "Running Python integration tests..."
        if python test_integration_complete.py > /dev/null 2>&1; then
            log_success "Python integration tests passed"
        else
            log_warning "Python integration tests failed or had issues"
        fi
    fi
    
    # Run pytest if tests directory exists
    if [ -d "tests" ]; then
        log_info "Running pytest test suite..."
        if python -m pytest tests/ -q > /dev/null 2>&1; then
            log_success "Pytest test suite passed"
            update_result "integration_tests" "PASSED" "All integration tests passed"
        else
            log_warning "Some pytest tests failed"
            update_result "integration_tests" "WARNING" "Some integration tests failed"
        fi
    else
        log_info "No tests directory found, skipping pytest"
        update_result "integration_tests" "SKIPPED" "No tests directory found"
    fi
}

# Monitoring and observability tests
test_monitoring() {
    log_info "Testing monitoring and observability..."
    
    # Test Prometheus
    if curl -f -s "http://localhost:9090/api/v1/status/config" > /dev/null 2>&1; then
        log_success "Prometheus is accessible"
        
        # Check if targets are being scraped
        local targets=$(curl -s "http://localhost:9090/api/v1/targets" | jq -r '.data.activeTargets | length')
        if [ "$targets" -gt 0 ]; then
            log_success "Prometheus has $targets active targets"
        else
            log_warning "Prometheus has no active targets"
        fi
    else
        log_warning "Prometheus is not accessible"
    fi
    
    # Test Grafana
    if curl -f -s "http://localhost:3000/api/health" > /dev/null 2>&1; then
        log_success "Grafana is accessible"
        update_result "monitoring" "PASSED" "Monitoring stack is operational"
    else
        log_warning "Grafana is not accessible"
        update_result "monitoring" "WARNING" "Some monitoring components are not accessible"
    fi
}

# Resource usage tests
test_resource_usage() {
    log_info "Testing resource usage..."
    
    # Get container resource usage
    local containers=$(docker ps --format "table {{.Names}}\t{{.CPUPerc}}\t{{.MemUsage}}" | grep -E "agent-forge|postgres|redis")
    
    if [ ! -z "$containers" ]; then
        log_info "Container resource usage:"
        echo "$containers" | while read line; do
            log_info "  $line"
        done
        update_result "resource_usage" "PASSED" "Resource usage monitoring active"
    else
        log_warning "No resource usage data available"
        update_result "resource_usage" "WARNING" "No resource usage data available"
    fi
}

# Generate comprehensive validation report
generate_validation_report() {
    log_info "Generating validation report..."
    
    # Calculate overall status
    local failed_tests=$(echo "$VALIDATION_RESULTS" | jq '.tests | to_entries | map(select(.value.status == "FAILED")) | length')
    local warning_tests=$(echo "$VALIDATION_RESULTS" | jq '.tests | to_entries | map(select(.value.status == "WARNING")) | length')
    local passed_tests=$(echo "$VALIDATION_RESULTS" | jq '.tests | to_entries | map(select(.value.status == "PASSED")) | length')
    local total_tests=$(echo "$VALIDATION_RESULTS" | jq '.tests | length')
    
    local overall_status="PASSED"
    if [ "$failed_tests" -gt 0 ]; then
        overall_status="FAILED"
    elif [ "$warning_tests" -gt 0 ]; then
        overall_status="WARNING"
    fi
    
    # Add summary to results
    VALIDATION_RESULTS=$(echo "$VALIDATION_RESULTS" | jq \
        --arg status "$overall_status" \
        --argjson passed "$passed_tests" \
        --argjson warnings "$warning_tests" \
        --argjson failed "$failed_tests" \
        --argjson total "$total_tests" \
        '. + {"summary": {"overall_status": $status, "passed": $passed, "warnings": $warnings, "failed": $failed, "total": $total}}')
    
    # Write report to file
    echo "$VALIDATION_RESULTS" | jq '.' > "$VALIDATION_REPORT"
    
    log_info "Validation Report Summary:"
    log_info "  Total Tests: $total_tests"
    log_info "  Passed: $passed_tests"
    log_info "  Warnings: $warning_tests"
    log_info "  Failed: $failed_tests"
    log_info "  Overall Status: $overall_status"
    
    if [ "$overall_status" = "PASSED" ]; then
        log_success "Staging environment validation completed successfully!"
    elif [ "$overall_status" = "WARNING" ]; then
        log_warning "Staging environment validation completed with warnings!"
    else
        log_error "Staging environment validation failed!"
    fi
    
    log_info "Detailed validation report saved to: $VALIDATION_REPORT"
}

# Main validation function
main() {
    log_info "Starting Agent Forge staging environment validation..."
    
    # Create logs directory
    mkdir -p "$PROJECT_ROOT/logs"
    
    # Run all validation tests
    test_service_connectivity
    test_database
    test_api_functionality
    test_websocket_functionality
    test_swarm_functionality
    test_performance
    test_security
    test_integration
    test_monitoring
    test_resource_usage
    
    # Generate final report
    generate_validation_report
    
    # Return appropriate exit code
    local overall_status=$(echo "$VALIDATION_RESULTS" | jq -r '.summary.overall_status')
    if [ "$overall_status" = "FAILED" ]; then
        exit 1
    else
        exit 0
    fi
}

# Execute main function
main "$@"