#!/bin/bash
# Agent Forge Staging Monitoring Script
# Version: 1.0.0
# Environment: staging

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
MONITORING_LOG="$PROJECT_ROOT/logs/staging_monitoring_$(date +%Y%m%d_%H%M%S).log"
ALERT_LOG="$PROJECT_ROOT/logs/staging_alerts_$(date +%Y%m%d_%H%M%S).log"
METRICS_FILE="$PROJECT_ROOT/logs/staging_metrics_$(date +%Y%m%d_%H%M%S).json"

# Default monitoring interval (seconds)
MONITORING_INTERVAL=${MONITORING_INTERVAL:-30}
ALERT_THRESHOLD_CPU=80
ALERT_THRESHOLD_MEMORY=85
ALERT_THRESHOLD_RESPONSE_TIME=2000  # ms
ALERT_THRESHOLD_ERROR_RATE=5        # percentage

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Monitoring state
MONITORING_ACTIVE=true
START_TIME=$(date +%s)

# Logging functions
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1" | tee -a "$MONITORING_LOG"
}

log_error() {
    echo -e "${RED}[ERROR] $1${NC}" | tee -a "$MONITORING_LOG" | tee -a "$ALERT_LOG"
}

log_success() {
    echo -e "${GREEN}[SUCCESS] $1${NC}" | tee -a "$MONITORING_LOG"
}

log_warning() {
    echo -e "${YELLOW}[WARNING] $1${NC}" | tee -a "$MONITORING_LOG" | tee -a "$ALERT_LOG"
}

log_info() {
    echo -e "${BLUE}[INFO] $1${NC}" | tee -a "$MONITORING_LOG"
}

log_metric() {
    echo -e "${CYAN}[METRIC] $1${NC}" | tee -a "$MONITORING_LOG"
}

log_alert() {
    echo -e "${MAGENTA}[ALERT] $1${NC}" | tee -a "$MONITORING_LOG" | tee -a "$ALERT_LOG"
}

# Signal handlers
cleanup() {
    MONITORING_ACTIVE=false
    log_info "Monitoring stopped by user"
    generate_monitoring_report
    exit 0
}

trap cleanup SIGINT SIGTERM

# Check if services are running
check_service_availability() {
    local services=(
        "api:8000:/health"
        "websocket:8001:/health"
        "swarm:8002:/health"
    )
    
    local timestamp=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    local availability_data="{\"timestamp\": \"$timestamp\", \"services\": {}}"
    
    for service in "${services[@]}"; do
        local name=$(echo "$service" | cut -d: -f1)
        local port=$(echo "$service" | cut -d: -f2 | cut -d/ -f1)
        local endpoint=$(echo "$service" | cut -d/ -f2-)
        local url="http://localhost:$port/$endpoint"
        
        local status="down"
        local response_time=0
        
        # Measure response time
        local start_time=$(date +%s%N)
        if curl -f -s --max-time 5 "$url" > /dev/null 2>&1; then
            local end_time=$(date +%s%N)
            response_time=$(( (end_time - start_time) / 1000000 ))  # Convert to milliseconds
            status="up"
            log_success "$name service is available (${response_time}ms)"
        else
            log_error "$name service is not available"
        fi
        
        # Add to availability data
        availability_data=$(echo "$availability_data" | jq \
            --arg name "$name" \
            --arg status "$status" \
            --argjson response_time "$response_time" \
            '.services[$name] = {"status": $status, "response_time_ms": $response_time}')
        
        # Alert on high response time
        if [ "$status" = "up" ] && [ "$response_time" -gt "$ALERT_THRESHOLD_RESPONSE_TIME" ]; then
            log_alert "High response time for $name: ${response_time}ms (threshold: ${ALERT_THRESHOLD_RESPONSE_TIME}ms)"
        fi
    done
    
    echo "$availability_data"
}

# Monitor container resource usage
monitor_container_resources() {
    local timestamp=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    local resource_data="{\"timestamp\": \"$timestamp\", \"containers\": {}}"
    
    # Get Docker stats for Agent Forge containers
    local containers=$(docker ps --format "{{.Names}}" | grep -E "agent-forge|staging" | head -10)
    
    if [ -z "$containers" ]; then
        log_warning "No Agent Forge containers found"
        echo "$resource_data"
        return
    fi
    
    while read -r container; do
        if [ ! -z "$container" ]; then
            # Get container stats
            local stats=$(docker stats "$container" --no-stream --format "table {{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}\t{{.NetIO}}\t{{.BlockIO}}" | tail -n 1)
            
            if [ ! -z "$stats" ]; then
                local cpu_percent=$(echo "$stats" | awk '{print $1}' | sed 's/%//')
                local memory_usage=$(echo "$stats" | awk '{print $2}')
                local memory_percent=$(echo "$stats" | awk '{print $3}' | sed 's/%//')
                local network_io=$(echo "$stats" | awk '{print $4}')
                local disk_io=$(echo "$stats" | awk '{print $5}')
                
                # Convert percentages to numbers for comparison
                local cpu_num=$(echo "$cpu_percent" | cut -d. -f1)
                local mem_num=$(echo "$memory_percent" | cut -d. -f1)
                
                log_metric "$container: CPU=${cpu_percent}% MEM=${memory_percent}% (${memory_usage})"
                
                # Add to resource data
                resource_data=$(echo "$resource_data" | jq \
                    --arg name "$container" \
                    --arg cpu "$cpu_percent" \
                    --arg memory_usage "$memory_usage" \
                    --arg memory_percent "$memory_percent" \
                    --arg network_io "$network_io" \
                    --arg disk_io "$disk_io" \
                    '.containers[$name] = {
                        "cpu_percent": $cpu,
                        "memory_usage": $memory_usage,
                        "memory_percent": $memory_percent,
                        "network_io": $network_io,
                        "disk_io": $disk_io
                    }')
                
                # Alerts for high resource usage
                if [ "${cpu_num:-0}" -gt "$ALERT_THRESHOLD_CPU" ]; then
                    log_alert "High CPU usage for $container: ${cpu_percent}% (threshold: ${ALERT_THRESHOLD_CPU}%)"
                fi
                
                if [ "${mem_num:-0}" -gt "$ALERT_THRESHOLD_MEMORY" ]; then
                    log_alert "High memory usage for $container: ${memory_percent}% (threshold: ${ALERT_THRESHOLD_MEMORY}%)"
                fi
            fi
        fi
    done <<< "$containers"
    
    echo "$resource_data"
}

# Monitor database health
monitor_database() {
    local timestamp=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    local db_data="{\"timestamp\": \"$timestamp\", \"database\": {}}"
    
    # Check PostgreSQL connection
    if docker-compose -f "$PROJECT_ROOT/config/staging/staging_docker-compose.yml" exec -T postgres pg_isready -U postgres > /dev/null 2>&1; then
        log_success "Database is accessible"
        
        # Get database stats
        local db_stats=$(docker-compose -f "$PROJECT_ROOT/config/staging/staging_docker-compose.yml" exec -T postgres \
            psql -U postgres -d agent_forge_staging -t -c "
                SELECT 
                    (SELECT count(*) FROM pg_stat_activity WHERE state = 'active') as active_connections,
                    (SELECT count(*) FROM pg_stat_activity) as total_connections,
                    (SELECT pg_database_size('agent_forge_staging')) as database_size;
            " | tr -s ' ' | sed 's/^[ \t]*//;s/[ \t]*$//' 2>/dev/null)
        
        if [ ! -z "$db_stats" ]; then
            local active_conn=$(echo "$db_stats" | cut -d'|' -f1 | tr -d ' ')
            local total_conn=$(echo "$db_stats" | cut -d'|' -f2 | tr -d ' ')
            local db_size=$(echo "$db_stats" | cut -d'|' -f3 | tr -d ' ')
            
            log_metric "Database: Active connections=${active_conn}, Total=${total_conn}, Size=${db_size} bytes"
            
            db_data=$(echo "$db_data" | jq \
                --arg active "$active_conn" \
                --arg total "$total_conn" \
                --arg size "$db_size" \
                '.database = {
                    "status": "up",
                    "active_connections": $active,
                    "total_connections": $total,
                    "database_size_bytes": $size
                }')
        else
            db_data=$(echo "$db_data" | jq '.database = {"status": "up", "stats": "unavailable"}')
        fi
    else
        log_error "Database is not accessible"
        db_data=$(echo "$db_data" | jq '.database = {"status": "down"}')
    fi
    
    echo "$db_data"
}

# Monitor Redis cache
monitor_redis() {
    local timestamp=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    local redis_data="{\"timestamp\": \"$timestamp\", \"redis\": {}}"
    
    # Check Redis connection
    if docker-compose -f "$PROJECT_ROOT/config/staging/staging_docker-compose.yml" exec -T redis redis-cli ping | grep -q "PONG"; then
        log_success "Redis is accessible"
        
        # Get Redis stats
        local redis_info=$(docker-compose -f "$PROJECT_ROOT/config/staging/staging_docker-compose.yml" exec -T redis \
            redis-cli info memory | grep -E "used_memory:|used_memory_human:|connected_clients:" | head -3)
        
        if [ ! -z "$redis_info" ]; then
            local used_memory=$(echo "$redis_info" | grep "used_memory:" | cut -d: -f2 | tr -d '\r')
            local used_memory_human=$(echo "$redis_info" | grep "used_memory_human:" | cut -d: -f2 | tr -d '\r')
            
            log_metric "Redis: Memory used=${used_memory_human}"
            
            redis_data=$(echo "$redis_data" | jq \
                --arg memory "$used_memory" \
                --arg memory_human "$used_memory_human" \
                '.redis = {
                    "status": "up",
                    "used_memory_bytes": $memory,
                    "used_memory_human": $memory_human
                }')
        else
            redis_data=$(echo "$redis_data" | jq '.redis = {"status": "up", "stats": "unavailable"}')
        fi
    else
        log_error "Redis is not accessible"
        redis_data=$(echo "$redis_data" | jq '.redis = {"status": "down"}')
    fi
    
    echo "$redis_data"
}

# Monitor disk space
monitor_disk_space() {
    local timestamp=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    local disk_data="{\"timestamp\": \"$timestamp\", \"disk\": {}}"
    
    # Check disk usage for project directory
    local disk_info=$(df -h "$PROJECT_ROOT" | tail -1)
    local disk_usage=$(echo "$disk_info" | awk '{print $5}' | sed 's/%//')
    local disk_available=$(echo "$disk_info" | awk '{print $4}')
    local disk_total=$(echo "$disk_info" | awk '{print $2}')
    
    log_metric "Disk: Usage=${disk_usage}%, Available=${disk_available}, Total=${disk_total}"
    
    disk_data=$(echo "$disk_data" | jq \
        --arg usage "$disk_usage" \
        --arg available "$disk_available" \
        --arg total "$disk_total" \
        '.disk = {
            "usage_percent": $usage,
            "available": $available,
            "total": $total
        }')
    
    # Alert on low disk space
    if [ "$disk_usage" -gt 80 ]; then
        log_alert "Low disk space: ${disk_usage}% used (threshold: 80%)"
    fi
    
    echo "$disk_data"
}

# Monitor API endpoints
monitor_api_endpoints() {
    local timestamp=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    local api_data="{\"timestamp\": \"$timestamp\", \"endpoints\": {}}"
    
    local endpoints=(
        "health:/health"
        "metrics:/metrics"
        "docs:/docs"
    )
    
    for endpoint in "${endpoints[@]}"; do
        local name=$(echo "$endpoint" | cut -d: -f1)
        local path=$(echo "$endpoint" | cut -d: -f2)
        local url="http://localhost:8000$path"
        
        local start_time=$(date +%s%N)
        local status_code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 5 "$url")
        local end_time=$(date +%s%N)
        local response_time=$(( (end_time - start_time) / 1000000 ))
        
        if [ "$status_code" = "200" ]; then
            log_success "$name endpoint is responding (${response_time}ms)"
            api_data=$(echo "$api_data" | jq \
                --arg name "$name" \
                --arg status "$status_code" \
                --argjson response_time "$response_time" \
                '.endpoints[$name] = {
                    "status_code": $status,
                    "response_time_ms": $response_time,
                    "status": "healthy"
                }')
        else
            log_error "$name endpoint returned status: $status_code"
            api_data=$(echo "$api_data" | jq \
                --arg name "$name" \
                --arg status "$status_code" \
                --argjson response_time "$response_time" \
                '.endpoints[$name] = {
                    "status_code": $status,
                    "response_time_ms": $response_time,
                    "status": "unhealthy"
                }')
        fi
    done
    
    echo "$api_data"
}

# Collect all metrics
collect_metrics() {
    local timestamp=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    local uptime=$(($(date +%s) - START_TIME))
    
    log_info "Collecting metrics at $timestamp (uptime: ${uptime}s)"
    
    # Collect all monitoring data
    local availability=$(check_service_availability)
    local resources=$(monitor_container_resources)
    local database=$(monitor_database)
    local redis=$(monitor_redis)
    local disk=$(monitor_disk_space)
    local api=$(monitor_api_endpoints)
    
    # Combine all metrics
    local combined_metrics=$(jq -n \
        --argjson availability "$availability" \
        --argjson resources "$resources" \
        --argjson database "$database" \
        --argjson redis "$redis" \
        --argjson disk "$disk" \
        --argjson api "$api" \
        --arg timestamp "$timestamp" \
        --argjson uptime "$uptime" \
        '{
            "timestamp": $timestamp,
            "monitoring_uptime_seconds": $uptime,
            "availability": $availability,
            "resources": $resources,
            "database": $database,
            "redis": $redis,
            "disk": $disk,
            "api": $api
        }')
    
    # Append to metrics file
    echo "$combined_metrics" >> "$METRICS_FILE"
    
    return 0
}

# Generate monitoring report
generate_monitoring_report() {
    local end_time=$(date +%s)
    local total_uptime=$((end_time - START_TIME))
    
    log_info "Generating monitoring report..."
    
    local report_file="$PROJECT_ROOT/staging_monitoring_report_$(date +%Y%m%d_%H%M%S).json"
    
    # Calculate basic statistics from metrics file
    local total_metrics=0
    if [ -f "$METRICS_FILE" ]; then
        total_metrics=$(wc -l < "$METRICS_FILE")
    fi
    
    cat > "$report_file" << EOF
{
  "monitoring_session": {
    "start_time": "$(date -d @$START_TIME -u +%Y-%m-%dT%H:%M:%SZ)",
    "end_time": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "duration_seconds": $total_uptime,
    "metrics_collected": $total_metrics,
    "monitoring_interval": $MONITORING_INTERVAL
  },
  "thresholds": {
    "cpu_alert_threshold": $ALERT_THRESHOLD_CPU,
    "memory_alert_threshold": $ALERT_THRESHOLD_MEMORY,
    "response_time_threshold_ms": $ALERT_THRESHOLD_RESPONSE_TIME,
    "error_rate_threshold_percent": $ALERT_THRESHOLD_ERROR_RATE
  },
  "files_generated": {
    "monitoring_log": "$(basename "$MONITORING_LOG")",
    "alert_log": "$(basename "$ALERT_LOG")",
    "metrics_file": "$(basename "$METRICS_FILE")",
    "report_file": "$(basename "$report_file")"
  },
  "summary": {
    "monitoring_completed": true,
    "alerts_generated": $([ -f "$ALERT_LOG" ] && wc -l < "$ALERT_LOG" || echo 0),
    "data_available": true
  }
}
EOF
    
    log_success "Monitoring report generated: $report_file"
}

# Display dashboard
display_dashboard() {
    clear
    echo -e "${BLUE}=================================================================================${NC}"
    echo -e "${BLUE}                    Agent Forge Staging Environment Monitor${NC}"
    echo -e "${BLUE}=================================================================================${NC}"
    echo -e "${CYAN}Start Time:${NC} $(date -d @$START_TIME)"
    echo -e "${CYAN}Uptime:${NC} $(($(date +%s) - START_TIME)) seconds"
    echo -e "${CYAN}Interval:${NC} ${MONITORING_INTERVAL}s"
    echo -e "${CYAN}Last Check:${NC} $(date)"
    echo -e "${BLUE}=================================================================================${NC}"
    echo ""
}

# Interactive monitoring mode
interactive_monitoring() {
    log_info "Starting interactive monitoring mode..."
    log_info "Press Ctrl+C to stop monitoring"
    
    while [ "$MONITORING_ACTIVE" = true ]; do
        display_dashboard
        collect_metrics
        echo ""
        echo -e "${YELLOW}Waiting ${MONITORING_INTERVAL} seconds for next check...${NC}"
        echo -e "${YELLOW}Press Ctrl+C to stop monitoring${NC}"
        sleep "$MONITORING_INTERVAL"
    done
}

# One-time check mode
oneshot_monitoring() {
    log_info "Performing one-time monitoring check..."
    collect_metrics
    log_success "One-time monitoring check completed"
}

# Show usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --interval SECONDS    Set monitoring interval (default: 30)"
    echo "  --oneshot            Perform single check and exit"
    echo "  --dashboard          Interactive dashboard mode"
    echo "  --help               Show this help message"
    echo ""
    echo "Environment Variables:"
    echo "  MONITORING_INTERVAL          Monitoring interval in seconds"
    echo "  ALERT_THRESHOLD_CPU          CPU usage alert threshold (default: 80%)"
    echo "  ALERT_THRESHOLD_MEMORY       Memory usage alert threshold (default: 85%)"
    echo "  ALERT_THRESHOLD_RESPONSE_TIME Response time alert threshold in ms (default: 2000)"
    echo "  ALERT_THRESHOLD_ERROR_RATE   Error rate alert threshold (default: 5%)"
    echo ""
    echo "Examples:"
    echo "  $0                           # Start continuous monitoring"
    echo "  $0 --interval 60             # Monitor every 60 seconds"
    echo "  $0 --oneshot                 # Single check"
    echo "  $0 --dashboard               # Interactive dashboard"
}

# Main function
main() {
    # Create logs directory
    mkdir -p "$PROJECT_ROOT/logs"
    
    log_info "Starting Agent Forge staging environment monitoring..."
    log_info "Monitoring interval: ${MONITORING_INTERVAL}s"
    log_info "Alert thresholds: CPU=${ALERT_THRESHOLD_CPU}% MEM=${ALERT_THRESHOLD_MEMORY}% RT=${ALERT_THRESHOLD_RESPONSE_TIME}ms"
    
    # Parse command line arguments
    local oneshot=false
    local dashboard=false
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --interval)
                MONITORING_INTERVAL="$2"
                shift 2
                ;;
            --oneshot)
                oneshot=true
                shift
                ;;
            --dashboard)
                dashboard=true
                shift
                ;;
            --help)
                usage
                exit 0
                ;;
            *)
                echo "Unknown option: $1"
                usage
                exit 1
                ;;
        esac
    done
    
    # Execute based on mode
    if [ "$oneshot" = true ]; then
        oneshot_monitoring
    elif [ "$dashboard" = true ]; then
        interactive_monitoring
    else
        # Continuous monitoring mode
        log_info "Starting continuous monitoring mode..."
        log_info "Press Ctrl+C to stop monitoring"
        
        while [ "$MONITORING_ACTIVE" = true ]; do
            collect_metrics
            sleep "$MONITORING_INTERVAL"
        done
    fi
}

# Execute main function
main "$@"