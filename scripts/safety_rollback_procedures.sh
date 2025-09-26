#!/bin/bash
# Safety Rollback Procedures for Agent Forge Integration
# Production Validation Specialist - Emergency Recovery Scripts

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
BACKUP_ROOT=".backups"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="$BACKUP_ROOT/$TIMESTAMP"
LOG_FILE="$BACKUP_DIR/rollback.log"

# Critical files to backup/restore
declare -a CRITICAL_FILES=(
    "src/api_server.py"
    "src/api-gateway/index.js"
    "analyzer/bridge.py"
    "src/linter-integration/integration-api.ts"
    "src/ui/components/PhaseController.tsx"
    "src/risk-dashboard/IntegratedRiskDashboard.tsx"
    "package.json"
    "src/compatibility_layer.py"
    "src/linter_manager.py"
)

# Service management
declare -a SERVICES=(
    "python.*api_server"
    "node.*gateway"
    "npm.*start"
)

log() {
    local level="$1"
    shift
    local message="$*"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo -e "${BLUE}[$timestamp]${NC} ${level}: $message" | tee -a "$LOG_FILE"
}

error() {
    log "${RED}ERROR${NC}" "$@"
}

warn() {
    log "${YELLOW}WARN${NC}" "$@"
}

info() {
    log "${GREEN}INFO${NC}" "$@"
}

# Create backup directory
create_backup_structure() {
    info "Creating backup structure at $BACKUP_DIR"
    mkdir -p "$BACKUP_DIR"
    mkdir -p "$BACKUP_DIR/src/api-gateway"
    mkdir -p "$BACKUP_DIR/src/linter-integration"
    mkdir -p "$BACKUP_DIR/src/ui/components"
    mkdir -p "$BACKUP_DIR/src/risk-dashboard"
    mkdir -p "$BACKUP_DIR/analyzer"

    # Create backup metadata
    cat > "$BACKUP_DIR/metadata.json" << EOF
{
  "timestamp": "$TIMESTAMP",
  "backup_type": "pre_modification_safety",
  "created_by": "safety_rollback_procedures.sh",
  "git_commit": "$(git rev-parse HEAD 2>/dev/null || echo 'unknown')",
  "system_info": {
    "hostname": "$(hostname)",
    "user": "$(whoami)",
    "pwd": "$(pwd)"
  }
}
EOF

    info "Backup structure created successfully"
}

# Stop all services safely
stop_services() {
    info "Stopping all services..."

    for service_pattern in "${SERVICES[@]}"; do
        info "Stopping service pattern: $service_pattern"
        pkill -f "$service_pattern" 2>/dev/null || warn "No processes found for: $service_pattern"
        sleep 1
    done

    # Wait for graceful shutdown
    sleep 3

    # Force kill if necessary
    for service_pattern in "${SERVICES[@]}"; do
        if pgrep -f "$service_pattern" > /dev/null; then
            warn "Force killing stubborn process: $service_pattern"
            pkill -9 -f "$service_pattern" 2>/dev/null || true
        fi
    done

    info "All services stopped"
}

# Start services in correct order
start_services() {
    info "Starting services in dependency order..."

    # Start Python API server first
    info "Starting Python API server on port 8000..."
    nohup python src/api_server.py > "$BACKUP_DIR/api_server.log" 2>&1 &
    local api_pid=$!

    # Wait for API server to be ready
    local max_wait=30
    local wait_count=0
    while ! curl -s http://localhost:8000/api/health > /dev/null 2>&1; do
        sleep 1
        wait_count=$((wait_count + 1))
        if [ $wait_count -ge $max_wait ]; then
            error "API server failed to start within ${max_wait} seconds"
            return 1
        fi
    done
    info "Python API server started successfully (PID: $api_pid)"

    # Start Express gateway
    info "Starting Express gateway on port 3000..."
    nohup npm run start > "$BACKUP_DIR/gateway.log" 2>&1 &
    local gateway_pid=$!

    # Wait for gateway to be ready
    wait_count=0
    while ! curl -s http://localhost:3000/health > /dev/null 2>&1; do
        sleep 1
        wait_count=$((wait_count + 1))
        if [ $wait_count -ge $max_wait ]; then
            error "Gateway failed to start within ${max_wait} seconds"
            return 1
        fi
    done
    info "Express gateway started successfully (PID: $gateway_pid)"

    # Save PIDs for monitoring
    echo "$api_pid" > "$BACKUP_DIR/api_server.pid"
    echo "$gateway_pid" > "$BACKUP_DIR/gateway.pid"

    info "All services started successfully"
}

# Backup critical files
backup_files() {
    info "Backing up critical files..."

    for file in "${CRITICAL_FILES[@]}"; do
        if [ -f "$file" ]; then
            local dest_dir="$BACKUP_DIR/$(dirname "$file")"
            mkdir -p "$dest_dir"
            cp "$file" "$dest_dir/"
            info "Backed up: $file"

            # Calculate file hash for integrity verification
            local hash=$(sha256sum "$file" | cut -d' ' -f1)
            echo "$hash $file" >> "$BACKUP_DIR/file_hashes.txt"
        else
            warn "File not found for backup: $file"
        fi
    done

    # Backup package-lock.json if it exists
    if [ -f "package-lock.json" ]; then
        cp "package-lock.json" "$BACKUP_DIR/"
        info "Backed up: package-lock.json"
    fi

    # Backup environment configuration
    if [ -f ".env" ]; then
        cp ".env" "$BACKUP_DIR/"
        info "Backed up: .env"
    fi

    info "File backup completed"
}

# Verify file integrity
verify_backup_integrity() {
    info "Verifying backup integrity..."

    if [ ! -f "$BACKUP_DIR/file_hashes.txt" ]; then
        error "Hash file not found - cannot verify integrity"
        return 1
    fi

    while IFS=' ' read -r hash file; do
        if [ -f "$BACKUP_DIR/$file" ]; then
            local backup_hash=$(sha256sum "$BACKUP_DIR/$file" | cut -d' ' -f1)
            if [ "$hash" = "$backup_hash" ]; then
                info "✓ Integrity verified: $file"
            else
                error "✗ Integrity check failed: $file"
                return 1
            fi
        else
            warn "Backup file missing: $file"
        fi
    done < "$BACKUP_DIR/file_hashes.txt"

    info "Backup integrity verification completed"
}

# Restore files from backup
restore_files() {
    local backup_to_restore="$1"

    if [ ! -d "$backup_to_restore" ]; then
        error "Backup directory not found: $backup_to_restore"
        return 1
    fi

    info "Restoring files from backup: $backup_to_restore"

    # Stop services before restore
    stop_services

    # Restore each critical file
    for file in "${CRITICAL_FILES[@]}"; do
        local backup_file="$backup_to_restore/$file"
        if [ -f "$backup_file" ]; then
            # Create destination directory if needed
            mkdir -p "$(dirname "$file")"

            # Backup current file before restore (just in case)
            if [ -f "$file" ]; then
                cp "$file" "$file.pre_restore_$(date +%H%M%S)" 2>/dev/null || true
            fi

            # Restore file
            cp "$backup_file" "$file"
            info "Restored: $file"
        else
            warn "Backup file not found: $backup_file"
        fi
    done

    # Restore package files
    if [ -f "$backup_to_restore/package-lock.json" ]; then
        cp "$backup_to_restore/package-lock.json" .
        info "Restored: package-lock.json"
    fi

    if [ -f "$backup_to_restore/.env" ]; then
        cp "$backup_to_restore/.env" .
        info "Restored: .env"
    fi

    info "File restoration completed"
}

# Test system functionality
test_system_functionality() {
    info "Testing system functionality..."

    local test_results="$BACKUP_DIR/test_results.json"
    local all_tests_passed=true

    cat > "$test_results" << 'EOF'
{
  "timestamp": "",
  "tests": {
    "api_server_health": false,
    "gateway_health": false,
    "python_bridge": false,
    "websocket": false
  },
  "overall_status": "unknown"
}
EOF

    # Update timestamp
    sed -i "s/\"timestamp\": \"\"/\"timestamp\": \"$(date -Iseconds)\"/" "$test_results"

    # Test 1: API Server Health
    info "Testing API server health..."
    if curl -s -f http://localhost:8000/api/health > /dev/null; then
        info "✓ API server health check passed"
        sed -i 's/"api_server_health": false/"api_server_health": true/' "$test_results"
    else
        error "✗ API server health check failed"
        all_tests_passed=false
    fi

    # Test 2: Gateway Health
    info "Testing gateway health..."
    if curl -s -f http://localhost:3000/health > /dev/null; then
        info "✓ Gateway health check passed"
        sed -i 's/"gateway_health": false/"gateway_health": true/' "$test_results"
    else
        error "✗ Gateway health check failed"
        all_tests_passed=false
    fi

    # Test 3: Python Bridge
    info "Testing Python bridge..."
    local bridge_test=$(curl -s -X POST http://localhost:3000/api/analyzer/connascence_scan \
                       -H "Content-Type: application/json" \
                       -d '{"path": ".", "depth": 1}' | jq '.success' 2>/dev/null || echo "false")

    if [ "$bridge_test" = "true" ]; then
        info "✓ Python bridge test passed"
        sed -i 's/"python_bridge": false/"python_bridge": true/' "$test_results"
    else
        error "✗ Python bridge test failed"
        all_tests_passed=false
    fi

    # Test 4: WebSocket connectivity (basic)
    info "Testing WebSocket connectivity..."
    if curl -s -H "Upgrade: websocket" -H "Connection: Upgrade" \
            http://localhost:3000 2>/dev/null | grep -q "websocket"; then
        info "✓ WebSocket test passed"
        sed -i 's/"websocket": false/"websocket": true/' "$test_results"
    else
        warn "WebSocket test inconclusive (requires full client test)"
        # Don't fail overall test for this
    fi

    # Update overall status
    if [ "$all_tests_passed" = true ]; then
        sed -i 's/"overall_status": "unknown"/"overall_status": "passed"/' "$test_results"
        info "✓ All critical tests passed"
    else
        sed -i 's/"overall_status": "unknown"/"overall_status": "failed"/' "$test_results"
        error "✗ Some critical tests failed"
    fi

    info "System functionality test completed"
    return $([ "$all_tests_passed" = true ] && echo 0 || echo 1)
}

# Create pre-modification backup
create_pre_modification_backup() {
    info "=== CREATING PRE-MODIFICATION BACKUP ==="

    create_backup_structure
    backup_files
    verify_backup_integrity

    # Test current system before any changes
    test_system_functionality

    # Create restore script
    cat > "$BACKUP_DIR/restore.sh" << EOF
#!/bin/bash
# Auto-generated restore script
# Created: $TIMESTAMP

cd "\$(dirname "\$0")/../.."
bash scripts/safety_rollback_procedures.sh restore "$BACKUP_DIR"
EOF

    chmod +x "$BACKUP_DIR/restore.sh"

    info "✓ Pre-modification backup created: $BACKUP_DIR"
    info "✓ To restore: bash $BACKUP_DIR/restore.sh"

    return 0
}

# Emergency rollback function
emergency_rollback() {
    local backup_dir="$1"

    error "=== EMERGENCY ROLLBACK INITIATED ==="
    warn "This will restore the system to a previous state"
    warn "Backup directory: $backup_dir"

    if [ ! -d "$backup_dir" ]; then
        error "Backup directory not found: $backup_dir"
        return 1
    fi

    # Stop all services
    stop_services

    # Restore files
    restore_files "$backup_dir"

    # Start services
    start_services

    # Test restored system
    if test_system_functionality; then
        info "✓ Emergency rollback completed successfully"
        info "✓ System functionality verified"
        return 0
    else
        error "✗ Emergency rollback completed but tests failed"
        error "Manual intervention required"
        return 1
    fi
}

# List available backups
list_backups() {
    info "Available backups in $BACKUP_ROOT:"

    if [ ! -d "$BACKUP_ROOT" ]; then
        warn "No backup directory found"
        return 1
    fi

    for backup in "$BACKUP_ROOT"/*; do
        if [ -d "$backup" ] && [ -f "$backup/metadata.json" ]; then
            local timestamp=$(basename "$backup")
            local git_commit=$(jq -r '.git_commit' "$backup/metadata.json" 2>/dev/null || echo "unknown")
            local created_by=$(jq -r '.created_by' "$backup/metadata.json" 2>/dev/null || echo "unknown")

            info "  $timestamp (commit: ${git_commit:0:7}, created by: $created_by)"

            # Show test results if available
            if [ -f "$backup/test_results.json" ]; then
                local test_status=$(jq -r '.overall_status' "$backup/test_results.json")
                info "    └─ Tests: $test_status"
            fi
        fi
    done
}

# Cleanup old backups (keep last 10)
cleanup_old_backups() {
    info "Cleaning up old backups (keeping last 10)..."

    if [ ! -d "$BACKUP_ROOT" ]; then
        warn "No backup directory found"
        return 0
    fi

    local backup_count=$(find "$BACKUP_ROOT" -maxdepth 1 -type d | wc -l)
    if [ "$backup_count" -le 11 ]; then  # 10 + 1 for the root dir
        info "No cleanup needed ($((backup_count - 1)) backups)"
        return 0
    fi

    # Remove oldest backups (keep newest 10)
    find "$BACKUP_ROOT" -maxdepth 1 -type d -name "20*" | sort | head -n -10 | while read -r old_backup; do
        warn "Removing old backup: $old_backup"
        rm -rf "$old_backup"
    done

    info "Backup cleanup completed"
}

# Main command dispatcher
main() {
    local command="${1:-help}"

    case "$command" in
        "backup")
            create_pre_modification_backup
            ;;
        "restore")
            if [ -z "${2:-}" ]; then
                error "Usage: $0 restore <backup_directory>"
                exit 1
            fi
            emergency_rollback "$2"
            ;;
        "list")
            list_backups
            ;;
        "test")
            test_system_functionality
            ;;
        "cleanup")
            cleanup_old_backups
            ;;
        "stop")
            stop_services
            ;;
        "start")
            start_services
            ;;
        "help"|*)
            cat << EOF
Safety Rollback Procedures - Agent Forge Integration

Usage: $0 <command> [options]

Commands:
  backup          Create pre-modification backup of critical files
  restore <dir>   Restore system from specified backup directory
  list            List available backups
  test            Test current system functionality
  cleanup         Remove old backups (keep last 10)
  stop            Stop all services
  start           Start all services
  help            Show this help message

Examples:
  $0 backup                               # Create backup before changes
  $0 restore .backups/20250925_143022     # Restore from specific backup
  $0 list                                 # Show all available backups
  $0 test                                 # Test current system health

Safety Notes:
- Always create backup before modifications
- Test system functionality after any changes
- Keep rollback procedures readily available
- Monitor logs in backup directories

EOF
            ;;
    esac
}

# Execute main function with all arguments
main "$@"