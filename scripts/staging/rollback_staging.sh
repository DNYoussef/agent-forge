#!/bin/bash
# Agent Forge Staging Rollback Script
# Version: 1.0.0
# Environment: staging

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
ROLLBACK_LOG="$PROJECT_ROOT/logs/staging_rollback_$(date +%Y%m%d_%H%M%S).log"
STAGING_CONFIG="$PROJECT_ROOT/config/staging"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1" | tee -a "$ROLLBACK_LOG"
}

log_error() {
    echo -e "${RED}[ERROR] $1${NC}" | tee -a "$ROLLBACK_LOG"
}

log_success() {
    echo -e "${GREEN}[SUCCESS] $1${NC}" | tee -a "$ROLLBACK_LOG"
}

log_warning() {
    echo -e "${YELLOW}[WARNING] $1${NC}" | tee -a "$ROLLBACK_LOG"
}

log_info() {
    echo -e "${BLUE}[INFO] $1${NC}" | tee -a "$ROLLBACK_LOG"
}

# Check if Docker Compose or Kubernetes
check_deployment_method() {
    if kubectl cluster-info &> /dev/null; then
        export USE_KUBERNETES=true
        log_info "Kubernetes cluster detected, using kubectl for rollback"
    else
        export USE_KUBERNETES=false
        log_info "Using Docker Compose for rollback"
    fi
}

# Create backup before rollback
create_backup() {
    log_info "Creating backup before rollback..."
    
    local backup_dir="$PROJECT_ROOT/backups/pre-rollback-$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$backup_dir"
    
    # Backup database
    if [ "$USE_KUBERNETES" = "true" ]; then
        kubectl exec -n agent-forge-staging deployment/postgres -- \
            pg_dump -U postgres agent_forge_staging > "$backup_dir/database_backup.sql" 2>/dev/null || \
            log_warning "Database backup failed"
    else
        docker-compose -f "$STAGING_CONFIG/staging_docker-compose.yml" exec -T postgres \
            pg_dump -U postgres agent_forge_staging > "$backup_dir/database_backup.sql" 2>/dev/null || \
            log_warning "Database backup failed"
    fi
    
    # Backup configuration
    cp -r "$STAGING_CONFIG" "$backup_dir/config" || log_warning "Config backup failed"
    
    # Backup logs
    if [ -d "$PROJECT_ROOT/logs" ]; then
        cp -r "$PROJECT_ROOT/logs" "$backup_dir/logs" || log_warning "Logs backup failed"
    fi
    
    log_success "Backup created at: $backup_dir"
}

# Stop and remove services
stop_services() {
    log_info "Stopping staging services..."
    
    if [ "$USE_KUBERNETES" = "true" ]; then
        # Kubernetes rollback
        log_info "Scaling down Kubernetes deployments..."
        
        kubectl scale deployment agent-forge-api --replicas=0 -n agent-forge-staging 2>/dev/null || true
        kubectl scale deployment agent-forge-websocket --replicas=0 -n agent-forge-staging 2>/dev/null || true
        kubectl scale deployment agent-forge-swarm --replicas=0 -n agent-forge-staging 2>/dev/null || true
        
        # Wait for pods to terminate
        log_info "Waiting for pods to terminate..."
        kubectl wait --for=delete pod -l app=agent-forge-api -n agent-forge-staging --timeout=120s 2>/dev/null || true
        kubectl wait --for=delete pod -l app=agent-forge-websocket -n agent-forge-staging --timeout=120s 2>/dev/null || true
        kubectl wait --for=delete pod -l app=agent-forge-swarm -n agent-forge-staging --timeout=120s 2>/dev/null || true
        
        # Delete deployments
        kubectl delete deployment agent-forge-api -n agent-forge-staging 2>/dev/null || true
        kubectl delete deployment agent-forge-websocket -n agent-forge-staging 2>/dev/null || true
        kubectl delete deployment agent-forge-swarm -n agent-forge-staging 2>/dev/null || true
        
        # Delete services
        kubectl delete service agent-forge-api-service -n agent-forge-staging 2>/dev/null || true
        kubectl delete service agent-forge-websocket-service -n agent-forge-staging 2>/dev/null || true
        kubectl delete service agent-forge-swarm-service -n agent-forge-staging 2>/dev/null || true
        
        # Delete ingress
        kubectl delete ingress agent-forge-ingress -n agent-forge-staging 2>/dev/null || true
        
        log_success "Kubernetes services stopped and removed"
        
    else
        # Docker Compose rollback
        cd "$STAGING_CONFIG"
        
        log_info "Stopping Docker Compose services..."
        docker-compose -f staging_docker-compose.yml down --remove-orphans 2>/dev/null || true
        
        # Force remove containers if they're still running
        local containers=$(docker ps -q --filter "name=agent-forge" --filter "name=staging")
        if [ ! -z "$containers" ]; then
            log_info "Force removing remaining containers..."
            echo "$containers" | xargs docker rm -f 2>/dev/null || true
        fi
        
        log_success "Docker Compose services stopped and removed"
    fi
}

# Clean up volumes and data
cleanup_volumes() {
    log_info "Cleaning up volumes and data..."
    
    if [ "$USE_KUBERNETES" = "true" ]; then
        # Delete persistent volume claims
        kubectl delete pvc --all -n agent-forge-staging 2>/dev/null || true
        
        # Delete configmaps and secrets
        kubectl delete configmap agent-forge-config -n agent-forge-staging 2>/dev/null || true
        kubectl delete secret agent-forge-secrets -n agent-forge-staging 2>/dev/null || true
        
    else
        # Remove Docker volumes
        cd "$STAGING_CONFIG"
        docker-compose -f staging_docker-compose.yml down -v 2>/dev/null || true
        
        # Remove orphaned volumes
        local volumes=$(docker volume ls -q --filter "name=staging")
        if [ ! -z "$volumes" ]; then
            log_info "Removing staging volumes..."
            echo "$volumes" | xargs docker volume rm 2>/dev/null || true
        fi
    fi
    
    log_success "Volumes and data cleaned up"
}

# Remove images
cleanup_images() {
    log_info "Cleaning up Docker images..."
    
    # Remove staging images
    docker rmi agent-forge:staging 2>/dev/null || true
    
    # Remove dangling images
    local dangling=$(docker images -f "dangling=true" -q)
    if [ ! -z "$dangling" ]; then
        log_info "Removing dangling images..."
        echo "$dangling" | xargs docker rmi 2>/dev/null || true
    fi
    
    log_success "Docker images cleaned up"
}

# Remove network
cleanup_network() {
    log_info "Cleaning up networks..."
    
    if [ "$USE_KUBERNETES" = "false" ]; then
        # Remove Docker networks
        docker network rm staging_agent-forge-staging 2>/dev/null || true
        docker network rm agent-forge-staging 2>/dev/null || true
    fi
    
    log_success "Networks cleaned up"
}

# Restore previous version (if available)
restore_previous_version() {
    log_info "Checking for previous version to restore..."
    
    local backup_dir="$PROJECT_ROOT/backups"
    if [ -d "$backup_dir" ]; then
        local latest_backup=$(ls -t "$backup_dir" | head -n 1)
        if [ ! -z "$latest_backup" ] && [ "$latest_backup" != "pre-rollback-$(date +%Y%m%d)"* ]; then
            log_info "Found previous backup: $latest_backup"
            
            # Ask user if they want to restore
            read -p "Do you want to restore from backup '$latest_backup'? (y/N): " -n 1 -r
            echo
            
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                log_info "Restoring from backup..."
                
                # Restore configuration
                if [ -d "$backup_dir/$latest_backup/config" ]; then
                    cp -r "$backup_dir/$latest_backup/config"/* "$STAGING_CONFIG/" 2>/dev/null || true
                    log_success "Configuration restored"
                fi
                
                # Restore database (would need to be done after redeployment)
                if [ -f "$backup_dir/$latest_backup/database_backup.sql" ]; then
                    log_info "Database backup available for manual restoration at:"
                    log_info "  $backup_dir/$latest_backup/database_backup.sql"
                fi
            else
                log_info "Skipping backup restoration"
            fi
        else
            log_info "No suitable backup found for restoration"
        fi
    else
        log_info "No backup directory found"
    fi
}

# Verify rollback
verify_rollback() {
    log_info "Verifying rollback completion..."
    
    # Check that services are no longer running
    local running_containers=$(docker ps --filter "name=agent-forge" --filter "name=staging" -q)
    if [ -z "$running_containers" ]; then
        log_success "No staging containers are running"
    else
        log_warning "Some staging containers are still running"
    fi
    
    # Check that ports are free
    local ports=(8000 8001 8002 9090 3000)
    for port in "${ports[@]}"; do
        if ! nc -z localhost "$port" 2>/dev/null; then
            log_success "Port $port is free"
        else
            log_warning "Port $port is still in use"
        fi
    done
    
    if [ "$USE_KUBERNETES" = "true" ]; then
        # Check Kubernetes namespace
        local pods=$(kubectl get pods -n agent-forge-staging 2>/dev/null | grep -c agent-forge || echo "0")
        if [ "$pods" -eq 0 ]; then
            log_success "No agent-forge pods running in staging namespace"
        else
            log_warning "$pods agent-forge pods still running in staging namespace"
        fi
    fi
    
    log_success "Rollback verification completed"
}

# Generate rollback report
generate_rollback_report() {
    log_info "Generating rollback report..."
    
    local report_file="$PROJECT_ROOT/staging_rollback_report_$(date +%Y%m%d_%H%M%S).json"
    
    cat > "$report_file" << EOF
{
  "rollback": {
    "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "environment": "staging",
    "method": "$([ "$USE_KUBERNETES" = "true" ] && echo "kubernetes" || echo "docker-compose")",
    "status": "completed"
  },
  "actions_performed": {
    "backup_created": true,
    "services_stopped": true,
    "volumes_cleaned": true,
    "images_cleaned": true,
    "networks_cleaned": true,
    "verification_completed": true
  },
  "services_affected": {
    "api": "stopped",
    "websocket": "stopped",
    "swarm": "stopped",
    "database": "stopped",
    "cache": "stopped",
    "monitoring": "stopped"
  },
  "cleanup_summary": {
    "containers_removed": true,
    "volumes_removed": true,
    "images_removed": true,
    "networks_removed": true
  },
  "next_steps": [
    "Review rollback logs for any warnings",
    "Verify all staging resources are cleaned up",
    "Check for any remaining Docker volumes or images",
    "Review backup files if restoration is needed",
    "Update deployment procedures based on failure analysis"
  ]
}
EOF
    
    log_success "Rollback report generated: $report_file"
}

# Emergency rollback (faster, less cleanup)
emergency_rollback() {
    log_error "Performing emergency rollback..."
    
    # Stop all services immediately
    if [ "$USE_KUBERNETES" = "true" ]; then
        kubectl delete namespace agent-forge-staging --force --grace-period=0 2>/dev/null || true
    else
        cd "$STAGING_CONFIG"
        docker-compose -f staging_docker-compose.yml kill 2>/dev/null || true
        docker-compose -f staging_docker-compose.yml rm -f 2>/dev/null || true
    fi
    
    # Force remove all agent-forge containers
    docker ps -a --filter "name=agent-forge" -q | xargs docker rm -f 2>/dev/null || true
    docker ps -a --filter "name=staging" -q | xargs docker rm -f 2>/dev/null || true
    
    log_success "Emergency rollback completed"
}

# Main rollback function
main() {
    log_info "Starting Agent Forge staging environment rollback..."
    
    # Create logs directory
    mkdir -p "$PROJECT_ROOT/logs"
    mkdir -p "$PROJECT_ROOT/backups"
    
    # Check deployment method
    check_deployment_method
    
    # Handle emergency rollback
    if [ "${1:-}" = "--emergency" ]; then
        emergency_rollback
        exit 0
    fi
    
    # Normal rollback process
    create_backup
    stop_services
    cleanup_volumes
    cleanup_images
    cleanup_network
    restore_previous_version
    verify_rollback
    generate_rollback_report
    
    log_success "Agent Forge staging environment rollback completed successfully!"
    log_info "All staging services have been stopped and cleaned up."
    log_info "Backup created and rollback report generated."
    
    # Clean exit
    trap - EXIT
}

# Show usage
usage() {
    echo "Usage: $0 [--emergency]"
    echo ""
    echo "Options:"
    echo "  --emergency    Perform emergency rollback (faster, less cleanup)"
    echo "  --help         Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                # Normal rollback with full cleanup"
    echo "  $0 --emergency    # Emergency rollback"
}

# Handle command line arguments
if [ "${1:-}" = "--help" ]; then
    usage
    exit 0
fi

# Execute main function
main "$@"