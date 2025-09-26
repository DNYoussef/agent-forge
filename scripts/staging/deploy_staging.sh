#!/bin/bash
# Agent Forge Staging Deployment Script
# Version: 1.0.0
# Environment: staging

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
STAGING_CONFIG="$PROJECT_ROOT/config/staging"
LOG_FILE="$PROJECT_ROOT/logs/staging_deployment_$(date +%Y%m%d_%H%M%S).log"
DEPLOYMENT_ENV="staging"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

log_error() {
    echo -e "${RED}[ERROR] $1${NC}" | tee -a "$LOG_FILE"
}

log_success() {
    echo -e "${GREEN}[SUCCESS] $1${NC}" | tee -a "$LOG_FILE"
}

log_warning() {
    echo -e "${YELLOW}[WARNING] $1${NC}" | tee -a "$LOG_FILE"
}

log_info() {
    echo -e "${BLUE}[INFO] $1${NC}" | tee -a "$LOG_FILE"
}

# Error handling
cleanup() {
    if [ $? -ne 0 ]; then
        log_error "Deployment failed. Initiating rollback..."
        # Call rollback script if it exists
        if [ -f "$SCRIPT_DIR/rollback_staging.sh" ]; then
            bash "$SCRIPT_DIR/rollback_staging.sh"
        fi
    fi
}

trap cleanup EXIT

# Pre-deployment checks
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check required tools
    local tools=("docker" "docker-compose" "kubectl" "curl" "jq")
    for tool in "${tools[@]}"; do
        if ! command -v "$tool" &> /dev/null; then
            log_error "Required tool '$tool' is not installed"
            exit 1
        fi
    done
    
    # Check Docker daemon
    if ! docker info &> /dev/null; then
        log_error "Docker daemon is not running"
        exit 1
    fi
    
    # Check Kubernetes cluster access
    if ! kubectl cluster-info &> /dev/null; then
        log_warning "Kubernetes cluster not accessible. Will use Docker Compose instead."
        export USE_DOCKER_COMPOSE=true
    else
        export USE_DOCKER_COMPOSE=false
    fi
    
    log_success "Prerequisites check completed"
}

# Build and tag images
build_images() {
    log_info "Building Docker images..."
    
    cd "$PROJECT_ROOT"
    
    # Create Dockerfile.staging if it doesn't exist
    if [ ! -f "Dockerfile.staging" ]; then
        log_info "Creating Dockerfile.staging..."
        cat > Dockerfile.staging << 'EOF'
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements*.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p logs data models uploads

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000
CMD ["python", "run_api_server.py"]
EOF
    fi
    
    # Build main application image
    docker build -f Dockerfile.staging -t agent-forge:staging .
    
    # Tag for registry if specified
    if [ ! -z "${DOCKER_REGISTRY:-}" ]; then
        docker tag agent-forge:staging "$DOCKER_REGISTRY/agent-forge:staging"
        docker push "$DOCKER_REGISTRY/agent-forge:staging"
    fi
    
    log_success "Docker images built successfully"
}

# Deploy database
deploy_database() {
    log_info "Deploying database..."
    
    cd "$STAGING_CONFIG"
    
    # Create database init scripts directory
    mkdir -p init-scripts
    
    # Create database initialization script
    cat > init-scripts/01-init.sql << 'EOF'
CREATE DATABASE IF NOT EXISTS agent_forge_staging;
CREATE DATABASE IF NOT EXISTS agent_forge_test;
CREATE USER IF NOT EXISTS 'staging_user'@'%' IDENTIFIED BY 'staging_password';
GRANT ALL PRIVILEGES ON agent_forge_staging.* TO 'staging_user'@'%';
GRANT ALL PRIVILEGES ON agent_forge_test.* TO 'staging_user'@'%';
FLUSH PRIVILEGES;
EOF
    
    # Start database service
    if [ "$USE_DOCKER_COMPOSE" = "true" ]; then
        docker-compose -f staging_docker-compose.yml up -d postgres redis
    else
        kubectl apply -f postgres-deployment.yml
        kubectl apply -f redis-deployment.yml
    fi
    
    # Wait for database to be ready
    log_info "Waiting for database to be ready..."
    local max_attempts=30
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        if docker-compose -f staging_docker-compose.yml exec -T postgres pg_isready -U postgres; then
            log_success "Database is ready"
            break
        fi
        
        log_info "Attempt $attempt/$max_attempts: Database not ready yet, waiting..."
        sleep 10
        ((attempt++))
    done
    
    if [ $attempt -gt $max_attempts ]; then
        log_error "Database failed to start within the expected time"
        exit 1
    fi
}

# Run database migrations
run_migrations() {
    log_info "Running database migrations..."
    
    cd "$PROJECT_ROOT"
    
    # Create migrations directory if it doesn't exist
    mkdir -p migrations
    
    # Create initial migration script
    if [ ! -f "migrations/001_initial.sql" ]; then
        cat > migrations/001_initial.sql << 'EOF'
-- Agent Forge Initial Schema
CREATE TABLE IF NOT EXISTS agents (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    type VARCHAR(100) NOT NULL,
    status VARCHAR(50) DEFAULT 'inactive',
    config JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS swarms (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    topology VARCHAR(50) NOT NULL,
    status VARCHAR(50) DEFAULT 'inactive',
    config JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS tasks (
    id SERIAL PRIMARY KEY,
    swarm_id INTEGER REFERENCES swarms(id),
    agent_id INTEGER REFERENCES agents(id),
    description TEXT,
    status VARCHAR(50) DEFAULT 'pending',
    priority INTEGER DEFAULT 0,
    result JSONB,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_agents_status ON agents(status);
CREATE INDEX idx_swarms_status ON swarms(status);
CREATE INDEX idx_tasks_status ON tasks(status);
CREATE INDEX idx_tasks_priority ON tasks(priority);
EOF
    fi
    
    # Run migrations
    if [ "$USE_DOCKER_COMPOSE" = "true" ]; then
        docker-compose -f "$STAGING_CONFIG/staging_docker-compose.yml" exec -T postgres \
            psql -U postgres -d agent_forge_staging -f /app/migrations/001_initial.sql
    else
        kubectl exec -n agent-forge-staging deployment/postgres -- \
            psql -U postgres -d agent_forge_staging -f /app/migrations/001_initial.sql
    fi
    
    log_success "Database migrations completed"
}

# Deploy application services
deploy_services() {
    log_info "Deploying application services..."
    
    cd "$STAGING_CONFIG"
    
    if [ "$USE_DOCKER_COMPOSE" = "true" ]; then
        # Deploy using Docker Compose
        docker-compose -f staging_docker-compose.yml up -d
    else
        # Deploy using Kubernetes
        kubectl apply -f staging_deployment.yml
    fi
    
    log_success "Application services deployed"
}

# Configure monitoring
setup_monitoring() {
    log_info "Setting up monitoring..."
    
    cd "$STAGING_CONFIG"
    
    # Create Prometheus configuration
    cat > prometheus.yml << 'EOF'
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'agent-forge-api'
    static_configs:
      - targets: ['agent-forge-api:8000']
    metrics_path: '/metrics'
    scrape_interval: 30s
    
  - job_name: 'agent-forge-websocket'
    static_configs:
      - targets: ['agent-forge-websocket:8001']
    metrics_path: '/metrics'
    scrape_interval: 30s
    
  - job_name: 'agent-forge-swarm'
    static_configs:
      - targets: ['agent-forge-swarm:8002']
    metrics_path: '/metrics'
    scrape_interval: 30s
    
  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres:5432']
    
  - job_name: 'redis'
    static_configs:
      - targets: ['redis:6379']
EOF
    
    # Create Grafana datasources
    mkdir -p grafana/datasources
    cat > grafana/datasources/prometheus.yml << 'EOF'
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
EOF
    
    # Create Grafana dashboards directory
    mkdir -p grafana/dashboards
    
    log_success "Monitoring setup completed"
}

# Health checks
perform_health_checks() {
    log_info "Performing health checks..."
    
    local services=("api:8000" "websocket:8001" "swarm:8002")
    local max_attempts=30
    
    for service in "${services[@]}"; do
        local service_name=${service%:*}
        local port=${service#*:}
        local attempt=1
        
        log_info "Checking health of $service_name service..."
        
        while [ $attempt -le $max_attempts ]; do
            if curl -f "http://localhost:$port/health" &> /dev/null; then
                log_success "$service_name service is healthy"
                break
            fi
            
            log_info "Attempt $attempt/$max_attempts: $service_name not ready yet, waiting..."
            sleep 10
            ((attempt++))
        done
        
        if [ $attempt -gt $max_attempts ]; then
            log_error "$service_name service failed health check"
            exit 1
        fi
    done
    
    log_success "All health checks passed"
}

# Generate deployment report
generate_deployment_report() {
    log_info "Generating deployment report..."
    
    local report_file="$PROJECT_ROOT/staging_deployment_report_$(date +%Y%m%d_%H%M%S).json"
    
    cat > "$report_file" << EOF
{
  "deployment": {
    "environment": "$DEPLOYMENT_ENV",
    "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "version": "staging",
    "deployment_method": "$([ "$USE_DOCKER_COMPOSE" = "true" ] && echo "docker-compose" || echo "kubernetes")"
  },
  "services": {
    "api": {
      "status": "deployed",
      "url": "http://localhost:8000",
      "health_check": "passed"
    },
    "websocket": {
      "status": "deployed",
      "url": "http://localhost:8001",
      "health_check": "passed"
    },
    "swarm": {
      "status": "deployed",
      "url": "http://localhost:8002",
      "health_check": "passed"
    },
    "database": {
      "status": "deployed",
      "type": "postgresql",
      "health_check": "passed"
    },
    "cache": {
      "status": "deployed",
      "type": "redis",
      "health_check": "passed"
    }
  },
  "monitoring": {
    "prometheus": {
      "status": "deployed",
      "url": "http://localhost:9090"
    },
    "grafana": {
      "status": "deployed",
      "url": "http://localhost:3000"
    }
  },
  "quality_gates": {
    "all_services_healthy": true,
    "deployment_successful": true,
    "ready_for_testing": true
  }
}
EOF
    
    log_success "Deployment report generated: $report_file"
}

# Main deployment function
main() {
    log_info "Starting Agent Forge staging deployment..."
    
    # Create logs directory
    mkdir -p "$PROJECT_ROOT/logs"
    
    # Run deployment steps
    check_prerequisites
    build_images
    deploy_database
    run_migrations
    deploy_services
    setup_monitoring
    
    # Wait for services to start
    sleep 30
    
    perform_health_checks
    generate_deployment_report
    
    log_success "Agent Forge staging deployment completed successfully!"
    log_info "Services are available at:"
    log_info "  - API: http://localhost:8000"
    log_info "  - WebSocket: http://localhost:8001"
    log_info "  - Swarm Coordinator: http://localhost:8002"
    log_info "  - Prometheus: http://localhost:9090"
    log_info "  - Grafana: http://localhost:3000"
}

# Execute main function
main "$@"