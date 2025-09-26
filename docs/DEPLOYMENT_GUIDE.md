# Agent Forge Deployment Guide

## Prerequisites

### System Requirements

**Minimum Requirements**:
- Python 3.9+
- 8GB RAM
- 4GB GPU memory (optional but recommended)
- 20GB disk space

**Recommended Requirements**:
- Python 3.10+
- 16GB+ RAM
- 8GB+ GPU memory (CUDA-compatible)
- 50GB+ disk space
- SSD storage for model caching

### Software Dependencies

```bash
# Core dependencies
pip install torch>=2.0.0 torchvision torchaudio
pip install transformers>=4.30.0
pip install datasets>=2.12.0
pip install numpy>=1.24.0
pip install tqdm>=4.65.0

# Web interface dependencies
pip install fastapi>=0.100.0
pip install uvicorn>=0.22.0
pip install websockets>=11.0
pip install jinja2>=3.1.0

# Optional dependencies for enhanced features
pip install wandb                    # For experiment tracking
pip install tensorboard             # For visualization
pip install accelerate              # For model acceleration
pip install bitsandbytes            # For 8-bit optimization
```

### Node.js Dependencies (for Web Dashboard)

```bash
# Install Node.js 18+ and npm
npm install -g npm@latest

# Dashboard dependencies (run in src/web/dashboard/)
npm install next@13.4.19
npm install react@18.2.0
npm install react-dom@18.2.0
npm install @types/node
npm install @types/react
npm install @types/react-dom
npm install typescript
```

## Installation Methods

### Method 1: Direct Installation

```bash
# Clone repository
git clone https://github.com/your-org/agent-forge.git
cd agent-forge

# Install Python dependencies
pip install -r requirements.txt

# Install development dependencies (optional)
pip install -r requirements-dev.txt

# Install web dashboard dependencies
cd src/web/dashboard
npm install
cd ../../..

# Verify installation
python -c "import agent_forge; print('Agent Forge installed successfully')"
```

### Method 2: Virtual Environment

```bash
# Create virtual environment
python -m venv agent-forge-env

# Activate environment
# On Windows:
agent-forge-env\Scripts\activate
# On macOS/Linux:
source agent-forge-env/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Set up environment variables
export AGENT_FORGE_HOME=$(pwd)
export PYTHONPATH="${PYTHONPATH}:${AGENT_FORGE_HOME}"
```

### Method 3: Docker Deployment

```dockerfile
# Dockerfile
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js
RUN curl -fsSL https://deb.nodesource.com/setup_18.x | bash - \
    && apt-get install -y nodejs

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Install web dashboard dependencies
WORKDIR /app/src/web/dashboard
RUN npm install && npm run build

# Set working directory back to app root
WORKDIR /app

# Expose ports
EXPOSE 8000 3000

# Start command
CMD ["python", "src/api/pipeline_server_fixed.py"]
```

```bash
# Build and run Docker container
docker build -t agent-forge .
docker run -p 8000:8000 -p 3000:3000 -v $(pwd)/data:/app/data agent-forge
```

## Configuration

### Environment Configuration

Create `.env` file in the project root:

```bash
# Agent Forge Configuration
AGENT_FORGE_HOME=/path/to/agent-forge
AGENT_FORGE_DATA_DIR=/path/to/data
AGENT_FORGE_CACHE_DIR=/path/to/cache
AGENT_FORGE_LOG_LEVEL=INFO

# Model Configuration
TRANSFORMERS_CACHE=/path/to/transformers/cache
HF_HOME=/path/to/huggingface/cache
TORCH_HOME=/path/to/torch/cache

# Hardware Configuration
CUDA_VISIBLE_DEVICES=0
OMP_NUM_THREADS=4

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
WEB_PORT=3000

# Security Configuration
API_SECRET_KEY=your-secret-key-here
ENABLE_CORS=true
ALLOWED_ORIGINS=http://localhost:3000

# Optional: Weights & Biases
WANDB_PROJECT=agent-forge
WANDB_ENTITY=your-wandb-entity
```

### Directory Structure Setup

```bash
# Create required directories
mkdir -p data/{models,outputs,cache,logs,checkpoints}
mkdir -p config/{phases,pipelines}
mkdir -p workspace/{experiments,results}

# Set permissions
chmod -R 755 data/
chmod -R 755 config/
chmod -R 755 workspace/
```

### Phase Configuration Files

#### EvoMerge Configuration

```yaml
# config/phases/evomerge.yaml
evomerge:
  population_size: 10
  generations: 50
  mutation_rate: 0.1
  crossover_rate: 0.8
  elite_size: 2

  selection:
    strategy: "tournament"
    tournament_size: 3

  merge_operators:
    - name: "slerp"
      weight: 0.4
    - name: "ties"
      weight: 0.4
    - name: "dare"
      weight: 0.2

  fitness:
    functions: ["accuracy", "efficiency", "diversity"]
    weights: [0.5, 0.3, 0.2]

  convergence:
    threshold: 0.001
    patience: 10
    max_generations: 100
```

#### Quiet-STaR Configuration

```yaml
# config/phases/quietstar.yaml
quietstar:
  num_thoughts: 4
  thought_length: 32
  coherence_threshold: 0.6
  temperature: 0.8
  top_p: 0.9

  special_tokens:
    start_thought: "<|startofthought|>"
    end_thought: "<|endofthought|>"
    thought_sep: "<|thoughtsep|>"

  coherence_weights:
    semantic_similarity: 0.3
    logical_consistency: 0.3
    relevance_score: 0.25
    fluency_score: 0.15
```

#### BitNet Configuration

```yaml
# config/phases/bitnet.yaml
bitnet:
  quantization_bits: 1.58
  preserve_embedding_precision: true
  preserve_output_precision: true
  sparsity_threshold: 0.1

  calibration:
    samples: 1000
    dataset: "openwebtext"
    batch_size: 4
    sequence_length: 512

  fine_tuning:
    enabled: true
    epochs: 2
    learning_rate: 1e-5
    warmup_steps: 50
    weight_decay: 0.01

  grokfast:
    enabled: true
    ema_alpha: 0.98
    lambda: 2.0

  targets:
    compression_ratio: 8.0
    max_accuracy_drop: 0.05
```

## Deployment Scenarios

### Local Development

```bash
# Start API server
python src/api/pipeline_server_fixed.py

# Start web dashboard (in separate terminal)
cd src/web/dashboard
npm run dev

# Access dashboard at http://localhost:3000
# API available at http://localhost:8000
```

### Production Deployment

#### Using systemd (Linux)

```ini
# /etc/systemd/system/agent-forge-api.service
[Unit]
Description=Agent Forge API Server
After=network.target

[Service]
Type=simple
User=agent-forge
Group=agent-forge
WorkingDirectory=/opt/agent-forge
Environment=PATH=/opt/agent-forge/venv/bin
ExecStart=/opt/agent-forge/venv/bin/python src/api/pipeline_server_fixed.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```ini
# /etc/systemd/system/agent-forge-web.service
[Unit]
Description=Agent Forge Web Dashboard
After=network.target agent-forge-api.service

[Service]
Type=simple
User=agent-forge
Group=agent-forge
WorkingDirectory=/opt/agent-forge/src/web/dashboard
Environment=PATH=/usr/bin:/bin
Environment=NODE_ENV=production
ExecStart=/usr/bin/npm start
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
# Enable and start services
sudo systemctl daemon-reload
sudo systemctl enable agent-forge-api
sudo systemctl enable agent-forge-web
sudo systemctl start agent-forge-api
sudo systemctl start agent-forge-web

# Check status
sudo systemctl status agent-forge-api
sudo systemctl status agent-forge-web
```

#### Using Docker Compose

```yaml
# docker-compose.yml
version: '3.8'

services:
  agent-forge-api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./data:/app/data
      - ./config:/app/config
      - ./workspace:/app/workspace
    environment:
      - AGENT_FORGE_DATA_DIR=/app/data
      - AGENT_FORGE_CACHE_DIR=/app/data/cache
      - API_HOST=0.0.0.0
      - API_PORT=8000
    restart: unless-stopped

  agent-forge-web:
    build:
      context: .
      dockerfile: Dockerfile.web
    ports:
      - "3000:3000"
    environment:
      - NODE_ENV=production
      - NEXT_PUBLIC_API_URL=http://agent-forge-api:8000
    depends_on:
      - agent-forge-api
    restart: unless-stopped

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - agent-forge-api
      - agent-forge-web
    restart: unless-stopped
```

```bash
# Deploy with Docker Compose
docker-compose up -d

# View logs
docker-compose logs -f

# Scale services
docker-compose up -d --scale agent-forge-api=3
```

### Cloud Deployment

#### AWS Deployment

```yaml
# aws-deploy.yml (AWS ECS Task Definition)
family: agent-forge
networkMode: awsvpc
requiresCompatibilities:
  - FARGATE
cpu: 2048
memory: 4096

taskRoleArn: arn:aws:iam::account:role/agent-forge-task-role
executionRoleArn: arn:aws:iam::account:role/agent-forge-execution-role

containerDefinitions:
  - name: agent-forge-api
    image: your-registry/agent-forge:latest
    portMappings:
      - containerPort: 8000
        protocol: tcp
    environment:
      - name: AWS_REGION
        value: us-west-2
      - name: AGENT_FORGE_DATA_DIR
        value: /app/data
    logConfiguration:
      logDriver: awslogs
      options:
        awslogs-group: /ecs/agent-forge
        awslogs-region: us-west-2
        awslogs-stream-prefix: api
```

#### Kubernetes Deployment

```yaml
# k8s-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: agent-forge
  labels:
    app: agent-forge
spec:
  replicas: 3
  selector:
    matchLabels:
      app: agent-forge
  template:
    metadata:
      labels:
        app: agent-forge
    spec:
      containers:
      - name: agent-forge-api
        image: agent-forge:latest
        ports:
        - containerPort: 8000
        env:
        - name: API_HOST
          value: "0.0.0.0"
        - name: API_PORT
          value: "8000"
        resources:
          requests:
            memory: "2Gi"
            cpu: "1"
          limits:
            memory: "4Gi"
            cpu: "2"
        volumeMounts:
        - name: data-volume
          mountPath: /app/data
      volumes:
      - name: data-volume
        persistentVolumeClaim:
          claimName: agent-forge-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: agent-forge-service
spec:
  selector:
    app: agent-forge
  ports:
    - protocol: TCP
      port: 80
      targetPort: 8000
  type: LoadBalancer
```

## Performance Optimization

### Hardware Optimization

```python
# config/hardware.py
import torch

# GPU optimization
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# CPU optimization
torch.set_num_threads(4)  # Adjust based on CPU cores

# Memory optimization
torch.backends.cuda.max_split_size_mb = 512
```

### Model Caching Strategy

```bash
# Pre-download models to cache
python scripts/cache_models.py \
  --models microsoft/DialoGPT-medium \
           microsoft/DialoGPT-large \
           facebook/opt-350m \
  --cache-dir /app/data/cache
```

### Database Optimization (Optional)

```yaml
# config/database.yaml
database:
  type: "sqlite"  # or "postgresql" for production
  path: "data/agent_forge.db"

  # PostgreSQL settings (if used)
  host: "localhost"
  port: 5432
  name: "agent_forge"
  user: "agent_forge"
  password: "secure_password"

  # Connection pooling
  pool_size: 10
  max_overflow: 20
```

## Monitoring and Logging

### Logging Configuration

```python
# config/logging.py
import logging
import logging.handlers

LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        },
        'detailed': {
            'format': '%(asctime)s [%(levelname)s] %(name)s:%(lineno)d: %(message)s'
        }
    },
    'handlers': {
        'console': {
            'level': 'INFO',
            'class': 'logging.StreamHandler',
            'formatter': 'standard'
        },
        'file': {
            'level': 'DEBUG',
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': 'data/logs/agent_forge.log',
            'maxBytes': 10485760,  # 10MB
            'backupCount': 5,
            'formatter': 'detailed'
        }
    },
    'loggers': {
        'agent_forge': {
            'handlers': ['console', 'file'],
            'level': 'DEBUG',
            'propagate': False
        },
        'evomerge': {
            'handlers': ['console', 'file'],
            'level': 'INFO',
            'propagate': False
        },
        'quietstar': {
            'handlers': ['console', 'file'],
            'level': 'INFO',
            'propagate': False
        },
        'bitnet': {
            'handlers': ['console', 'file'],
            'level': 'INFO',
            'propagate': False
        }
    }
}
```

### Health Checks

```python
# healthcheck.py
import requests
import sys

def check_api_health():
    try:
        response = requests.get('http://localhost:8000/health', timeout=5)
        return response.status_code == 200
    except:
        return False

def check_web_health():
    try:
        response = requests.get('http://localhost:3000', timeout=5)
        return response.status_code == 200
    except:
        return False

if __name__ == "__main__":
    api_healthy = check_api_health()
    web_healthy = check_web_health()

    if api_healthy and web_healthy:
        print("✅ All services healthy")
        sys.exit(0)
    else:
        print("❌ Health check failed")
        print(f"API: {'✅' if api_healthy else '❌'}")
        print(f"Web: {'✅' if web_healthy else '❌'}")
        sys.exit(1)
```

### Prometheus Metrics (Optional)

```python
# metrics.py
from prometheus_client import Counter, Histogram, Gauge, start_http_server

# Metrics
PHASE_EXECUTIONS = Counter('agent_forge_phase_executions_total',
                          'Total phase executions', ['phase', 'status'])
PHASE_DURATION = Histogram('agent_forge_phase_duration_seconds',
                          'Phase execution duration', ['phase'])
ACTIVE_PHASES = Gauge('agent_forge_active_phases',
                     'Number of currently active phases')

# Start metrics server
start_http_server(9090)
```

## Security Considerations

### API Security

```python
# security.py
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt

security = HTTPBearer()

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=["HS256"])
        return payload
    except jwt.PyJWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials"
        )
```

### Model Security

```python
# model_security.py
import hashlib

def verify_model_integrity(model_path: str, expected_hash: str) -> bool:
    """Verify model file integrity using SHA-256 hash"""
    sha256_hash = hashlib.sha256()
    with open(model_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256_hash.update(chunk)

    return sha256_hash.hexdigest() == expected_hash
```

## Backup and Recovery

### Backup Strategy

```bash
#!/bin/bash
# backup.sh

BACKUP_DIR="/backup/agent-forge/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

# Backup configuration
cp -r config/ "$BACKUP_DIR/"

# Backup models (selective)
cp -r data/models/checkpoints/ "$BACKUP_DIR/checkpoints/"

# Backup results
cp -r workspace/results/ "$BACKUP_DIR/results/"

# Backup database (if using)
sqlite3 data/agent_forge.db ".backup $BACKUP_DIR/agent_forge.db"

# Create archive
tar -czf "$BACKUP_DIR.tar.gz" -C /backup/agent-forge "$(basename $BACKUP_DIR)"
rm -rf "$BACKUP_DIR"

echo "Backup completed: $BACKUP_DIR.tar.gz"
```

### Recovery Procedure

```bash
#!/bin/bash
# restore.sh

BACKUP_FILE="$1"

if [ -z "$BACKUP_FILE" ]; then
    echo "Usage: $0 <backup_file.tar.gz>"
    exit 1
fi

# Stop services
sudo systemctl stop agent-forge-api
sudo systemctl stop agent-forge-web

# Extract backup
TEMP_DIR="/tmp/agent-forge-restore"
mkdir -p "$TEMP_DIR"
tar -xzf "$BACKUP_FILE" -C "$TEMP_DIR"

# Restore files
cp -r "$TEMP_DIR"/*/config/ ./
cp -r "$TEMP_DIR"/*/checkpoints/ ./data/models/
cp -r "$TEMP_DIR"/*/results/ ./workspace/

# Restore database
cp "$TEMP_DIR"/*/agent_forge.db ./data/

# Start services
sudo systemctl start agent-forge-api
sudo systemctl start agent-forge-web

# Cleanup
rm -rf "$TEMP_DIR"

echo "Restore completed"
```

## Troubleshooting

### Common Issues

1. **Out of Memory Errors**
   ```bash
   # Reduce batch sizes in configuration
   export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

   # Use gradient checkpointing
   model.gradient_checkpointing_enable()
   ```

2. **Model Loading Failures**
   ```bash
   # Clear cache and retry
   rm -rf ~/.cache/huggingface/
   python -c "from transformers import AutoModel; AutoModel.from_pretrained('model_name')"
   ```

3. **Permission Errors**
   ```bash
   # Fix permissions
   sudo chown -R agent-forge:agent-forge /opt/agent-forge
   chmod -R 755 /opt/agent-forge/data
   ```

4. **Port Conflicts**
   ```bash
   # Check port usage
   netstat -tulpn | grep :8000

   # Kill conflicting processes
   sudo kill -9 $(lsof -t -i:8000)
   ```

### Debug Mode

```bash
# Enable debug logging
export AGENT_FORGE_LOG_LEVEL=DEBUG
export TRANSFORMERS_VERBOSITY=debug

# Run with detailed logging
python -u src/api/pipeline_server_fixed.py 2>&1 | tee debug.log
```

This deployment guide provides comprehensive instructions for deploying Agent Forge in various environments, from local development to production cloud deployments with proper monitoring, security, and backup strategies.