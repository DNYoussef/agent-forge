# Safety Tools Quick Start Guide
**Agent Forge Integration - Production Validation Tools**

## 🚨 Emergency Procedures

### Immediate Rollback (< 60 seconds)
```bash
# If something breaks during modifications:
cd /path/to/spek-template
bash scripts/safety_rollback_procedures.sh restore .backups/LATEST_TIMESTAMP
```

### Health Check
```bash
# Check if all services are running correctly:
bash scripts/safety_rollback_procedures.sh test
```

---

## 📋 Pre-Modification Checklist

### 1. Create System Backup
```bash
# ALWAYS run before ANY modifications:
bash scripts/safety_rollback_procedures.sh backup
```
**Output**: Creates timestamped backup in `.backups/YYYYMMDD_HHMMSS/`

### 2. Validate Current System
```bash
# Test all integrations work correctly:
node tests/integration/integration_safety_tests.js
```
**Expected**: 34 tests pass (100% pass rate)

### 3. Check Service Status
```bash
# Verify all services are running:
curl http://localhost:3000/health    # Express Gateway
curl http://localhost:8000/api/health # Python API
```

---

## 🔧 Safety Tools Reference

### Rollback Script Usage
```bash
# Location: scripts/safety_rollback_procedures.sh

# Available commands:
bash scripts/safety_rollback_procedures.sh backup           # Create backup
bash scripts/safety_rollback_procedures.sh restore <dir>    # Restore from backup
bash scripts/safety_rollback_procedures.sh list            # Show all backups
bash scripts/safety_rollback_procedures.sh test            # Test functionality
bash scripts/safety_rollback_procedures.sh cleanup         # Remove old backups
bash scripts/safety_rollback_procedures.sh stop            # Stop services
bash scripts/safety_rollback_procedures.sh start           # Start services
bash scripts/safety_rollback_procedures.sh help            # Show help
```

### Integration Testing
```bash
# Location: tests/integration/integration_safety_tests.js

# Run all tests:
node tests/integration/integration_safety_tests.js

# Reports generated:
.claude/.artifacts/integration_test_report.json     # Detailed results
.claude/.artifacts/integration_test_summary.md      # Human-readable summary
```

---

## ⚠️ Critical Safety Rules

### NEVER MODIFY WITHOUT:
1. ✅ **System backup created** (`scripts/safety_rollback_procedures.sh backup`)
2. ✅ **Integration tests passing** (node `tests/integration/integration_safety_tests.js`)
3. ✅ **Target files verified to exist** (especially `cognate_creator.py`)
4. ✅ **Port conflicts resolved** (Flask 8000 vs FastAPI)

### ALWAYS VERIFY AFTER CHANGES:
1. ✅ **Run integration tests** → Should show 100% pass rate
2. ✅ **Check health endpoints** → All should return 200 OK
3. ✅ **Test WebSocket connections** → Should connect successfully
4. ✅ **Validate Python bridge** → Should execute commands correctly

---

## 🎯 Test Status Indicators

### Integration Test Results:
```bash
# Expected output from integration tests:
🎉 ALL INTEGRATION TESTS PASSED - SYSTEM IS READY FOR MODIFICATIONS
# OR
❌ SOME TESTS FAILED - DO NOT PROCEED WITH MODIFICATIONS
```

### Service Health Indicators:
```bash
# Expected responses:
curl http://localhost:3000/health
{"status":"healthy","timestamp":"2025-09-25T...","uptime":123.45}

curl http://localhost:8000/api/health
{"status":"healthy","timestamp":"2025-09-25T...","version":"1.0.0","defense_ready":true}
```

---

## 🚦 Go/No-Go Decision Matrix

### ✅ **GO** Conditions (Safe to Proceed):
- All integration tests passing (34/34)
- System backup created and verified
- Target files exist or creation plan approved
- Port conflicts resolved
- Defense compliance preserved

### ❌ **NO-GO** Conditions (STOP Immediately):
- Any integration test failing
- Target cognate_creator.py not found
- Port 8000/8001 conflicts unresolved
- DFARS compliance endpoints broken
- WebSocket fallback not implemented

### ⚠️ **CONDITIONAL** Conditions (Requires Mitigation):
- New dependencies added without testing
- API response formats modified
- Authentication systems changed
- Error handling modified

---

## 📊 Monitoring During Changes

### Real-time Monitoring:
```bash
# Terminal 1: System logs
tail -f .backups/*/api_server.log
tail -f .backups/*/gateway.log

# Terminal 2: Health monitoring (every 5 seconds)
watch -n 5 'curl -s http://localhost:3000/health && curl -s http://localhost:8000/api/health'

# Terminal 3: Integration tests (run periodically)
while true; do
  node tests/integration/integration_safety_tests.js
  sleep 300  # Every 5 minutes
done
```

### Performance Metrics to Watch:
- API response time: < 200ms
- Memory usage: < 500MB per service
- WebSocket connections: < 100 concurrent
- Error rate: < 1%

---

## 🔍 Troubleshooting Common Issues

### Issue: Integration Tests Failing
```bash
# Diagnosis:
node tests/integration/integration_safety_tests.js
# Check: .claude/.artifacts/integration_test_report.json

# Common fixes:
bash scripts/safety_rollback_procedures.sh stop
bash scripts/safety_rollback_procedures.sh start
# Wait 30 seconds, then retry tests
```

### Issue: Service Won't Start
```bash
# Check what's running on ports:
netstat -tulpn | grep ':3000\|:8000\|:8001'

# Kill conflicting processes:
pkill -f "python.*api_server"
pkill -f "node.*gateway"

# Restart properly:
bash scripts/safety_rollback_procedures.sh start
```

### Issue: WebSocket Connection Failed
```bash
# Test WebSocket directly:
npm install -g wscat
wscat -c ws://localhost:3000
# Should connect successfully

# Check logs:
grep -i websocket .backups/*/gateway.log
```

### Issue: Python Bridge Not Responding
```bash
# Test bridge directly:
python -c "
import sys
sys.path.append('analyzer')
from bridge import AnalyzerBridge
bridge = AnalyzerBridge()
print(bridge.execute('quality_metrics', {'path': '.'}))"
```

---

## 📞 Emergency Response Plan

### Level 1: Service Restart Required
```bash
bash scripts/safety_rollback_procedures.sh stop
bash scripts/safety_rollback_procedures.sh start
bash scripts/safety_rollback_procedures.sh test
```

### Level 2: Partial Rollback Required
```bash
# Find latest backup:
bash scripts/safety_rollback_procedures.sh list

# Restore specific files (example):
cp .backups/TIMESTAMP/src/api_server.py src/
bash scripts/safety_rollback_procedures.sh start
```

### Level 3: Complete System Rollback
```bash
# Nuclear option - complete restoration:
bash scripts/safety_rollback_procedures.sh restore .backups/LAST_KNOWN_GOOD
```

### Level 4: System Corruption
```bash
# If rollback fails:
1. Stop all processes: pkill -f "python\|node"
2. Restore from backup manually: cp -r .backups/TIMESTAMP/* .
3. Check file integrity: bash scripts/safety_rollback_procedures.sh test
4. Contact system administrator if issues persist
```

---

## 📈 Success Criteria

### Before Declaring "COMPLETE":
- [ ] All integration tests pass (34/34)
- [ ] All health checks return 200 OK
- [ ] WebSocket connections work with fallback
- [ ] Python bridge responds correctly
- [ ] Defense compliance endpoints operational
- [ ] No memory leaks detected (< 500MB per service)
- [ ] Response times within SLA (< 200ms)

### Documentation Updated:
- [ ] System architecture diagrams
- [ ] API endpoint documentation
- [ ] Deployment procedures
- [ ] Monitoring and alerting setup

---

**Remember**: When in doubt, **STOP** and run the safety checks. It's always better to be cautious with production systems.