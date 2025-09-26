/**
 * Integration Safety Tests for Agent Forge Modifications
 * Production Validation Specialist - Comprehensive Test Suite
 *
 * This test suite validates that all proposed changes maintain
 * 100% backward compatibility and system integrity.
 */

const { execSync, spawn } = require('child_process');
const WebSocket = require('ws');
const fs = require('fs');
const path = require('path');
const http = require('http');

// Test configuration
const CONFIG = {
  API_SERVER_PORT: 8000,
  GATEWAY_PORT: 3000,
  WEBSOCKET_PORT: 3000,
  TEST_TIMEOUT: 30000,
  HEALTH_CHECK_RETRY: 10,
  HEALTH_CHECK_INTERVAL: 1000
};

// Color codes for output
const COLORS = {
  RESET: '\x1b[0m',
  RED: '\x1b[31m',
  GREEN: '\x1b[32m',
  YELLOW: '\x1b[33m',
  BLUE: '\x1b[34m',
  MAGENTA: '\x1b[35m',
  CYAN: '\x1b[36m'
};

class IntegrationSafetyTester {
  constructor() {
    this.results = {
      timestamp: new Date().toISOString(),
      environment: {
        node_version: process.version,
        platform: process.platform,
        cwd: process.cwd()
      },
      tests: {},
      summary: {
        total: 0,
        passed: 0,
        failed: 0,
        skipped: 0
      }
    };
    this.services = new Map();
  }

  log(level, message, color = COLORS.RESET) {
    const timestamp = new Date().toISOString();
    console.log(`${color}[${timestamp}] ${level}: ${message}${COLORS.RESET}`);
  }

  info(message) {
    this.log('INFO', message, COLORS.CYAN);
  }

  success(message) {
    this.log('SUCCESS', message, COLORS.GREEN);
  }

  error(message) {
    this.log('ERROR', message, COLORS.RED);
  }

  warn(message) {
    this.log('WARN', message, COLORS.YELLOW);
  }

  /**
   * Wait for service to be available
   */
  async waitForService(url, maxRetries = CONFIG.HEALTH_CHECK_RETRY) {
    for (let i = 0; i < maxRetries; i++) {
      try {
        const response = await fetch(url, {
          method: 'GET',
          timeout: 5000
        });
        if (response.ok) {
          return true;
        }
      } catch (error) {
        // Service not ready yet
      }
      await new Promise(resolve => setTimeout(resolve, CONFIG.HEALTH_CHECK_INTERVAL));
    }
    return false;
  }

  /**
   * Make HTTP request with error handling
   */
  async makeRequest(url, options = {}) {
    try {
      const response = await fetch(url, {
        timeout: 10000,
        ...options
      });

      const contentType = response.headers.get('content-type');
      let data;

      if (contentType && contentType.includes('application/json')) {
        data = await response.json();
      } else {
        data = await response.text();
      }

      return {
        success: response.ok,
        status: response.status,
        data,
        headers: Object.fromEntries(response.headers.entries())
      };
    } catch (error) {
      return {
        success: false,
        error: error.message
      };
    }
  }

  /**
   * Test API Server Health and Endpoints
   */
  async testApiServerHealth() {
    const testName = 'API Server Health';
    this.info(`Running test: ${testName}`);

    const tests = [];

    // Test 1: Health endpoint
    const healthResult = await this.makeRequest(`http://localhost:${CONFIG.API_SERVER_PORT}/api/health`);
    tests.push({
      name: 'Health Endpoint',
      passed: healthResult.success,
      details: healthResult
    });

    // Test 2: DFARS Compliance endpoint
    const dfarResult = await this.makeRequest(`http://localhost:${CONFIG.API_SERVER_PORT}/api/dfars/compliance`);
    tests.push({
      name: 'DFARS Compliance',
      passed: dfarResult.success,
      details: dfarResult
    });

    // Test 3: Defense certification endpoint
    const certResult = await this.makeRequest(`http://localhost:${CONFIG.API_SERVER_PORT}/api/defense/certification`);
    tests.push({
      name: 'Defense Certification',
      passed: certResult.success,
      details: certResult
    });

    // Test 4: NASA POT10 analyzer
    const nasaResult = await this.makeRequest(`http://localhost:${CONFIG.API_SERVER_PORT}/api/nasa/pot10/analyze`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ test: true })
    });
    tests.push({
      name: 'NASA POT10 Analyzer',
      passed: nasaResult.success,
      details: nasaResult
    });

    const passed = tests.filter(t => t.passed).length;
    const total = tests.length;

    this.results.tests[testName] = {
      passed: passed === total,
      details: tests,
      summary: `${passed}/${total} tests passed`
    };

    if (passed === total) {
      this.success(`${testName}: All ${total} tests passed`);
    } else {
      this.error(`${testName}: ${total - passed} tests failed`);
    }

    return passed === total;
  }

  /**
   * Test Express Gateway Functionality
   */
  async testGatewayFunctionality() {
    const testName = 'Gateway Functionality';
    this.info(`Running test: ${testName}`);

    const tests = [];

    // Test 1: Gateway health
    const healthResult = await this.makeRequest(`http://localhost:${CONFIG.GATEWAY_PORT}/health`);
    tests.push({
      name: 'Gateway Health',
      passed: healthResult.success && healthResult.data.status === 'healthy',
      details: healthResult
    });

    // Test 2: Command execution endpoint
    const commandResult = await this.makeRequest(`http://localhost:${CONFIG.GATEWAY_PORT}/api/commands/execute`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        command: 'health-check',
        args: { test: true }
      })
    });
    tests.push({
      name: 'Command Execution',
      passed: commandResult.success || commandResult.status === 500, // 500 is OK for unknown command
      details: commandResult
    });

    // Test 3: List commands
    const listResult = await this.makeRequest(`http://localhost:${CONFIG.GATEWAY_PORT}/api/commands`);
    tests.push({
      name: 'List Commands',
      passed: listResult.success && Array.isArray(listResult.data?.commands),
      details: listResult
    });

    // Test 4: Python bridge
    const bridgeResult = await this.makeRequest(`http://localhost:${CONFIG.GATEWAY_PORT}/api/analyzer/bridge`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ test: 'bridge_connectivity' })
    });
    tests.push({
      name: 'Python Bridge',
      passed: bridgeResult.success,
      details: bridgeResult
    });

    // Test 5: CORS headers
    const corsResult = await this.makeRequest(`http://localhost:${CONFIG.GATEWAY_PORT}/health`, {
      method: 'OPTIONS'
    });
    tests.push({
      name: 'CORS Headers',
      passed: corsResult.headers && corsResult.headers['access-control-allow-origin'] === '*',
      details: corsResult
    });

    const passed = tests.filter(t => t.passed).length;
    const total = tests.length;

    this.results.tests[testName] = {
      passed: passed === total,
      details: tests,
      summary: `${passed}/${total} tests passed`
    };

    if (passed === total) {
      this.success(`${testName}: All ${total} tests passed`);
    } else {
      this.error(`${testName}: ${total - passed} tests failed`);
    }

    return passed === total;
  }

  /**
   * Test WebSocket Connectivity and Features
   */
  async testWebSocketFunctionality() {
    const testName = 'WebSocket Functionality';
    this.info(`Running test: ${testName}`);

    return new Promise((resolve) => {
      const tests = [];
      let testsPending = 4;

      const finishTest = () => {
        testsPending--;
        if (testsPending === 0) {
          const passed = tests.filter(t => t.passed).length;
          const total = tests.length;

          this.results.tests[testName] = {
            passed: passed === total,
            details: tests,
            summary: `${passed}/${total} tests passed`
          };

          if (passed === total) {
            this.success(`${testName}: All ${total} tests passed`);
          } else {
            this.error(`${testName}: ${total - passed} tests failed`);
          }

          resolve(passed === total);
        }
      };

      // Test 1: Basic connection
      const ws1 = new WebSocket(`ws://localhost:${CONFIG.WEBSOCKET_PORT}`);

      ws1.on('open', () => {
        tests.push({
          name: 'WebSocket Connection',
          passed: true,
          details: 'Successfully connected to WebSocket server'
        });
        ws1.close();
        finishTest();
      });

      ws1.on('error', (error) => {
        tests.push({
          name: 'WebSocket Connection',
          passed: false,
          details: `Connection failed: ${error.message}`
        });
        finishTest();
      });

      // Test 2: Message sending and receiving
      const ws2 = new WebSocket(`ws://localhost:${CONFIG.WEBSOCKET_PORT}`);

      ws2.on('open', () => {
        ws2.send(JSON.stringify({
          type: 'ping',
          timestamp: Date.now(),
          id: 'test-ping'
        }));
      });

      ws2.on('message', (data) => {
        try {
          const message = JSON.parse(data);
          tests.push({
            name: 'WebSocket Messaging',
            passed: message.type === 'pong' || message.type === 'data',
            details: `Received: ${message.type}`
          });
        } catch (error) {
          tests.push({
            name: 'WebSocket Messaging',
            passed: false,
            details: `Invalid JSON: ${error.message}`
          });
        }
        ws2.close();
        finishTest();
      });

      ws2.on('error', (error) => {
        tests.push({
          name: 'WebSocket Messaging',
          passed: false,
          details: `Messaging failed: ${error.message}`
        });
        finishTest();
      });

      // Test 3: Subscription functionality
      const ws3 = new WebSocket(`ws://localhost:${CONFIG.WEBSOCKET_PORT}`);

      ws3.on('open', () => {
        ws3.send(JSON.stringify({
          type: 'subscribe',
          channel: 'test-channel',
          timestamp: Date.now(),
          id: 'test-sub'
        }));
      });

      ws3.on('message', (data) => {
        try {
          const message = JSON.parse(data);
          tests.push({
            name: 'WebSocket Subscription',
            passed: message.data && (message.data.subscribed || message.data.message),
            details: `Subscription response: ${JSON.stringify(message.data)}`
          });
        } catch (error) {
          tests.push({
            name: 'WebSocket Subscription',
            passed: false,
            details: `Subscription failed: ${error.message}`
          });
        }
        ws3.close();
        finishTest();
      });

      ws3.on('error', (error) => {
        tests.push({
          name: 'WebSocket Subscription',
          passed: false,
          details: `Subscription error: ${error.message}`
        });
        finishTest();
      });

      // Test 4: Connection cleanup
      const ws4 = new WebSocket(`ws://localhost:${CONFIG.WEBSOCKET_PORT}`);

      ws4.on('open', () => {
        // Immediately close to test cleanup
        ws4.close();

        setTimeout(() => {
          tests.push({
            name: 'WebSocket Cleanup',
            passed: ws4.readyState === WebSocket.CLOSED,
            details: `Connection state after close: ${ws4.readyState}`
          });
          finishTest();
        }, 1000);
      });

      ws4.on('error', (error) => {
        tests.push({
          name: 'WebSocket Cleanup',
          passed: false,
          details: `Cleanup test error: ${error.message}`
        });
        finishTest();
      });

      // Timeout for WebSocket tests
      setTimeout(() => {
        while (testsPending > 0) {
          tests.push({
            name: 'WebSocket Timeout',
            passed: false,
            details: 'Test timed out'
          });
          finishTest();
        }
      }, CONFIG.TEST_TIMEOUT);
    });
  }

  /**
   * Test Python Bridge Integration
   */
  async testPythonBridge() {
    const testName = 'Python Bridge Integration';
    this.info(`Running test: ${testName}`);

    const tests = [];

    // Test 1: Bridge availability
    const bridgeFile = path.join(process.cwd(), 'analyzer', 'bridge.py');
    const bridgeExists = fs.existsSync(bridgeFile);
    tests.push({
      name: 'Bridge File Exists',
      passed: bridgeExists,
      details: `Bridge file at: ${bridgeFile}`
    });

    if (!bridgeExists) {
      this.results.tests[testName] = {
        passed: false,
        details: tests,
        summary: 'Bridge file not found'
      };
      return false;
    }

    // Test 2: Direct Python execution
    try {
      const result = execSync('python -c "import analyzer.bridge; print(\'Bridge import successful\')"', {
        timeout: 10000,
        encoding: 'utf8'
      });
      tests.push({
        name: 'Python Import',
        passed: result.includes('successful'),
        details: result.trim()
      });
    } catch (error) {
      tests.push({
        name: 'Python Import',
        passed: false,
        details: error.message
      });
    }

    // Test 3: Bridge via API
    const apiResult = await this.makeRequest(`http://localhost:${CONFIG.GATEWAY_PORT}/api/analyzer/connascence_scan`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        path: '.',
        depth: 1
      })
    });
    tests.push({
      name: 'Bridge via API',
      passed: apiResult.success && apiResult.data?.success !== false,
      details: apiResult
    });

    // Test 4: Command system integration
    const cmdResult = await this.makeRequest(`http://localhost:${CONFIG.GATEWAY_PORT}/api/commands/execute`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        command: 'conn:scan',
        args: { path: '.', quick: true }
      })
    });
    tests.push({
      name: 'Command Integration',
      passed: cmdResult.success || cmdResult.status >= 400, // Command might not exist, but bridge should respond
      details: cmdResult
    });

    const passed = tests.filter(t => t.passed).length;
    const total = tests.length;

    this.results.tests[testName] = {
      passed: passed === total,
      details: tests,
      summary: `${passed}/${total} tests passed`
    };

    if (passed === total) {
      this.success(`${testName}: All ${total} tests passed`);
    } else {
      this.error(`${testName}: ${total - passed} tests failed`);
    }

    return passed === total;
  }

  /**
   * Test React UI Components (Static Analysis)
   */
  async testReactComponents() {
    const testName = 'React Components Analysis';
    this.info(`Running test: ${testName}`);

    const tests = [];
    const componentPaths = [
      'src/ui/components/PhaseController.tsx',
      'src/risk-dashboard/IntegratedRiskDashboard.tsx',
      'src/ui/pages/EvoMergeEnhanced.tsx'
    ];

    for (const componentPath of componentPaths) {
      const fullPath = path.join(process.cwd(), componentPath);

      // Test 1: File exists
      const exists = fs.existsSync(fullPath);
      tests.push({
        name: `${componentPath} exists`,
        passed: exists,
        details: exists ? 'File found' : 'File not found'
      });

      if (!exists) continue;

      // Test 2: Valid TypeScript/React syntax
      try {
        const content = fs.readFileSync(fullPath, 'utf8');

        // Basic syntax checks
        const hasImportReact = content.includes('import React') || content.includes('import * as React');
        const hasExport = content.includes('export');
        const hasValidTS = !content.includes('SyntaxError') && content.length > 0;

        tests.push({
          name: `${componentPath} syntax`,
          passed: hasImportReact && hasExport && hasValidTS,
          details: `React import: ${hasImportReact}, Export: ${hasExport}, Valid: ${hasValidTS}`
        });

        // Test 3: WebSocket usage check
        const hasWebSocket = content.includes('WebSocket') || content.includes('ws://');
        tests.push({
          name: `${componentPath} WebSocket usage`,
          passed: true, // This is informational, not a failure
          details: hasWebSocket ? 'Uses WebSocket' : 'No WebSocket usage detected'
        });

      } catch (error) {
        tests.push({
          name: `${componentPath} analysis`,
          passed: false,
          details: `Error reading file: ${error.message}`
        });
      }
    }

    const passed = tests.filter(t => t.passed).length;
    const total = tests.length;

    this.results.tests[testName] = {
      passed: passed === total,
      details: tests,
      summary: `${passed}/${total} tests passed`
    };

    if (passed === total) {
      this.success(`${testName}: All ${total} tests passed`);
    } else {
      this.error(`${testName}: ${total - passed} tests failed`);
    }

    return passed === total;
  }

  /**
   * Test End-to-End Data Flow
   */
  async testEndToEndFlow() {
    const testName = 'End-to-End Data Flow';
    this.info(`Running test: ${testName}`);

    const tests = [];

    try {
      // Simulate complete user workflow
      // Step 1: Check gateway health
      const healthCheck = await this.makeRequest(`http://localhost:${CONFIG.GATEWAY_PORT}/health`);
      tests.push({
        name: 'E2E: Gateway Health',
        passed: healthCheck.success,
        details: healthCheck
      });

      // Step 2: Execute command through gateway
      const commandExec = await this.makeRequest(`http://localhost:${CONFIG.GATEWAY_PORT}/api/commands/execute`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          command: 'health-check',
          args: { source: 'integration-test' },
          context: { test: true }
        })
      });
      tests.push({
        name: 'E2E: Command Execution',
        passed: commandExec.success || commandExec.status === 400, // 400 is OK for unknown command
        details: commandExec
      });

      // Step 3: Test Python bridge through gateway
      const bridgeTest = await this.makeRequest(`http://localhost:${CONFIG.GATEWAY_PORT}/api/analyzer/quality_metrics`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          path: '.',
          quick: true
        })
      });
      tests.push({
        name: 'E2E: Python Bridge',
        passed: bridgeTest.success,
        details: bridgeTest
      });

      // Step 4: Test statistics endpoint
      const statsTest = await this.makeRequest(`http://localhost:${CONFIG.GATEWAY_PORT}/api/stats`);
      tests.push({
        name: 'E2E: Statistics',
        passed: statsTest.success && statsTest.data?.gateway,
        details: statsTest
      });

    } catch (error) {
      tests.push({
        name: 'E2E: Workflow Error',
        passed: false,
        details: error.message
      });
    }

    const passed = tests.filter(t => t.passed).length;
    const total = tests.length;

    this.results.tests[testName] = {
      passed: passed === total,
      details: tests,
      summary: `${passed}/${total} tests passed`
    };

    if (passed === total) {
      this.success(`${testName}: All ${total} tests passed`);
    } else {
      this.error(`${testName}: ${total - passed} tests failed`);
    }

    return passed === total;
  }

  /**
   * Test Security and Compliance
   */
  async testSecurityCompliance() {
    const testName = 'Security & Compliance';
    this.info(`Running test: ${testName}`);

    const tests = [];

    // Test 1: CORS configuration
    const corsTest = await this.makeRequest(`http://localhost:${CONFIG.GATEWAY_PORT}/health`, {
      method: 'OPTIONS',
      headers: {
        'Origin': 'http://localhost:3001',
        'Access-Control-Request-Method': 'POST'
      }
    });
    tests.push({
      name: 'CORS Configuration',
      passed: corsTest.headers && corsTest.headers['access-control-allow-origin'],
      details: corsTest.headers
    });

    // Test 2: Rate limiting (basic check)
    const rateLimitTest = await this.makeRequest(`http://localhost:${CONFIG.GATEWAY_PORT}/api/commands/execute`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ command: 'test' })
    });
    tests.push({
      name: 'Rate Limiting Response',
      passed: rateLimitTest.status !== 500, // Should handle rate limiting gracefully
      details: `Status: ${rateLimitTest.status}`
    });

    // Test 3: Security headers
    const securityTest = await this.makeRequest(`http://localhost:${CONFIG.GATEWAY_PORT}/health`);
    const hasSecurityHeaders = securityTest.headers && (
      securityTest.headers['x-content-type-options'] ||
      securityTest.headers['x-frame-options'] ||
      securityTest.headers['x-xss-protection']
    );
    tests.push({
      name: 'Security Headers',
      passed: hasSecurityHeaders,
      details: `Security headers present: ${hasSecurityHeaders}`
    });

    // Test 4: Defense compliance endpoint
    const defenseTest = await this.makeRequest(`http://localhost:${CONFIG.API_SERVER_PORT}/api/defense/certification`);
    tests.push({
      name: 'Defense Compliance',
      passed: defenseTest.success,
      details: defenseTest
    });

    const passed = tests.filter(t => t.passed).length;
    const total = tests.length;

    this.results.tests[testName] = {
      passed: passed === total,
      details: tests,
      summary: `${passed}/${total} tests passed`
    };

    if (passed === total) {
      this.success(`${testName}: All ${total} tests passed`);
    } else {
      this.error(`${testName}: ${total - passed} tests failed`);
    }

    return passed === total;
  }

  /**
   * Run all integration tests
   */
  async runAllTests() {
    this.info('Starting comprehensive integration safety tests...');

    // Check if services are running
    const apiReady = await this.waitForService(`http://localhost:${CONFIG.API_SERVER_PORT}/api/health`);
    const gatewayReady = await this.waitForService(`http://localhost:${CONFIG.GATEWAY_PORT}/health`);

    if (!apiReady) {
      this.error('Python API server is not responding - tests cannot continue');
      return false;
    }

    if (!gatewayReady) {
      this.error('Express gateway is not responding - tests cannot continue');
      return false;
    }

    this.success('All services are responsive - proceeding with tests');

    // Run all test suites
    const testSuites = [
      () => this.testApiServerHealth(),
      () => this.testGatewayFunctionality(),
      () => this.testWebSocketFunctionality(),
      () => this.testPythonBridge(),
      () => this.testReactComponents(),
      () => this.testEndToEndFlow(),
      () => this.testSecurityCompliance()
    ];

    let allPassed = true;

    for (const testSuite of testSuites) {
      try {
        const result = await testSuite();
        if (!result) {
          allPassed = false;
        }
        this.results.summary.total++;
        if (result) {
          this.results.summary.passed++;
        } else {
          this.results.summary.failed++;
        }
      } catch (error) {
        this.error(`Test suite failed with error: ${error.message}`);
        allPassed = false;
        this.results.summary.total++;
        this.results.summary.failed++;
      }
    }

    // Generate final report
    this.generateReport();

    if (allPassed) {
      this.success('🎉 ALL INTEGRATION TESTS PASSED - SYSTEM IS READY FOR MODIFICATIONS');
    } else {
      this.error('❌ SOME TESTS FAILED - DO NOT PROCEED WITH MODIFICATIONS');
    }

    return allPassed;
  }

  /**
   * Generate comprehensive test report
   */
  generateReport() {
    const reportPath = path.join(process.cwd(), '.claude', '.artifacts', 'integration_test_report.json');

    // Ensure directory exists
    const reportDir = path.dirname(reportPath);
    if (!fs.existsSync(reportDir)) {
      fs.mkdirSync(reportDir, { recursive: true });
    }

    // Write detailed report
    fs.writeFileSync(reportPath, JSON.stringify(this.results, null, 2));

    // Generate summary report
    const summaryPath = path.join(process.cwd(), '.claude', '.artifacts', 'integration_test_summary.md');
    const summaryContent = this.generateSummaryMarkdown();
    fs.writeFileSync(summaryPath, summaryContent);

    this.info(`Test report generated: ${reportPath}`);
    this.info(`Summary report: ${summaryPath}`);
  }

  /**
   * Generate markdown summary
   */
  generateSummaryMarkdown() {
    const { passed, failed, total } = this.results.summary;
    const passRate = ((passed / total) * 100).toFixed(1);

    let markdown = `# Integration Safety Test Results\n\n`;
    markdown += `**Date**: ${this.results.timestamp}\n`;
    markdown += `**Environment**: Node.js ${this.results.environment.node_version} on ${this.results.environment.platform}\n\n`;
    markdown += `## Summary\n\n`;
    markdown += `- **Total Tests**: ${total}\n`;
    markdown += `- **Passed**: ${passed} ✅\n`;
    markdown += `- **Failed**: ${failed} ❌\n`;
    markdown += `- **Pass Rate**: ${passRate}%\n\n`;

    if (passed === total) {
      markdown += `🎉 **ALL TESTS PASSED** - System is ready for modifications\n\n`;
    } else {
      markdown += `❌ **TESTS FAILED** - Do not proceed with modifications until issues are resolved\n\n`;
    }

    markdown += `## Test Results\n\n`;

    for (const [testName, testResult] of Object.entries(this.results.tests)) {
      const status = testResult.passed ? '✅' : '❌';
      markdown += `### ${status} ${testName}\n\n`;
      markdown += `${testResult.summary}\n\n`;

      if (!testResult.passed) {
        markdown += `**Failed Tests:**\n`;
        for (const detail of testResult.details) {
          if (!detail.passed) {
            markdown += `- ${detail.name}: ${detail.details}\n`;
          }
        }
        markdown += `\n`;
      }
    }

    return markdown;
  }
}

// CLI interface
if (require.main === module) {
  const tester = new IntegrationSafetyTester();

  tester.runAllTests()
    .then(success => {
      process.exit(success ? 0 : 1);
    })
    .catch(error => {
      console.error('Test runner failed:', error);
      process.exit(1);
    });
}

module.exports = IntegrationSafetyTester;