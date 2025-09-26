#!/usr/bin/env node
/**
 * Comprehensive Test Runner for API Integration
 * Runs all validation tests and generates compatibility reports
 */

import { compatibilityValidator, ValidationResult } from './compatibility-validator';
import { apiClient } from '../utils/api-client';
import { sessionManager, sessionUtils } from '../utils/session-manager';

interface TestSuite {
  name: string;
  tests: Array<{
    name: string;
    description: string;
    run: () => Promise<ValidationResult>;
  }>;
}

class APITestRunner {
  private results: Map<string, ValidationResult> = new Map();
  private startTime: Date;
  private endTime: Date;

  constructor() {
    this.startTime = new Date();
  }

  /**
   * Run all test suites
   */
  async runAllTests(): Promise<{
    passed: boolean;
    totalTests: number;
    passedTests: number;
    failedTests: number;
    duration: number;
    results: Array<{ testName: string; result: ValidationResult }>;
    summary: string;
  }> {
    console.log('🚀 Starting API Integration Test Suite...\n');

    const testSuites = this.getTestSuites();
    let totalTests = 0;
    let passedTests = 0;
    let failedTests = 0;

    for (const suite of testSuites) {
      console.log(`📋 Running test suite: ${suite.name}`);
      console.log('─'.repeat(50));

      for (const test of suite.tests) {
        totalTests++;
        console.log(`  ⏳ ${test.name}: ${test.description}`);

        try {
          const result = await test.run();
          this.results.set(test.name, result);

          if (result.passed) {
            passedTests++;
            console.log(`  ✅ ${test.name}: PASSED${result.duration ? ` (${result.duration}ms)` : ''}`);
          } else {
            failedTests++;
            console.log(`  ❌ ${test.name}: FAILED`);
            console.log(`     Errors: ${result.errors.join(', ')}`);
          }

          if (result.warnings.length > 0) {
            console.log(`     ⚠️  Warnings: ${result.warnings.join(', ')}`);
          }

        } catch (error) {
          failedTests++;
          const failResult: ValidationResult = {
            passed: false,
            errors: [error.message],
            warnings: []
          };
          this.results.set(test.name, failResult);
          console.log(`  💥 ${test.name}: ERROR - ${error.message}`);
        }

        // Small delay between tests
        await new Promise(resolve => setTimeout(resolve, 100));
      }

      console.log(''); // Empty line between suites
    }

    this.endTime = new Date();
    const duration = this.endTime.getTime() - this.startTime.getTime();

    const results = Array.from(this.results.entries()).map(([testName, result]) => ({
      testName,
      result
    }));

    const summary = this.generateSummary(totalTests, passedTests, failedTests, duration);
    console.log(summary);

    return {
      passed: failedTests === 0,
      totalTests,
      passedTests,
      failedTests,
      duration,
      results,
      summary
    };
  }

  /**
   * Define all test suites
   */
  private getTestSuites(): TestSuite[] {
    return [
      {
        name: 'Backend Health Tests',
        tests: [
          {
            name: 'backend_health_check',
            description: 'Check if Python backend is accessible',
            run: async () => {
              const healthResult = await apiClient.healthCheck();
              return {
                passed: healthResult.healthy,
                errors: healthResult.healthy ? [] : [healthResult.error || 'Health check failed'],
                warnings: []
              };
            }
          },
          {
            name: 'backend_response_time',
            description: 'Check backend response time',
            run: async () => {
              const startTime = Date.now();
              try {
                await apiClient.get('/health');
                const duration = Date.now() - startTime;
                return {
                  passed: true,
                  errors: [],
                  warnings: duration > 1000 ? [`Slow response: ${duration}ms`] : [],
                  duration
                };
              } catch (error) {
                return {
                  passed: false,
                  errors: [error.message],
                  warnings: []
                };
              }
            }
          }
        ]
      },
      {
        name: 'Session Management Tests',
        tests: [
          {
            name: 'session_creation',
            description: 'Test session creation and validation',
            run: async () => {
              const testSessionId = `test_${Date.now()}`;
              const { mapping, isNew } = await sessionUtils.getOrCreateSession(testSessionId);

              return {
                passed: mapping && isNew && mapping.nextjsSessionId === testSessionId,
                errors: !mapping ? ['Failed to create session'] : !isNew ? ['Session not marked as new'] : [],
                warnings: []
              };
            }
          },
          {
            name: 'session_validation',
            description: 'Test session ID validation',
            run: async () => {
              const validId = sessionUtils.validateSessionId('valid_session_123');
              const invalidId = sessionUtils.validateSessionId('');
              const invalidId2 = sessionUtils.validateSessionId(null);

              return {
                passed: validId !== null && invalidId === null && invalidId2 === null,
                errors: validId === null ? ['Valid ID rejected'] : invalidId !== null ? ['Invalid ID accepted'] : [],
                warnings: []
              };
            }
          }
        ]
      },
      {
        name: 'API Compatibility Tests',
        tests: [
          {
            name: 'compatibility_suite',
            description: 'Run full compatibility validation',
            run: async () => {
              const compatResult = await compatibilityValidator.runFullCompatibilityTests();
              return {
                passed: compatResult.overallPassed,
                errors: compatResult.results
                  .filter(r => !r.compatibility.passed)
                  .map(r => `${r.testName}: compatibility failed`),
                warnings: compatResult.results
                  .flatMap(r => [...r.backend.warnings, ...r.simulation.warnings]),
                details: compatResult.summary
              };
            }
          }
        ]
      },
      {
        name: 'Fallback Mechanism Tests',
        tests: [
          {
            name: 'fallback_on_backend_failure',
            description: 'Test fallback when backend is unavailable',
            run: async () => {
              // Temporarily break the backend URL
              const originalConfig = apiClient.getConfig();
              apiClient.updateConfig({ baseUrl: 'http://localhost:9999' }); // Non-existent port

              try {
                const response = await fetch('http://localhost:3000/api/phases/cognate', {
                  method: 'POST',
                  headers: { 'Content-Type': 'application/json' },
                  body: JSON.stringify({ sessionId: `fallback_test_${Date.now()}` })
                });

                const result = await response.json();

                // Restore original config
                apiClient.updateConfig(originalConfig);

                return {
                  passed: response.ok && result.success,
                  errors: !response.ok ? [`HTTP ${response.status}`] : !result.success ? ['Response not successful'] : [],
                  warnings: []
                };
              } catch (error) {
                // Restore original config
                apiClient.updateConfig(originalConfig);

                return {
                  passed: false,
                  errors: [error.message],
                  warnings: []
                };
              }
            }
          }
        ]
      },
      {
        name: 'Performance Tests',
        tests: [
          {
            name: 'concurrent_requests',
            description: 'Test handling of concurrent API requests',
            run: async () => {
              const sessionId = `perf_test_${Date.now()}`;
              const concurrentRequests = 5;
              const requests: Promise<any>[] = [];

              for (let i = 0; i < concurrentRequests; i++) {
                requests.push(
                  fetch('http://localhost:3000/api/phases/cognate', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ sessionId: `${sessionId}_${i}` })
                  }).then(r => r.json())
                );
              }

              try {
                const startTime = Date.now();
                const results = await Promise.all(requests);
                const duration = Date.now() - startTime;

                const allSuccessful = results.every(r => r.success);

                return {
                  passed: allSuccessful,
                  errors: allSuccessful ? [] : ['Some concurrent requests failed'],
                  warnings: duration > 5000 ? [`Slow concurrent processing: ${duration}ms`] : [],
                  duration
                };
              } catch (error) {
                return {
                  passed: false,
                  errors: [error.message],
                  warnings: []
                };
              }
            }
          }
        ]
      }
    ];
  }

  /**
   * Generate test summary
   */
  private generateSummary(total: number, passed: number, failed: number, duration: number): string {
    const passRate = ((passed / total) * 100).toFixed(1);
    const durationSeconds = (duration / 1000).toFixed(2);

    const summary = `
📊 Test Summary
${'='.repeat(50)}
Total Tests:     ${total}
Passed:          ${passed} (${passRate}%)
Failed:          ${failed}
Duration:        ${durationSeconds}s

${failed === 0 ? '🎉 All tests passed!' : `⚠️  ${failed} test(s) failed`}

${failed === 0
  ? '✅ API Integration is working correctly with proper fallback mechanisms.'
  : '❌ Some issues detected. Check the failed tests above.'
}
`;

    return summary;
  }

  /**
   * Generate detailed report
   */
  generateDetailedReport(): string {
    const results = Array.from(this.results.entries());
    let report = `# API Integration Test Report\n\nGenerated: ${new Date().toISOString()}\n\n`;

    for (const [testName, result] of results) {
      report += `## ${testName}\n`;
      report += `**Status:** ${result.passed ? '✅ PASSED' : '❌ FAILED'}\n`;

      if (result.duration) {
        report += `**Duration:** ${result.duration}ms\n`;
      }

      if (result.errors.length > 0) {
        report += `**Errors:**\n${result.errors.map(e => `- ${e}`).join('\n')}\n`;
      }

      if (result.warnings.length > 0) {
        report += `**Warnings:**\n${result.warnings.map(w => `- ${w}`).join('\n')}\n`;
      }

      if (result.details) {
        report += `**Details:** ${JSON.stringify(result.details, null, 2)}\n`;
      }

      report += '\n---\n\n';
    }

    return report;
  }
}

// CLI interface
async function main() {
  const runner = new APITestRunner();
  const results = await runner.runAllTests();

  // Write detailed report if requested
  if (process.argv.includes('--report')) {
    const fs = require('fs');
    const path = require('path');

    const reportPath = path.join(process.cwd(), 'api-test-report.md');
    const report = runner.generateDetailedReport();

    fs.writeFileSync(reportPath, report);
    console.log(`📄 Detailed report written to: ${reportPath}`);
  }

  // Exit with appropriate code
  process.exit(results.passed ? 0 : 1);
}

// Run if called directly
if (require.main === module) {
  main().catch(console.error);
}

export { APITestRunner };