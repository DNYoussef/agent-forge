/**
 * Compatibility Validation and Testing Utilities
 * Ensures API integration maintains exact compatibility with existing UI
 */

import {
  CognateConfig,
  CognateMetrics,
  EvoMergeMetrics,
  CognateStartResponse,
  CognateStatusResponse,
  PhaseStatus,
  ApiResponse,
} from '../types/phase-interfaces';

import { apiClient } from '../utils/api-client';
import {
  startCognatePhase as simulateStartCognatePhase,
  getCognateStatus as simulateGetCognateStatus,
} from '../simulation/cognate-simulation';
import { getEvoMergeMetrics as simulateGetEvoMergeMetrics } from '../simulation/evomerge-simulation';

export interface ValidationResult {
  passed: boolean;
  errors: string[];
  warnings: string[];
  details?: any;
  duration?: number;
}

export interface CompatibilityTestResult {
  testName: string;
  backend: ValidationResult;
  simulation: ValidationResult;
  compatibility: {
    passed: boolean;
    schemaMatches: boolean;
    fieldMatches: string[];
    fieldMismatches: string[];
    typeMatches: boolean;
  };
}

class CompatibilityValidator {
  private testSessionId: string;

  constructor() {
    this.testSessionId = `test_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  /**
   * Run full compatibility test suite
   */
  async runFullCompatibilityTests(): Promise<{
    overallPassed: boolean;
    results: CompatibilityTestResult[];
    summary: {
      totalTests: number;
      passed: number;
      failed: number;
      warnings: number;
    };
  }> {
    console.log('[Validator] Starting full compatibility test suite...');

    const tests = [
      'cognateStartResponse',
      'cognateStatusResponse',
      'evoMergeMetricsResponse',
      'errorHandling',
      'sessionManagement',
    ];

    const results: CompatibilityTestResult[] = [];

    for (const testName of tests) {
      console.log(`[Validator] Running test: ${testName}`);
      const result = await this.runSingleTest(testName);
      results.push(result);
    }

    const summary = {
      totalTests: results.length,
      passed: results.filter(r => r.compatibility.passed).length,
      failed: results.filter(r => !r.compatibility.passed).length,
      warnings: results.reduce((sum, r) => sum + r.backend.warnings.length + r.simulation.warnings.length, 0),
    };

    const overallPassed = summary.failed === 0;

    console.log('[Validator] Compatibility test suite completed:', summary);

    return {
      overallPassed,
      results,
      summary,
    };
  }

  /**
   * Run a single compatibility test
   */
  private async runSingleTest(testName: string): Promise<CompatibilityTestResult> {
    const sessionId = `${this.testSessionId}_${testName}`;

    switch (testName) {
      case 'cognateStartResponse':
        return await this.testCognateStartResponse(sessionId);

      case 'cognateStatusResponse':
        return await this.testCognateStatusResponse(sessionId);

      case 'evoMergeMetricsResponse':
        return await this.testEvoMergeMetricsResponse(sessionId);

      case 'errorHandling':
        return await this.testErrorHandling(sessionId);

      case 'sessionManagement':
        return await this.testSessionManagement(sessionId);

      default:
        return {
          testName,
          backend: { passed: false, errors: ['Unknown test'], warnings: [] },
          simulation: { passed: false, errors: ['Unknown test'], warnings: [] },
          compatibility: { passed: false, schemaMatches: false, fieldMatches: [], fieldMismatches: [], typeMatches: false },
        };
    }
  }

  /**
   * Test Cognate start response compatibility
   */
  private async testCognateStartResponse(sessionId: string): Promise<CompatibilityTestResult> {
    const config: CognateConfig = {
      sessionId,
      maxIterations: 5,
      convergenceThreshold: 0.9,
      parallelAgents: 2,
    };

    // Test backend
    let backendResult: ValidationResult;
    let backendResponse: any = null;

    try {
      const startTime = Date.now();
      backendResponse = await apiClient.post('/api/cognate/start', config);
      const duration = Date.now() - startTime;

      backendResult = this.validateCognateStartResponse(backendResponse, duration);
    } catch (error) {
      backendResult = {
        passed: false,
        errors: [error.message || 'Backend request failed'],
        warnings: [],
      };
    }

    // Test simulation
    let simulationResult: ValidationResult;
    let simulationResponse: any = null;

    try {
      const startTime = Date.now();
      simulationResponse = await simulateStartCognatePhase(sessionId, config);
      const duration = Date.now() - startTime;

      simulationResult = this.validateCognateStartResponse(simulationResponse, duration);
    } catch (error) {
      simulationResult = {
        passed: false,
        errors: [error.message || 'Simulation request failed'],
        warnings: [],
      };
    }

    // Compare responses
    const compatibility = this.compareResponses(backendResponse, simulationResponse, 'CognateStartResponse');

    return {
      testName: 'cognateStartResponse',
      backend: backendResult,
      simulation: simulationResult,
      compatibility,
    };
  }

  /**
   * Test Cognate status response compatibility
   */
  private async testCognateStatusResponse(sessionId: string): Promise<CompatibilityTestResult> {
    // First start a phase (use simulation to ensure it exists)
    const config: CognateConfig = { sessionId };
    await simulateStartCognatePhase(sessionId, config);

    // Test backend
    let backendResult: ValidationResult;
    let backendResponse: any = null;

    try {
      const startTime = Date.now();
      backendResponse = await apiClient.get(`/api/cognate/status/${sessionId}`);
      const duration = Date.now() - startTime;

      backendResult = this.validateCognateStatusResponse(backendResponse, duration);
    } catch (error) {
      backendResult = {
        passed: false,
        errors: [error.message || 'Backend request failed'],
        warnings: [],
      };
    }

    // Test simulation
    let simulationResult: ValidationResult;
    let simulationResponse: any = null;

    try {
      const startTime = Date.now();
      simulationResponse = await simulateGetCognateStatus(sessionId);
      const duration = Date.now() - startTime;

      simulationResult = this.validateCognateStatusResponse(simulationResponse, duration);
    } catch (error) {
      simulationResult = {
        passed: false,
        errors: [error.message || 'Simulation request failed'],
        warnings: [],
      };
    }

    // Compare responses
    const compatibility = this.compareResponses(backendResponse, simulationResponse, 'CognateStatusResponse');

    return {
      testName: 'cognateStatusResponse',
      backend: backendResult,
      simulation: simulationResult,
      compatibility,
    };
  }

  /**
   * Test EvoMerge metrics response compatibility
   */
  private async testEvoMergeMetricsResponse(sessionId: string): Promise<CompatibilityTestResult> {
    // Test backend
    let backendResult: ValidationResult;
    let backendResponse: any = null;

    try {
      const startTime = Date.now();
      backendResponse = await apiClient.get(`/api/evomerge/metrics/${sessionId}`);
      const duration = Date.now() - startTime;

      backendResult = this.validateEvoMergeMetricsResponse(backendResponse, duration);
    } catch (error) {
      backendResult = {
        passed: false,
        errors: [error.message || 'Backend request failed'],
        warnings: [],
      };
    }

    // Test simulation
    let simulationResult: ValidationResult;
    let simulationResponse: any = null;

    try {
      const startTime = Date.now();
      simulationResponse = await simulateGetEvoMergeMetrics(sessionId);
      const duration = Date.now() - startTime;

      simulationResult = this.validateEvoMergeMetricsResponse(simulationResponse, duration);
    } catch (error) {
      simulationResult = {
        passed: false,
        errors: [error.message || 'Simulation request failed'],
        warnings: [],
      };
    }

    // Compare responses
    const compatibility = this.compareResponses(backendResponse, simulationResponse, 'EvoMergeMetrics');

    return {
      testName: 'evoMergeMetricsResponse',
      backend: backendResult,
      simulation: simulationResult,
      compatibility,
    };
  }

  /**
   * Test error handling compatibility
   */
  private async testErrorHandling(sessionId: string): Promise<CompatibilityTestResult> {
    // Test invalid session ID handling
    const invalidSessionId = '';

    let backendResult: ValidationResult;
    let simulationResult: ValidationResult;

    try {
      await apiClient.get(`/api/cognate/status/${invalidSessionId}`);
      backendResult = { passed: false, errors: ['Should have failed with invalid session'], warnings: [] };
    } catch (error) {
      backendResult = { passed: true, errors: [], warnings: [], details: error.message };
    }

    try {
      await simulateGetCognateStatus(invalidSessionId);
      simulationResult = { passed: false, errors: ['Should have failed with invalid session'], warnings: [] };
    } catch (error) {
      simulationResult = { passed: true, errors: [], warnings: [], details: error.message };
    }

    const compatibility = {
      passed: backendResult.passed && simulationResult.passed,
      schemaMatches: true, // Both should fail
      fieldMatches: [],
      fieldMismatches: [],
      typeMatches: true,
    };

    return {
      testName: 'errorHandling',
      backend: backendResult,
      simulation: simulationResult,
      compatibility,
    };
  }

  /**
   * Test session management compatibility
   */
  private async testSessionManagement(sessionId: string): Promise<CompatibilityTestResult> {
    // This is a placeholder for session-related tests
    // In a real implementation, you would test session creation, cleanup, etc.

    const backendResult: ValidationResult = { passed: true, errors: [], warnings: ['Session management test not fully implemented'] };
    const simulationResult: ValidationResult = { passed: true, errors: [], warnings: ['Session management test not fully implemented'] };

    const compatibility = {
      passed: true,
      schemaMatches: true,
      fieldMatches: [],
      fieldMismatches: [],
      typeMatches: true,
    };

    return {
      testName: 'sessionManagement',
      backend: backendResult,
      simulation: simulationResult,
      compatibility,
    };
  }

  /**
   * Validate Cognate start response structure
   */
  private validateCognateStartResponse(response: any, duration?: number): ValidationResult {
    const errors: string[] = [];
    const warnings: string[] = [];

    if (!response) {
      errors.push('Response is null or undefined');
      return { passed: false, errors, warnings };
    }

    // Required fields
    const requiredFields = ['success', 'sessionId', 'status', 'timestamp'];
    for (const field of requiredFields) {
      if (!(field in response)) {
        errors.push(`Missing required field: ${field}`);
      }
    }

    // Type validation
    if (typeof response.success !== 'boolean') {
      errors.push('success field must be boolean');
    }

    if (typeof response.sessionId !== 'string') {
      errors.push('sessionId field must be string');
    }

    if (typeof response.status !== 'string') {
      errors.push('status field must be string');
    }

    // Performance warnings
    if (duration && duration > 5000) {
      warnings.push(`Response took ${duration}ms, which is longer than expected`);
    }

    return {
      passed: errors.length === 0,
      errors,
      warnings,
      duration,
    };
  }

  /**
   * Validate Cognate status response structure
   */
  private validateCognateStatusResponse(response: any, duration?: number): ValidationResult {
    const errors: string[] = [];
    const warnings: string[] = [];

    if (!response) {
      errors.push('Response is null or undefined');
      return { passed: false, errors, warnings };
    }

    // Required fields
    const requiredFields = ['sessionId', 'status', 'currentPhase', 'startTime', 'lastActivity'];
    for (const field of requiredFields) {
      if (!(field in response)) {
        errors.push(`Missing required field: ${field}`);
      }
    }

    // Metrics validation
    if (response.metrics) {
      const metricsFields = ['iterationsCompleted', 'convergenceScore', 'activeAgents'];
      for (const field of metricsFields) {
        if (!(field in response.metrics)) {
          warnings.push(`Missing metrics field: ${field}`);
        }
      }
    }

    return {
      passed: errors.length === 0,
      errors,
      warnings,
      duration,
    };
  }

  /**
   * Validate EvoMerge metrics response structure
   */
  private validateEvoMergeMetricsResponse(response: any, duration?: number): ValidationResult {
    const errors: string[] = [];
    const warnings: string[] = [];

    if (!response) {
      errors.push('Response is null or undefined');
      return { passed: false, errors, warnings };
    }

    // Required fields
    const requiredFields = ['generationsCompleted', 'populationSize', 'fitnessScore', 'lastUpdated'];
    for (const field of requiredFields) {
      if (!(field in response)) {
        errors.push(`Missing required field: ${field}`);
      }
    }

    // Type and range validation
    if (typeof response.fitnessScore !== 'number' || response.fitnessScore < 0 || response.fitnessScore > 1) {
      warnings.push('fitnessScore should be a number between 0 and 1');
    }

    return {
      passed: errors.length === 0,
      errors,
      warnings,
      duration,
    };
  }

  /**
   * Compare two responses for schema compatibility
   */
  private compareResponses(backend: any, simulation: any, responseName: string): {
    passed: boolean;
    schemaMatches: boolean;
    fieldMatches: string[];
    fieldMismatches: string[];
    typeMatches: boolean;
  } {
    const fieldMatches: string[] = [];
    const fieldMismatches: string[] = [];

    if (!backend || !simulation) {
      return {
        passed: false,
        schemaMatches: false,
        fieldMatches,
        fieldMismatches: ['One or both responses are null'],
        typeMatches: false,
      };
    }

    // Get all unique field names
    const backendFields = new Set(this.getAllFieldNames(backend));
    const simulationFields = new Set(this.getAllFieldNames(simulation));
    const allFields = new Set([...backendFields, ...simulationFields]);

    let typeMatches = true;

    for (const field of allFields) {
      const backendHasField = backendFields.has(field);
      const simulationHasField = simulationFields.has(field);

      if (backendHasField && simulationHasField) {
        const backendValue = this.getNestedValue(backend, field);
        const simulationValue = this.getNestedValue(simulation, field);
        const backendType = typeof backendValue;
        const simulationType = typeof simulationValue;

        if (backendType === simulationType) {
          fieldMatches.push(field);
        } else {
          fieldMismatches.push(`${field}: backend(${backendType}) vs simulation(${simulationType})`);
          typeMatches = false;
        }
      } else if (backendHasField) {
        fieldMismatches.push(`${field}: only in backend`);
      } else {
        fieldMismatches.push(`${field}: only in simulation`);
      }
    }

    const schemaMatches = fieldMismatches.length === 0;
    const passed = schemaMatches && typeMatches;

    return {
      passed,
      schemaMatches,
      fieldMatches,
      fieldMismatches,
      typeMatches,
    };
  }

  /**
   * Get all field names recursively (for nested objects)
   */
  private getAllFieldNames(obj: any, prefix = ''): string[] {
    const fields: string[] = [];

    if (obj && typeof obj === 'object' && !Array.isArray(obj)) {
      for (const [key, value] of Object.entries(obj)) {
        const fieldName = prefix ? `${prefix}.${key}` : key;
        fields.push(fieldName);

        if (value && typeof value === 'object' && !Array.isArray(value)) {
          fields.push(...this.getAllFieldNames(value, fieldName));
        }
      }
    }

    return fields;
  }

  /**
   * Get nested value by field path
   */
  private getNestedValue(obj: any, fieldPath: string): any {
    const parts = fieldPath.split('.');
    let current = obj;

    for (const part of parts) {
      if (current && typeof current === 'object' && part in current) {
        current = current[part];
      } else {
        return undefined;
      }
    }

    return current;
  }
}

// Export default validator instance
export const compatibilityValidator = new CompatibilityValidator();

// Utility functions for quick tests
export const validationUtils = {
  /**
   * Quick health check
   */
  async quickHealthCheck(): Promise<boolean> {
    try {
      const result = await apiClient.healthCheck();
      return result.healthy;
    } catch {
      return false;
    }
  },

  /**
   * Quick response format validation
   */
  validateResponseFormat(response: any, expectedType: 'CognateStartResponse' | 'CognateStatusResponse' | 'EvoMergeMetrics'): string[] {
    const validator = new CompatibilityValidator();

    switch (expectedType) {
      case 'CognateStartResponse':
        return validator['validateCognateStartResponse'](response).errors;
      case 'CognateStatusResponse':
        return validator['validateCognateStatusResponse'](response).errors;
      case 'EvoMergeMetrics':
        return validator['validateEvoMergeMetricsResponse'](response).errors;
      default:
        return ['Unknown response type'];
    }
  },
};