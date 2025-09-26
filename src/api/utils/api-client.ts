/**
 * HTTP Client Utilities for API Integration
 * Handles communication with Python backend and fallback to simulation
 */

import {
  ApiClientConfig,
  ApiError,
  NetworkError,
  TimeoutError,
  FallbackOptions,
} from '../types/phase-interfaces';

export class ApiClient {
  private config: ApiClientConfig;

  constructor(config: Partial<ApiClientConfig> = {}) {
    this.config = {
      baseUrl: 'http://localhost:8001',
      timeout: 5000,
      retryAttempts: 3,
      retryDelay: 1000,
      enableFallback: true,
      fallbackDelay: 500,
      ...config,
    };
  }

  /**
   * Make HTTP request with timeout and retry logic
   */
  async request<T = any>(
    endpoint: string,
    options: RequestInit = {},
    fallbackOptions: Partial<FallbackOptions> = {}
  ): Promise<T> {
    const url = `${this.config.baseUrl}${endpoint}`;
    const finalOptions: RequestInit = {
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
      ...options,
    };

    // Attempt real backend request with retries
    let lastError: Error | null = null;

    for (let attempt = 1; attempt <= this.config.retryAttempts; attempt++) {
      try {
        console.log(`[ApiClient] Attempting request ${attempt}/${this.config.retryAttempts} to ${url}`);

        const response = await this.makeRequestWithTimeout(url, finalOptions);

        if (!response.ok) {
          if (response.status >= 500) {
            // Server error - might be worth retrying
            throw new ApiError(`Server error: ${response.status}`, response.status);
          } else {
            // Client error - don't retry
            throw new ApiError(`Client error: ${response.status}`, response.status);
          }
        }

        const data = await response.json();
        console.log(`[ApiClient] Request successful on attempt ${attempt}`);
        return data;

      } catch (error) {
        lastError = error;
        console.warn(`[ApiClient] Request attempt ${attempt} failed:`, error);

        // Don't retry for client errors or on last attempt
        if (error instanceof ApiError && error.status && error.status < 500) {
          break;
        }

        if (attempt < this.config.retryAttempts) {
          await this.delay(this.config.retryDelay * attempt);
        }
      }
    }

    // If fallback is enabled, try fallback logic
    if (this.config.enableFallback || fallbackOptions.enableFallback) {
      console.warn('[ApiClient] All attempts failed, falling back to simulation');

      if (fallbackOptions.fallbackDelay || this.config.fallbackDelay) {
        await this.delay(fallbackOptions.fallbackDelay || this.config.fallbackDelay);
      }

      throw new NetworkError(
        'Backend unavailable - fallback should be handled by caller',
        lastError
      );
    }

    // Re-throw the last error if no fallback
    throw lastError || new NetworkError('All requests failed');
  }

  /**
   * Make request with timeout
   */
  private async makeRequestWithTimeout(
    url: string,
    options: RequestInit
  ): Promise<Response> {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), this.config.timeout);

    try {
      const response = await fetch(url, {
        ...options,
        signal: controller.signal,
      });
      return response;
    } catch (error) {
      if (error.name === 'AbortError') {
        throw new TimeoutError(`Request timeout after ${this.config.timeout}ms`);
      }
      throw new NetworkError('Network request failed', error);
    } finally {
      clearTimeout(timeoutId);
    }
  }

  /**
   * GET request
   */
  async get<T = any>(
    endpoint: string,
    fallbackOptions: Partial<FallbackOptions> = {}
  ): Promise<T> {
    return this.request<T>(endpoint, { method: 'GET' }, fallbackOptions);
  }

  /**
   * POST request
   */
  async post<T = any>(
    endpoint: string,
    body: any,
    fallbackOptions: Partial<FallbackOptions> = {}
  ): Promise<T> {
    return this.request<T>(
      endpoint,
      {
        method: 'POST',
        body: JSON.stringify(body),
      },
      fallbackOptions
    );
  }

  /**
   * PUT request
   */
  async put<T = any>(
    endpoint: string,
    body: any,
    fallbackOptions: Partial<FallbackOptions> = {}
  ): Promise<T> {
    return this.request<T>(
      endpoint,
      {
        method: 'PUT',
        body: JSON.stringify(body),
      },
      fallbackOptions
    );
  }

  /**
   * DELETE request
   */
  async delete<T = any>(
    endpoint: string,
    fallbackOptions: Partial<FallbackOptions> = {}
  ): Promise<T> {
    return this.request<T>(endpoint, { method: 'DELETE' }, fallbackOptions);
  }

  /**
   * Health check
   */
  async healthCheck(): Promise<{ healthy: boolean; error?: string }> {
    try {
      await this.get('/health', { enableFallback: false });
      return { healthy: true };
    } catch (error) {
      return {
        healthy: false,
        error: error.message || 'Health check failed'
      };
    }
  }

  /**
   * Utility delay function
   */
  private delay(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  /**
   * Update configuration
   */
  updateConfig(newConfig: Partial<ApiClientConfig>): void {
    this.config = { ...this.config, ...newConfig };
  }

  /**
   * Get current configuration
   */
  getConfig(): ApiClientConfig {
    return { ...this.config };
  }
}

// Default instance
export const apiClient = new ApiClient();

// Utility functions for specific endpoints
export const apiUtils = {
  /**
   * Start Cognate phase with fallback
   */
  async startCognatePhase(config: any, fallback: () => Promise<any>) {
    try {
      return await apiClient.post('/api/cognate/start', config);
    } catch (error) {
      if (error instanceof NetworkError) {
        console.warn('Falling back to simulation for cognate phase start');
        return await fallback();
      }
      throw error;
    }
  },

  /**
   * Get Cognate status with fallback
   */
  async getCognateStatus(sessionId: string, fallback: () => Promise<any>) {
    try {
      return await apiClient.get(`/api/cognate/status/${sessionId}`);
    } catch (error) {
      if (error instanceof NetworkError) {
        console.warn('Falling back to simulation for cognate status');
        return await fallback();
      }
      throw error;
    }
  },

  /**
   * Get EvoMerge metrics with fallback
   */
  async getEvoMergeMetrics(sessionId: string, fallback: () => Promise<any>) {
    try {
      return await apiClient.get(`/api/evomerge/metrics/${sessionId}`);
    } catch (error) {
      if (error instanceof NetworkError) {
        console.warn('Falling back to simulation for evomerge metrics');
        return await fallback();
      }
      throw error;
    }
  },
};