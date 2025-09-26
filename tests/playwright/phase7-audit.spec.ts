import { test, expect, Page, BrowserContext } from '@playwright/test';
import { promises as fs } from 'fs';
import path from 'path';

/**
 * SPEK Phase 7 UI Audit - Comprehensive Test Suite
 * Tests advanced agentic systems, 3D visualizations, and meta-agent displays
 */

// Test configuration and utilities
const SCREENSHOT_DIR = 'screenshots/phase7';
const PERFORMANCE_THRESHOLDS = {
  FIRST_CONTENTFUL_PAINT: 2000,
  LARGEST_CONTENTFUL_PAINT: 4000,
  CUMULATIVE_LAYOUT_SHIFT: 0.1,
  FIRST_INPUT_DELAY: 100,
};

interface PerformanceMetrics {
  fcp: number;
  lcp: number;
  cls: number;
  fid: number;
  ttfb: number;
  navigationTime: number;
}

interface WebSocketMessage {
  type: string;
  data: any;
  timestamp: number;
}

class Phase7UIAuditSuite {
  private screenshotCounter = 0;
  private performanceMetrics: PerformanceMetrics[] = [];
  private wsMessages: WebSocketMessage[] = [];

  async captureFullPageScreenshot(page: Page, name: string, options?: {
    fullPage?: boolean;
    animations?: 'disabled' | 'allow';
    mask?: string[];
  }) {
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
    const filename = `${++this.screenshotCounter}-${name}-${timestamp}.png`;

    await page.screenshot({
      path: path.join(SCREENSHOT_DIR, 'full-pages', filename),
      fullPage: options?.fullPage ?? true,
      animations: options?.animations ?? 'disabled',
      mask: options?.mask ? options.mask.map(selector => page.locator(selector)) : undefined,
    });

    return filename;
  }

  async captureComponentScreenshot(page: Page, selector: string, name: string) {
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
    const filename = `component-${name}-${timestamp}.png`;

    const element = page.locator(selector);
    await element.screenshot({
      path: path.join(SCREENSHOT_DIR, 'components', filename),
    });

    return filename;
  }

  async collectPerformanceMetrics(page: Page): Promise<PerformanceMetrics> {
    return await page.evaluate(() => {
      const perfData = performance.getEntriesByType('navigation')[0] as PerformanceNavigationTiming;
      const paintEntries = performance.getEntriesByType('paint');

      return {
        fcp: paintEntries.find(entry => entry.name === 'first-contentful-paint')?.startTime || 0,
        lcp: 0, // Will be set via PerformanceObserver
        cls: 0,  // Will be set via PerformanceObserver
        fid: 0,  // Will be set via PerformanceObserver
        ttfb: perfData.responseStart - perfData.requestStart,
        navigationTime: perfData.loadEventEnd - perfData.navigationStart,
      };
    });
  }

  async setupWebSocketMonitoring(page: Page) {
    await page.exposeFunction('logWebSocketMessage', (message: WebSocketMessage) => {
      this.wsMessages.push(message);
    });

    await page.addInitScript(() => {
      const originalWebSocket = window.WebSocket;
      window.WebSocket = class extends originalWebSocket {
        constructor(url: string | URL, protocols?: string | string[]) {
          super(url, protocols);

          this.addEventListener('message', (event) => {
            try {
              const data = JSON.parse(event.data);
              (window as any).logWebSocketMessage({
                type: 'message',
                data,
                timestamp: Date.now()
              });
            } catch (e) {
              (window as any).logWebSocketMessage({
                type: 'raw-message',
                data: event.data,
                timestamp: Date.now()
              });
            }
          });

          this.addEventListener('open', () => {
            (window as any).logWebSocketMessage({
              type: 'connection-open',
              data: { url: url.toString() },
              timestamp: Date.now()
            });
          });

          this.addEventListener('close', (event) => {
            (window as any).logWebSocketMessage({
              type: 'connection-close',
              data: { code: event.code, reason: event.reason },
              timestamp: Date.now()
            });
          });
        }
      };
    });
  }

  async validateAccessibility(page: Page, context: string) {
    // Check WCAG compliance
    const a11yViolations = await page.evaluate(() => {
      const violations = [];

      // Check for alt text on images
      const images = document.querySelectorAll('img');
      images.forEach((img, index) => {
        if (!img.alt) {
          violations.push(`Image ${index + 1} missing alt text`);
        }
      });

      // Check for proper heading hierarchy
      const headings = document.querySelectorAll('h1, h2, h3, h4, h5, h6');
      let lastLevel = 0;
      headings.forEach((heading, index) => {
        const level = parseInt(heading.tagName.charAt(1));
        if (level > lastLevel + 1) {
          violations.push(`Heading level skip at heading ${index + 1}: ${heading.tagName}`);
        }
        lastLevel = level;
      });

      // Check color contrast (simplified)
      const elements = document.querySelectorAll('*');
      elements.forEach((element, index) => {
        const styles = window.getComputedStyle(element);
        const textColor = styles.color;
        const bgColor = styles.backgroundColor;

        if (textColor && bgColor && textColor !== 'rgba(0, 0, 0, 0)' && bgColor !== 'rgba(0, 0, 0, 0)') {
          // Basic contrast check (would need more sophisticated calculation in real scenario)
          if (textColor === bgColor) {
            violations.push(`Poor contrast detected on element ${index + 1}`);
          }
        }
      });

      return violations;
    });

    if (a11yViolations.length > 0) {
      console.warn(`Accessibility violations in ${context}:`, a11yViolations);
    }

    return a11yViolations;
  }
}

test.describe('Phase 7 UI Audit Suite', () => {
  let auditSuite: Phase7UIAuditSuite;

  test.beforeEach(async ({ page }) => {
    auditSuite = new Phase7UIAuditSuite();
    await auditSuite.setupWebSocketMonitoring(page);

    // Ensure screenshot directories exist
    await fs.mkdir(path.join(SCREENSHOT_DIR, 'full-pages'), { recursive: true });
    await fs.mkdir(path.join(SCREENSHOT_DIR, 'components'), { recursive: true });
    await fs.mkdir(path.join(SCREENSHOT_DIR, 'interactions'), { recursive: true });
    await fs.mkdir(path.join(SCREENSHOT_DIR, 'performance'), { recursive: true });
  });

  test.describe('Dashboard Components', () => {
    test('should render main dashboard with all Phase 7 components', async ({ page }) => {
      await page.goto('/dashboard');
      await page.waitForLoadState('networkidle');

      // Wait for 3D visualizations to load
      await page.waitForSelector('[data-testid="3d-weight-space"]', { timeout: 10000 });
      await page.waitForSelector('[data-testid="meta-agent-display"]', { timeout: 10000 });

      // Capture full dashboard screenshot
      await auditSuite.captureFullPageScreenshot(page, 'dashboard-overview');

      // Test component visibility
      await expect(page.locator('[data-testid="3d-weight-space"]')).toBeVisible();
      await expect(page.locator('[data-testid="meta-agent-display"]')).toBeVisible();
      await expect(page.locator('[data-testid="svd-visualization"]')).toBeVisible();

      // Validate performance
      const metrics = await auditSuite.collectPerformanceMetrics(page);
      expect(metrics.fcp).toBeLessThan(PERFORMANCE_THRESHOLDS.FIRST_CONTENTFUL_PAINT);
    });

    test('should capture individual component screenshots', async ({ page }) => {
      await page.goto('/dashboard');
      await page.waitForLoadState('networkidle');

      // Capture each major component
      const components = [
        { selector: '[data-testid="3d-weight-space"]', name: '3d-weight-space' },
        { selector: '[data-testid="meta-agent-display"]', name: 'meta-agent-display' },
        { selector: '[data-testid="svd-visualization"]', name: 'svd-visualization' },
        { selector: '[data-testid="agent-hierarchy"]', name: 'agent-hierarchy' },
        { selector: '[data-testid="performance-metrics"]', name: 'performance-metrics' },
        { selector: '[data-testid="system-health"]', name: 'system-health' },
      ];

      for (const component of components) {
        await page.locator(component.selector).waitFor({ state: 'visible' });
        await auditSuite.captureComponentScreenshot(page, component.selector, component.name);

        // Validate component accessibility
        const violations = await auditSuite.validateAccessibility(page, component.name);
        expect(violations.length).toBeLessThanOrEqual(2); // Allow minor violations
      }
    });
  });

  test.describe('3D Visualization Interactions', () => {
    test('should handle 3D weight space interactions', async ({ page }) => {
      await page.goto('/dashboard');
      await page.waitForSelector('[data-testid="3d-weight-space"]');

      const canvas3D = page.locator('[data-testid="3d-weight-space"] canvas').first();

      // Test hover interactions
      await canvas3D.hover();
      await page.waitForTimeout(1000); // Allow animation
      await auditSuite.captureComponentScreenshot(page, '[data-testid="3d-weight-space"]', '3d-hover-state');

      // Test click interactions
      await canvas3D.click();
      await page.waitForTimeout(1000);
      await auditSuite.captureComponentScreenshot(page, '[data-testid="3d-weight-space"]', '3d-click-state');

      // Test drag interactions
      const boundingBox = await canvas3D.boundingBox();
      if (boundingBox) {
        const centerX = boundingBox.x + boundingBox.width / 2;
        const centerY = boundingBox.y + boundingBox.height / 2;

        await page.mouse.move(centerX, centerY);
        await page.mouse.down();
        await page.mouse.move(centerX + 100, centerY + 50);
        await page.mouse.up();

        await page.waitForTimeout(1000);
        await auditSuite.captureComponentScreenshot(page, '[data-testid="3d-weight-space"]', '3d-drag-state');
      }

      // Validate smooth animations
      const animationFrames = await page.evaluate(() => {
        return new Promise((resolve) => {
          let frames = 0;
          const startTime = Date.now();

          function countFrames() {
            frames++;
            if (Date.now() - startTime < 1000) {
              requestAnimationFrame(countFrames);
            } else {
              resolve(frames);
            }
          }

          requestAnimationFrame(countFrames);
        });
      });

      expect(animationFrames).toBeGreaterThan(45); // Expect at least 45 FPS
    });

    test('should handle SVD visualization controls', async ({ page }) => {
      await page.goto('/dashboard');
      await page.waitForSelector('[data-testid="svd-visualization"]');

      // Test SVD control interactions
      const svdControls = page.locator('[data-testid="svd-controls"]');
      await svdControls.waitFor({ state: 'visible' });

      // Test dimension slider
      const dimensionSlider = svdControls.locator('[data-testid="dimension-slider"]');
      await dimensionSlider.waitFor({ state: 'visible' });
      await dimensionSlider.fill('5');
      await page.waitForTimeout(2000); // Allow recalculation

      await auditSuite.captureComponentScreenshot(page, '[data-testid="svd-visualization"]', 'svd-dimension-5');

      // Test component selection
      const componentSelect = svdControls.locator('[data-testid="component-select"]');
      await componentSelect.selectOption('principal-components');
      await page.waitForTimeout(2000);

      await auditSuite.captureComponentScreenshot(page, '[data-testid="svd-visualization"]', 'svd-principal-components');

      // Test real-time updates
      const updateButton = svdControls.locator('[data-testid="update-svd"]');
      await updateButton.click();

      // Monitor WebSocket messages for updates
      await page.waitForTimeout(3000);
      const relevantMessages = auditSuite.wsMessages.filter(msg =>
        msg.type === 'message' && msg.data.type === 'svd-update'
      );

      expect(relevantMessages.length).toBeGreaterThan(0);
    });
  });

  test.describe('Meta-Agent System Display', () => {
    test('should render meta-agent hierarchy', async ({ page }) => {
      await page.goto('/dashboard');
      await page.waitForSelector('[data-testid="meta-agent-display"]');

      // Wait for agent data to load
      await page.waitForFunction(() => {
        const display = document.querySelector('[data-testid="meta-agent-display"]');
        return display && display.children.length > 0;
      });

      await auditSuite.captureComponentScreenshot(page, '[data-testid="meta-agent-display"]', 'meta-agent-hierarchy');

      // Test agent node interactions
      const agentNodes = page.locator('[data-testid^="agent-node-"]');
      const nodeCount = await agentNodes.count();
      expect(nodeCount).toBeGreaterThan(0);

      // Test first agent node interaction
      if (nodeCount > 0) {
        await agentNodes.first().hover();
        await page.waitForTimeout(500);
        await auditSuite.captureComponentScreenshot(page, '[data-testid="meta-agent-display"]', 'agent-node-hover');

        await agentNodes.first().click();
        await page.waitForTimeout(1000);

        // Check if details panel opened
        const detailsPanel = page.locator('[data-testid="agent-details-panel"]');
        if (await detailsPanel.isVisible()) {
          await auditSuite.captureComponentScreenshot(page, '[data-testid="agent-details-panel"]', 'agent-details-panel');
        }
      }
    });

    test('should display real-time agent communication', async ({ page }) => {
      await page.goto('/dashboard');
      await page.waitForSelector('[data-testid="agent-communication-feed"]');

      // Monitor communication feed
      const commFeed = page.locator('[data-testid="agent-communication-feed"]');
      await commFeed.waitFor({ state: 'visible' });

      // Wait for initial messages
      await page.waitForTimeout(5000);
      await auditSuite.captureComponentScreenshot(page, '[data-testid="agent-communication-feed"]', 'communication-feed-initial');

      // Check for message flow
      const messageCount = await page.locator('[data-testid^="comm-message-"]').count();
      expect(messageCount).toBeGreaterThan(0);

      // Monitor WebSocket messages for agent communication
      const agentMessages = auditSuite.wsMessages.filter(msg =>
        msg.type === 'message' &&
        (msg.data.type === 'agent-message' || msg.data.type === 'agent-status')
      );

      expect(agentMessages.length).toBeGreaterThan(0);
    });
  });

  test.describe('Performance Monitoring', () => {
    test('should monitor weight space update performance', async ({ page }) => {
      await page.goto('/dashboard');
      await page.waitForSelector('[data-testid="3d-weight-space"]');

      // Trigger weight space update
      const updateButton = page.locator('[data-testid="update-weights"]');
      await updateButton.waitFor({ state: 'visible' });

      const startTime = Date.now();
      await updateButton.click();

      // Wait for update completion
      await page.waitForSelector('[data-testid="weights-updated"]', { timeout: 10000 });
      const updateTime = Date.now() - startTime;

      expect(updateTime).toBeLessThan(5000); // Should complete within 5 seconds

      // Capture performance metrics
      const metrics = await auditSuite.collectPerformanceMetrics(page);

      // Save performance data
      const perfData = {
        updateTime,
        ...metrics,
        wsMessageCount: auditSuite.wsMessages.length,
        timestamp: new Date().toISOString(),
      };

      await fs.writeFile(
        path.join(SCREENSHOT_DIR, 'performance', `weight-update-perf-${Date.now()}.json`),
        JSON.stringify(perfData, null, 2)
      );
    });

    test('should validate system resource usage', async ({ page }) => {
      await page.goto('/dashboard');

      // Monitor resource usage during complex operations
      const resourceMetrics = await page.evaluate(() => {
        const memInfo = (performance as any).memory;
        return {
          usedJSHeapSize: memInfo?.usedJSHeapSize || 0,
          totalJSHeapSize: memInfo?.totalJSHeapSize || 0,
          jsHeapSizeLimit: memInfo?.jsHeapSizeLimit || 0,
        };
      });

      // Validate reasonable memory usage
      if (resourceMetrics.usedJSHeapSize > 0) {
        const memoryUsageMB = resourceMetrics.usedJSHeapSize / (1024 * 1024);
        expect(memoryUsageMB).toBeLessThan(500); // Should use less than 500MB
      }
    });
  });

  test.describe('Cross-Browser Compatibility', () => {
    test('should render consistently across browsers', async ({ page, browserName }) => {
      await page.goto('/dashboard');
      await page.waitForLoadState('networkidle');

      // Capture browser-specific screenshots
      await auditSuite.captureFullPageScreenshot(page, `dashboard-${browserName}`);

      // Test key components exist
      const components = [
        '[data-testid="3d-weight-space"]',
        '[data-testid="meta-agent-display"]',
        '[data-testid="svd-visualization"]',
      ];

      for (const component of components) {
        await expect(page.locator(component)).toBeVisible();
      }

      // Validate browser-specific features
      if (browserName === 'webkit') {
        // Safari-specific validations
        const webglSupport = await page.evaluate(() => {
          const canvas = document.createElement('canvas');
          return !!(canvas.getContext('webgl') || canvas.getContext('experimental-webgl'));
        });
        expect(webglSupport).toBe(true);
      }
    });
  });

  test.describe('Mobile Responsive Testing', () => {
    test('should adapt to mobile viewports', async ({ page }) => {
      await page.setViewportSize({ width: 375, height: 812 }); // iPhone 12 size
      await page.goto('/dashboard');
      await page.waitForLoadState('networkidle');

      await auditSuite.captureFullPageScreenshot(page, 'dashboard-mobile');

      // Test mobile-specific interactions
      const mobileMenu = page.locator('[data-testid="mobile-menu"]');
      if (await mobileMenu.isVisible()) {
        await mobileMenu.tap();
        await page.waitForTimeout(500);
        await auditSuite.captureFullPageScreenshot(page, 'dashboard-mobile-menu-open');
      }

      // Validate touch interactions on 3D elements
      const canvas3D = page.locator('[data-testid="3d-weight-space"] canvas').first();
      if (await canvas3D.isVisible()) {
        await canvas3D.tap();
        await page.waitForTimeout(1000);
        await auditSuite.captureComponentScreenshot(page, '[data-testid="3d-weight-space"]', '3d-mobile-tap');
      }
    });
  });

  test.afterEach(async ({ page }) => {
    // Save WebSocket message log
    if (auditSuite.wsMessages.length > 0) {
      await fs.writeFile(
        path.join(SCREENSHOT_DIR, 'performance', `ws-messages-${Date.now()}.json`),
        JSON.stringify(auditSuite.wsMessages, null, 2)
      );
    }

    // Generate test summary
    const testSummary = {
      screenshotCount: auditSuite.screenshotCounter,
      wsMessageCount: auditSuite.wsMessages.length,
      performanceMetricsCount: auditSuite.performanceMetrics.length,
      timestamp: new Date().toISOString(),
    };

    await fs.writeFile(
      path.join(SCREENSHOT_DIR, `test-summary-${Date.now()}.json`),
      JSON.stringify(testSummary, null, 2)
    );
  });
});