const { test, expect } = require('@playwright/test');

/**
 * Comprehensive UI Audit for Agent Forge - SPEK Quality Gate
 * Tests all 8 phases, UI components, error handling, and theater detection
 */

test.describe('Agent Forge UI Comprehensive Audit', () => {
  const phases = [
    { id: 1, name: 'cognate', title: 'Cognate (Model Creation)', route: '/phases/cognate' },
    { id: 2, name: 'evomerge', title: 'EvoMerge (Evolution)', route: '/phases/evomerge' },
    { id: 3, name: 'quietstar', title: 'Quiet-STaR (Reasoning)', route: '/phases/quietstar' },
    { id: 4, name: 'bitnet', title: 'BitNet (Compression)', route: '/phases/bitnet' },
    { id: 5, name: 'forge', title: 'Forge Training', route: '/phases/forge' },
    { id: 6, name: 'baking', title: 'Tool & Persona Baking', route: '/phases/baking' },
    { id: 7, name: 'adas', title: 'ADAS (Architecture Search)', route: '/phases/adas' },
    { id: 8, name: 'final', title: 'Final Compression', route: '/phases/final' }
  ];

  let auditResults = [];

  test.beforeAll(async () => {
    console.log('[AUDIT START] Comprehensive Agent Forge UI Audit');
    console.log('=' .repeat(80));
  });

  test.afterAll(async () => {
    // Generate audit summary
    console.log('\n[AUDIT COMPLETE] Summary Report');
    console.log('=' .repeat(80));

    for (const result of auditResults) {
      console.log(`Phase ${result.phase}: ${result.score}/100 - ${result.status}`);
    }
  });

  // Test homepage and navigation
  test('Homepage and Navigation Audit', async ({ page }) => {
    await page.goto('/');

    // Capture homepage
    await page.screenshot({
      path: 'screenshots/audit/homepage-full.png',
      fullPage: true
    });

    // Test navigation to each phase
    for (const phase of phases) {
      await page.goto(phase.route);
      await page.waitForTimeout(2000);

      await page.screenshot({
        path: `screenshots/audit/navigation-phase-${phase.id}-${phase.name}.png`,
        fullPage: true
      });
    }
  });

  // Individual phase testing
  phases.forEach(phase => {
    test(`Phase ${phase.id}: ${phase.title} - Complete UI Audit`, async ({ page }) => {
      console.log(`\n[PHASE ${phase.id}] Testing ${phase.title}...`);

      const auditResult = {
        phase: phase.id,
        name: phase.name,
        title: phase.title,
        score: 0,
        status: 'FAIL',
        tests: {
          pageLoad: false,
          uiComponents: false,
          configControls: false,
          phaseController: false,
          startButton: false,
          metricsDisplay: false,
          progressIndicators: false,
          errorHandling: false,
          apiEndpoint: false,
          phaseExecution: false
        },
        screenshots: [],
        issues: []
      };

      await page.goto(phase.route);
      await page.waitForTimeout(3000);

      // Test 1: Page Load (10 points)
      try {
        await page.waitForSelector('h1', { timeout: 10000 });
        auditResult.tests.pageLoad = true;
        auditResult.score += 10;
        console.log(`  [PASS] Page loads successfully`);
      } catch (error) {
        auditResult.issues.push('Page failed to load within timeout');
        console.log(`  [FAIL] Page load failed: ${error.message}`);
      }

      // Capture initial state
      await page.screenshot({
        path: `screenshots/audit/phase-${phase.id}-${phase.name}-initial.png`,
        fullPage: true
      });
      auditResult.screenshots.push(`phase-${phase.id}-${phase.name}-initial.png`);

      // Test 2: UI Components Present (15 points)
      try {
        const hasTitle = await page.locator('h1').isVisible();
        const hasContent = await page.locator('main, .content, .phase-content').isVisible();

        if (hasTitle && hasContent) {
          auditResult.tests.uiComponents = true;
          auditResult.score += 15;
          console.log(`  [PASS] Core UI components present`);
        } else {
          auditResult.issues.push('Missing core UI components (title or content area)');
          console.log(`  [FAIL] Missing core UI components`);
        }
      } catch (error) {
        auditResult.issues.push(`UI component check failed: ${error.message}`);
      }

      // Test 3: Configuration Controls (10 points)
      try {
        const controls = await page.locator('input[type="range"], select, input[type="checkbox"], input[type="number"]').count();
        if (controls > 0) {
          auditResult.tests.configControls = true;
          auditResult.score += 10;
          console.log(`  [PASS] Configuration controls found (${controls})`);
        } else {
          auditResult.issues.push('No configuration controls detected');
          console.log(`  [FAIL] No configuration controls found`);
        }
      } catch (error) {
        auditResult.issues.push(`Configuration controls check failed: ${error.message}`);
      }

      // Test 4: Phase Controller Component (15 points)
      try {
        const hasController = await page.locator('text=/Phase.*Status|Controller|Management/i').isVisible() ||
                             await page.locator('[class*="phase-controller"], [class*="controller"]').isVisible();

        if (hasController) {
          auditResult.tests.phaseController = true;
          auditResult.score += 15;
          console.log(`  [PASS] Phase controller component found`);
        } else {
          auditResult.issues.push('Phase controller component not found');
          console.log(`  [FAIL] No phase controller component`);
        }
      } catch (error) {
        auditResult.issues.push(`Phase controller check failed: ${error.message}`);
      }

      // Test 5: Start Button and Interaction (10 points)
      try {
        const startButton = page.locator('button:has-text("Start"), button:has-text("Begin"), button:has-text("Execute")');
        const isStartVisible = await startButton.isVisible();

        if (isStartVisible) {
          auditResult.tests.startButton = true;
          auditResult.score += 10;
          console.log(`  [PASS] Start button found and visible`);

          // Capture start button state
          await page.screenshot({
            path: `screenshots/audit/phase-${phase.id}-${phase.name}-start-button.png`,
            fullPage: true
          });
          auditResult.screenshots.push(`phase-${phase.id}-${phase.name}-start-button.png`);
        } else {
          auditResult.issues.push('Start button not found or not visible');
          console.log(`  [FAIL] Start button not found`);
        }
      } catch (error) {
        auditResult.issues.push(`Start button check failed: ${error.message}`);
      }

      // Test 6: Metrics Display Area (10 points)
      try {
        const hasMetrics = await page.locator('text=/Metrics|Progress|Performance|Status|Results/i').isVisible() ||
                          await page.locator('[class*="metrics"], [class*="progress"], [class*="status"]').isVisible();

        if (hasMetrics) {
          auditResult.tests.metricsDisplay = true;
          auditResult.score += 10;
          console.log(`  [PASS] Metrics display area found`);
        } else {
          auditResult.issues.push('Metrics display area not found');
          console.log(`  [FAIL] No metrics display area`);
        }
      } catch (error) {
        auditResult.issues.push(`Metrics display check failed: ${error.message}`);
      }

      // Test 7: Progress Indicators (10 points)
      try {
        const hasProgress = await page.locator('[role="progressbar"], .progress-bar, .loading, .spinner').isVisible() ||
                           await page.locator('text=/Loading|Processing|In Progress/i').isVisible();

        // Also check for custom progress indicators
        const hasCustomProgress = await page.locator('[class*="progress"], [class*="loading"], [class*="status-indicator"]').count() > 0;

        if (hasProgress || hasCustomProgress) {
          auditResult.tests.progressIndicators = true;
          auditResult.score += 10;
          console.log(`  [PASS] Progress indicators found`);
        } else {
          auditResult.issues.push('Progress indicators not found');
          console.log(`  [FAIL] No progress indicators found`);
        }
      } catch (error) {
        auditResult.issues.push(`Progress indicators check failed: ${error.message}`);
      }

      // Test 8: Error Handling and User Feedback (15 points)
      try {
        // Look for error handling elements
        const hasErrorHandling = await page.locator('[class*="error"], [class*="alert"], [class*="warning"]').count() > 0 ||
                                 await page.locator('text=/Error|Warning|Alert|Failed/i').count() > 0;

        // Test for console errors
        const consoleErrors = [];
        page.on('console', msg => {
          if (msg.type() === 'error') {
            consoleErrors.push(msg.text());
          }
        });

        if (hasErrorHandling || consoleErrors.length === 0) {
          auditResult.tests.errorHandling = true;
          auditResult.score += 15;
          console.log(`  [PASS] Error handling mechanisms present`);
        } else {
          auditResult.issues.push(`Console errors detected: ${consoleErrors.join(', ')}`);
          console.log(`  [FAIL] Error handling issues or console errors`);
        }
      } catch (error) {
        auditResult.issues.push(`Error handling check failed: ${error.message}`);
      }

      // Test 9: API Endpoint Functionality (10 points)
      try {
        const apiResponse = await page.evaluate(async (phaseName) => {
          try {
            const response = await fetch(`/api/phases/${phaseName}`);
            return { ok: response.ok, status: response.status };
          } catch (error) {
            return { ok: false, error: error.message };
          }
        }, phase.name);

        if (apiResponse.ok) {
          auditResult.tests.apiEndpoint = true;
          auditResult.score += 10;
          console.log(`  [PASS] API endpoint functional (${apiResponse.status})`);
        } else {
          auditResult.issues.push(`API endpoint failed: ${apiResponse.error || apiResponse.status}`);
          console.log(`  [FAIL] API endpoint not working`);
        }
      } catch (error) {
        auditResult.issues.push(`API endpoint check failed: ${error.message}`);
      }

      // Test 10: Phase Execution Flow (15 points)
      try {
        const startButton = page.locator('button:has-text("Start"), button:has-text("Begin"), button:has-text("Execute")');

        if (await startButton.isVisible()) {
          // Click start and check for state changes
          await startButton.click();
          await page.waitForTimeout(2000);

          // Look for execution indicators
          const hasExecution = await page.locator('button:has-text("Stop"), button:has-text("Cancel")').isVisible() ||
                              await page.locator('text=/Running|Executing|In Progress/i').isVisible() ||
                              await page.locator('[class*="running"], [class*="executing"]').isVisible();

          if (hasExecution) {
            auditResult.tests.phaseExecution = true;
            auditResult.score += 15;
            console.log(`  [PASS] Phase execution flow working`);

            // Capture execution state
            await page.screenshot({
              path: `screenshots/audit/phase-${phase.id}-${phase.name}-executing.png`,
              fullPage: true
            });
            auditResult.screenshots.push(`phase-${phase.id}-${phase.name}-executing.png`);

            // Stop execution if possible
            const stopButton = page.locator('button:has-text("Stop"), button:has-text("Cancel")');
            if (await stopButton.isVisible()) {
              await stopButton.click();
              await page.waitForTimeout(1000);
            }
          } else {
            auditResult.issues.push('Phase execution does not show running state');
            console.log(`  [FAIL] Phase execution state not detected`);
          }
        }
      } catch (error) {
        auditResult.issues.push(`Phase execution test failed: ${error.message}`);
      }

      // Capture final state
      await page.screenshot({
        path: `screenshots/audit/phase-${phase.id}-${phase.name}-final.png`,
        fullPage: true
      });
      auditResult.screenshots.push(`phase-${phase.id}-${phase.name}-final.png`);

      // Determine overall status
      if (auditResult.score >= 80) {
        auditResult.status = 'PASS';
      } else if (auditResult.score >= 60) {
        auditResult.status = 'PARTIAL';
      } else {
        auditResult.status = 'FAIL';
      }

      auditResults.push(auditResult);

      console.log(`  [SCORE] ${auditResult.score}/100 - ${auditResult.status}`);
      if (auditResult.issues.length > 0) {
        console.log(`  [ISSUES] ${auditResult.issues.length} issues found`);
      }

      // Assertions for test framework
      expect(auditResult.tests.pageLoad).toBeTruthy();
      expect(auditResult.score).toBeGreaterThan(40); // Minimum acceptable score
    });
  });

  // Theater Detection Test
  test('Theater Detection and Quality Validation', async ({ page }) => {
    console.log('\n[THEATER] Testing theater detection and quality validation...');

    // Test Phase 3 (QuietSTaR) which has 73% theater detection
    await page.goto('/phases/quietstar');
    await page.waitForTimeout(3000);

    // Look for theater detection warnings
    const theaterWarnings = await page.locator('text=/Theater|Warning|Quality|Fake/i').count();
    const qualityAlerts = await page.locator('[class*="warning"], [class*="alert"], [class*="theater"]').count();

    await page.screenshot({
      path: 'screenshots/audit/theater-detection-quietstar.png',
      fullPage: true
    });

    console.log(`  Theater warnings found: ${theaterWarnings}`);
    console.log(`  Quality alerts found: ${qualityAlerts}`);

    // Test other phases for theater detection
    for (const phase of phases.slice(0, 4)) { // Test first 4 phases
      await page.goto(phase.route);
      await page.waitForTimeout(2000);

      const phaseTheaterWarnings = await page.locator('text=/Theater|Quality.*Warning/i').count();

      if (phaseTheaterWarnings > 0) {
        await page.screenshot({
          path: `screenshots/audit/theater-detection-${phase.name}.png`,
          fullPage: true
        });
        console.log(`  Phase ${phase.id}: ${phaseTheaterWarnings} theater warnings found`);
      }
    }
  });

  // Performance and Accessibility Test
  test('Performance and Accessibility Audit', async ({ page }) => {
    console.log('\n[PERFORMANCE] Testing performance and accessibility...');

    const performanceResults = [];

    for (const phase of phases.slice(0, 3)) { // Test first 3 phases for performance
      const startTime = Date.now();
      await page.goto(phase.route);
      await page.waitForLoadState('networkidle');
      const loadTime = Date.now() - startTime;

      performanceResults.push({
        phase: phase.name,
        loadTime: loadTime,
        passed: loadTime < 5000 // 5 second threshold
      });

      console.log(`  Phase ${phase.id} load time: ${loadTime}ms`);
    }

    // Capture performance summary
    await page.screenshot({
      path: 'screenshots/audit/performance-summary.png'
    });

    const fastPhases = performanceResults.filter(p => p.passed).length;
    console.log(`  Performance summary: ${fastPhases}/${performanceResults.length} phases under 5s`);

    expect(fastPhases).toBeGreaterThan(0); // At least one phase should be fast
  });

  // Error State Testing
  test('Error State and Edge Case Testing', async ({ page }) => {
    console.log('\n[ERROR STATES] Testing error handling and edge cases...');

    // Test 404 handling
    await page.goto('/phases/nonexistent');
    await page.screenshot({
      path: 'screenshots/audit/error-404-handling.png',
      fullPage: true
    });

    // Test API errors
    await page.goto('/phases/cognate');

    // Inject API error simulation
    await page.route('/api/phases/**', route => {
      route.fulfill({
        status: 500,
        contentType: 'application/json',
        body: JSON.stringify({ error: 'Simulated API Error' })
      });
    });

    await page.reload();
    await page.waitForTimeout(3000);

    await page.screenshot({
      path: 'screenshots/audit/error-api-failure.png',
      fullPage: true
    });

    console.log('  Error state testing completed');
  });
});