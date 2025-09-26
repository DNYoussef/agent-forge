const { test, expect } = require('@playwright/test');

/**
 * Performance Metrics and Theater Detection Test Suite
 * Measures performance, detects quality issues, and validates real implementation
 */

test.describe('Performance and Quality Metrics', () => {
  let performanceData = [];
  let theaterFindings = [];

  test('Performance Benchmark - All Phases', async ({ page }) => {
    console.log('\n[PERFORMANCE] Starting comprehensive performance testing...');

    const phases = [
      { id: 1, name: 'cognate', route: '/phases/cognate' },
      { id: 2, name: 'evomerge', route: '/phases/evomerge' },
      { id: 3, name: 'quietstar', route: '/phases/quietstar' },
      { id: 4, name: 'bitnet', route: '/phases/bitnet' },
      { id: 5, name: 'forge', route: '/phases/forge' },
      { id: 6, name: 'baking', route: '/phases/baking' },
      { id: 7, name: 'adas', route: '/phases/adas' },
      { id: 8, name: 'final', route: '/phases/final' }
    ];

    for (const phase of phases) {
      console.log(`\n[PERF] Testing Phase ${phase.id}: ${phase.name}`);

      // Measure page load performance
      const loadStart = Date.now();
      await page.goto(phase.route);

      // Wait for essential elements
      await page.waitForSelector('h1', { timeout: 10000 }).catch(() => null);
      await page.waitForLoadState('networkidle', { timeout: 15000 }).catch(() => null);

      const loadTime = Date.now() - loadStart;

      // Measure DOM complexity
      const domStats = await page.evaluate(() => ({
        totalElements: document.querySelectorAll('*').length,
        images: document.querySelectorAll('img').length,
        scripts: document.querySelectorAll('script').length,
        stylesheets: document.querySelectorAll('link[rel="stylesheet"]').length
      }));

      // Measure JavaScript errors
      const jsErrors = [];
      page.on('console', msg => {
        if (msg.type() === 'error') {
          jsErrors.push(msg.text());
        }
      });

      // Measure memory usage (approximation)
      const memoryInfo = await page.evaluate(() => {
        if (performance.memory) {
          return {
            usedJSHeapSize: performance.memory.usedJSHeapSize,
            totalJSHeapSize: performance.memory.totalJSHeapSize,
            jsHeapSizeLimit: performance.memory.jsHeapSizeLimit
          };
        }
        return null;
      });

      // Test interaction responsiveness
      const interactionStart = Date.now();
      try {
        const button = page.locator('button').first();
        if (await button.isVisible()) {
          await button.hover();
        }
      } catch (e) {
        // Interaction test failed, but continue
      }
      const interactionTime = Date.now() - interactionStart;

      const phasePerformance = {
        phase: phase.name,
        phaseId: phase.id,
        loadTime: loadTime,
        domElements: domStats.totalElements,
        images: domStats.images,
        scripts: domStats.scripts,
        stylesheets: domStats.stylesheets,
        jsErrors: jsErrors.length,
        errorMessages: jsErrors,
        interactionTime: interactionTime,
        memoryUsage: memoryInfo,
        grade: getPerformanceGrade(loadTime, domStats.totalElements, jsErrors.length)
      };

      performanceData.push(phasePerformance);

      console.log(`  Load time: ${loadTime}ms`);
      console.log(`  DOM elements: ${domStats.totalElements}`);
      console.log(`  JS errors: ${jsErrors.length}`);
      console.log(`  Grade: ${phasePerformance.grade}`);

      // Capture performance screenshot
      await page.screenshot({
        path: `screenshots/audit/performance-${phase.name}-metrics.png`,
        fullPage: true
      });

      // Small delay between tests
      await page.waitForTimeout(1000);
    }

    // Generate performance report
    generatePerformanceReport(performanceData);
  });

  test('Theater Detection - Quality Validation', async ({ page }) => {
    console.log('\n[THEATER] Starting theater detection and quality validation...');

    // Phase 3 (QuietSTaR) has known theater issues (73% theater score)
    await page.goto('/phases/quietstar');
    await page.waitForTimeout(3000);

    // Look for theater indicators
    const theaterIndicators = await page.evaluate(() => {
      const indicators = [];

      // Check for fake progress bars or animations without real backing
      const progressBars = document.querySelectorAll('[role="progressbar"], .progress-bar, .loading');
      progressBars.forEach((bar, index) => {
        // Check if progress bar has actual data binding
        const hasDataBinding = bar.getAttribute('aria-valuenow') ||
                              bar.style.width ||
                              bar.dataset.progress;
        if (!hasDataBinding) {
          indicators.push(`Fake progress bar detected: element ${index}`);
        }
      });

      // Check for placeholder text that suggests incomplete implementation
      const placeholderText = document.body.innerHTML.match(/TODO|PLACEHOLDER|Lorem ipsum|Coming soon|Under construction/gi);
      if (placeholderText) {
        indicators.push(`Placeholder content found: ${placeholderText.join(', ')}`);
      }

      // Check for non-functional buttons
      const buttons = document.querySelectorAll('button');
      let nonFunctionalButtons = 0;
      buttons.forEach(button => {
        const hasOnClick = button.onclick ||
                          button.getAttribute('onclick') ||
                          button.hasAttribute('disabled') ||
                          button.closest('form');
        if (!hasOnClick && button.textContent.trim()) {
          nonFunctionalButtons++;
        }
      });

      if (nonFunctionalButtons > 0) {
        indicators.push(`Non-functional buttons detected: ${nonFunctionalButtons}`);
      }

      // Check for console warnings about missing implementations
      return indicators;
    });

    theaterFindings.push({
      phase: 'quietstar',
      phaseId: 3,
      indicators: theaterIndicators,
      score: calculateTheaterScore(theaterIndicators),
      timestamp: new Date().toISOString()
    });

    console.log(`  Theater indicators found: ${theaterIndicators.length}`);
    theaterIndicators.forEach(indicator => {
      console.log(`    - ${indicator}`);
    });

    // Capture theater detection evidence
    await page.screenshot({
      path: 'screenshots/audit/theater-detection-evidence.png',
      fullPage: true
    });

    // Test other high-risk phases
    const highRiskPhases = ['cognate', 'evomerge', 'bitnet'];

    for (const phaseName of highRiskPhases) {
      await page.goto(`/phases/${phaseName}`);
      await page.waitForTimeout(2000);

      const phaseTheaterIndicators = await page.evaluate(() => {
        const indicators = [];

        // Check for mock data or hardcoded values
        const textContent = document.body.textContent;
        if (textContent.includes('mock') || textContent.includes('sample') || textContent.includes('demo')) {
          indicators.push('Mock/demo content detected');
        }

        // Check for missing API integrations
        const forms = document.querySelectorAll('form');
        forms.forEach(form => {
          if (!form.action && !form.onsubmit) {
            indicators.push('Form without action detected');
          }
        });

        return indicators;
      });

      if (phaseTheaterIndicators.length > 0) {
        theaterFindings.push({
          phase: phaseName,
          indicators: phaseTheaterIndicators,
          score: calculateTheaterScore(phaseTheaterIndicators)
        });

        await page.screenshot({
          path: `screenshots/audit/theater-${phaseName}-evidence.png`,
          fullPage: true
        });
      }
    }

    generateTheaterReport(theaterFindings);
  });

  test('Reality Validation - Implementation Check', async ({ page }) => {
    console.log('\n[REALITY] Validating real implementation vs theater...');

    const realityChecks = [];

    // Test API endpoints actually exist
    const phases = ['cognate', 'evomerge', 'quietstar', 'bitnet'];

    for (const phaseName of phases) {
      const apiCheck = await page.evaluate(async (phase) => {
        try {
          const response = await fetch(`/api/phases/${phase}`);
          return {
            exists: response.status !== 404,
            status: response.status,
            hasContent: response.status === 200
          };
        } catch (error) {
          return {
            exists: false,
            error: error.message
          };
        }
      }, phaseName);

      realityChecks.push({
        phase: phaseName,
        apiExists: apiCheck.exists,
        apiStatus: apiCheck.status,
        hasRealData: apiCheck.hasContent
      });

      console.log(`  API /api/phases/${phaseName}: ${apiCheck.exists ? 'EXISTS' : 'MISSING'} (${apiCheck.status})`);
    }

    // Test actual functionality vs mock functionality
    await page.goto('/phases/cognate');
    await page.waitForTimeout(2000);

    const functionalityTest = await page.evaluate(() => {
      const startButton = document.querySelector('button:contains("Start")') ||
                         document.querySelector('button[id*="start"], button[class*="start"]') ||
                         document.querySelector('button');

      if (startButton) {
        // Check if button has real event handlers
        const hasRealHandler = startButton.onclick ||
                              startButton.addEventListener ||
                              startButton.hasAttribute('onclick') ||
                              startButton.closest('form');

        return {
          hasStartButton: true,
          hasRealFunctionality: !!hasRealHandler,
          buttonText: startButton.textContent.trim()
        };
      }

      return {
        hasStartButton: false,
        hasRealFunctionality: false
      };
    });

    console.log(`  Start button functionality: ${functionalityTest.hasRealFunctionality ? 'REAL' : 'THEATER'}`);

    // Generate reality validation report
    generateRealityReport(realityChecks, functionalityTest);
  });

  // Helper function to calculate performance grade
  function getPerformanceGrade(loadTime, domElements, jsErrors) {
    let score = 100;

    // Deduct points for slow loading
    if (loadTime > 5000) score -= 30;
    else if (loadTime > 3000) score -= 15;
    else if (loadTime > 1000) score -= 5;

    // Deduct points for DOM complexity
    if (domElements > 1000) score -= 20;
    else if (domElements > 500) score -= 10;

    // Deduct points for JavaScript errors
    score -= jsErrors * 10;

    if (score >= 90) return 'A';
    if (score >= 80) return 'B';
    if (score >= 70) return 'C';
    if (score >= 60) return 'D';
    return 'F';
  }

  // Helper function to calculate theater score
  function calculateTheaterScore(indicators) {
    return Math.min(100, indicators.length * 15); // Each indicator adds 15 points to theater score
  }

  // Helper function to generate performance report
  function generatePerformanceReport(data) {
    console.log('\n' + '=' .repeat(80));
    console.log('PERFORMANCE REPORT');
    console.log('=' .repeat(80));

    const avgLoadTime = data.reduce((sum, d) => sum + d.loadTime, 0) / data.length;
    const slowPhases = data.filter(d => d.loadTime > 3000).length;
    const errorPhases = data.filter(d => d.jsErrors > 0).length;

    console.log(`Average load time: ${Math.round(avgLoadTime)}ms`);
    console.log(`Slow phases (>3s): ${slowPhases}/${data.length}`);
    console.log(`Phases with JS errors: ${errorPhases}/${data.length}`);

    data.forEach(phase => {
      console.log(`Phase ${phase.phaseId} (${phase.phase}): ${phase.loadTime}ms - Grade ${phase.grade}`);
    });
  }

  // Helper function to generate theater report
  function generateTheaterReport(findings) {
    console.log('\n' + '=' .repeat(80));
    console.log('THEATER DETECTION REPORT');
    console.log('=' .repeat(80));

    const totalFindings = findings.reduce((sum, f) => sum + f.indicators.length, 0);
    const avgTheaterScore = findings.reduce((sum, f) => sum + f.score, 0) / findings.length;

    console.log(`Total theater indicators: ${totalFindings}`);
    console.log(`Average theater score: ${Math.round(avgTheaterScore)}%`);

    findings.forEach(finding => {
      console.log(`Phase ${finding.phase}: ${finding.score}% theater (${finding.indicators.length} indicators)`);
    });
  }

  // Helper function to generate reality report
  function generateRealityReport(apiChecks, functionalityTest) {
    console.log('\n' + '=' .repeat(80));
    console.log('REALITY VALIDATION REPORT');
    console.log('=' .repeat(80));

    const workingAPIs = apiChecks.filter(check => check.apiExists).length;
    console.log(`Working APIs: ${workingAPIs}/${apiChecks.length}`);
    console.log(`Button functionality: ${functionalityTest.hasRealFunctionality ? 'REAL' : 'THEATER'}`);

    apiChecks.forEach(check => {
      console.log(`${check.phase}: API ${check.apiExists ? 'EXISTS' : 'MISSING'}`);
    });
  }
});