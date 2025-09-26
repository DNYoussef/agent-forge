const { chromium } = require('playwright');

async function testAllPhasesConnected() {
  const browser = await chromium.launch({ headless: false });
  const page = await browser.newPage();

  console.log('[U+1F680] AGENT FORGE PIPELINE - FINAL VERIFICATION TEST');
  console.log('=' .repeat(80));

  const phases = [
    { id: 1, name: 'cognate', title: 'Cognate (Model Creation)' },
    { id: 2, name: 'evomerge', title: 'EvoMerge (Evolution)' },
    { id: 3, name: 'quietstar', title: 'Quiet-STaR (Reasoning)' },
    { id: 4, name: 'bitnet', title: 'BitNet (Compression)' },
    { id: 5, name: 'forge', title: 'Forge Training' },
    { id: 6, name: 'baking', title: 'Tool & Persona Baking' },
    { id: 7, name: 'adas', title: 'ADAS (Architecture Search)' },
    { id: 8, name: 'final', title: 'Final Compression' }
  ];

  const testResults = [];

  for (const phase of phases) {
    console.log(`\n[PHASE ${phase.id}] Testing ${phase.title}...`);
    console.log('-' .repeat(60));

    await page.goto(`http://localhost:3000/phases/${phase.name}`);
    await page.waitForTimeout(2000);

    const result = {
      phase: phase.name,
      phaseId: phase.id,
      title: phase.title,
      tests: {
        pageLoads: false,
        hasConfigControls: false,
        hasPhaseController: false,
        hasStartButton: false,
        hasMetricsDisplay: false,
        apiWorks: false,
        canStart: false
      }
    };

    // Test 1: Page loads successfully
    try {
      await page.waitForSelector('h1', { timeout: 5000 });
      result.tests.pageLoads = true;
      console.log('  [OK] Page loads');
    } catch {
      console.log('  [FAIL] Page does not load');
    }

    // Test 2: Has configuration controls (not 3D animation)
    try {
      const hasControls = await page.locator('input[type="range"], select, input[type="checkbox"]')
        .first()
        .isVisible()
        .catch(() => false);
      result.tests.hasConfigControls = hasControls;
      console.log(hasControls ? '  [OK] Configuration controls found' : '  [FAIL] No configuration controls');
    } catch {
      console.log('  [FAIL] Error checking controls');
    }

    // Test 3: Has PhaseController component
    try {
      const hasController = await page.locator('text=/Phase.*Status/i')
        .isVisible()
        .catch(() => false);
      result.tests.hasPhaseController = hasController;
      console.log(hasController ? '  [OK] PhaseController component found' : '  [FAIL] No PhaseController');
    } catch {
      console.log('  [FAIL] Error checking PhaseController');
    }

    // Test 4: Has Start button
    try {
      const hasStart = await page.locator('button:has-text("Start")')
        .isVisible()
        .catch(() => false);
      result.tests.hasStartButton = hasStart;
      console.log(hasStart ? '  [OK] Start button found' : '  [FAIL] No Start button');
    } catch {
      console.log('  [FAIL] Error checking Start button');
    }

    // Test 5: Has metrics display area
    try {
      const hasMetrics = await page.locator('text=/Metrics|Progress|Performance/i')
        .first()
        .isVisible()
        .catch(() => false);
      result.tests.hasMetricsDisplay = hasMetrics;
      console.log(hasMetrics ? '  [OK] Metrics display area found' : '  [FAIL] No metrics display');
    } catch {
      console.log('  [FAIL] Error checking metrics display');
    }

    // Test 6: API endpoint works
    try {
      const response = await page.evaluate(async (phaseName) => {
        try {
          const res = await fetch(`/api/phases/${phaseName}`);
          return res.ok;
        } catch {
          return false;
        }
      }, phase.name);
      result.tests.apiWorks = response;
      console.log(response ? '  [OK] API endpoint responds' : '  [FAIL] API endpoint not working');
    } catch {
      console.log('  [FAIL] Error testing API');
    }

    // Test 7: Can start phase (click Start and check for session)
    if (result.tests.hasStartButton) {
      try {
        await page.click('button:has-text("Start")');
        await page.waitForTimeout(1000);

        // Check if Stop button appears (indicates phase started)
        const hasStop = await page.locator('button:has-text("Stop")')
          .isVisible()
          .catch(() => false);
        result.tests.canStart = hasStop;
        console.log(hasStop ? '  [OK] Phase can be started' : '  [FAIL] Phase does not start');

        // Stop the phase if started
        if (hasStop) {
          await page.click('button:has-text("Stop")');
          await page.waitForTimeout(500);
        }
      } catch {
        console.log('  [FAIL] Error starting phase');
      }
    }

    testResults.push(result);

    // Take screenshot
    await page.screenshot({
      path: `test-phase-${phase.id}-${phase.name}.png`,
      fullPage: true
    });
  }

  // Generate summary report
  console.log('\n\n' + '=' .repeat(80));
  console.log('FINAL TEST REPORT - AGENT FORGE PIPELINE CONNECTION');
  console.log('=' .repeat(80));

  let fullyWorking = 0;
  let partiallyWorking = 0;
  let notWorking = 0;

  for (const result of testResults) {
    const passedTests = Object.values(result.tests).filter(t => t).length;
    const totalTests = Object.values(result.tests).length;
    const percentage = Math.round((passedTests / totalTests) * 100);

    console.log(`\nPhase ${result.phaseId}: ${result.title}`);
    console.log(`  Tests Passed: ${passedTests}/${totalTests} (${percentage}%)`);

    if (passedTests === totalTests) {
      console.log(`  Status: [U+2713] FULLY CONNECTED`);
      fullyWorking++;
    } else if (passedTests >= 4) {
      console.log(`  Status: [U+26A0] PARTIALLY CONNECTED`);
      partiallyWorking++;
    } else {
      console.log(`  Status: [U+274C] NOT CONNECTED`);
      notWorking++;
    }

    // Show failing tests
    const failedTests = Object.entries(result.tests)
      .filter(([_, passed]) => !passed)
      .map(([test, _]) => test);

    if (failedTests.length > 0) {
      console.log(`  Failed Tests: ${failedTests.join(', ')}`);
    }
  }

  console.log('\n' + '=' .repeat(80));
  console.log('OVERALL SUMMARY:');
  console.log(`  Fully Connected: ${fullyWorking}/8 phases`);
  console.log(`  Partially Connected: ${partiallyWorking}/8 phases`);
  console.log(`  Not Connected: ${notWorking}/8 phases`);

  if (fullyWorking === 8) {
    console.log('\n[U+1F389] SUCCESS! All 8 phases are fully connected to the Agent Forge pipeline!');
  } else if (fullyWorking >= 6) {
    console.log('\n[U+1F44D] GOOD PROGRESS! Most phases are connected, some need minor fixes.');
  } else {
    console.log('\n[U+26A0] NEEDS WORK! Several phases still need to be properly connected.');
  }

  console.log('=' .repeat(80));

  await browser.close();
}

// Run the test
testAllPhasesConnected()
  .then(() => console.log('\n[COMPLETE] Test finished successfully'))
  .catch(console.error);