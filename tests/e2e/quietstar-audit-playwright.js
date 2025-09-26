const { chromium } = require('playwright');
const fs = require('fs');
const path = require('path');

async function captureQuietStarAudit() {
  console.log('🎭 Starting Playwright audit of Quiet-STaR Phase...');

  const browser = await chromium.launch({
    headless: false,
    slowMo: 1000 // Slow down for better visibility
  });

  const page = await browser.newPage();
  page.setViewportSize({ width: 1920, height: 1080 });

  try {
    // Navigate to the Quiet-STaR phase
    console.log('📍 Navigating to Quiet-STaR phase...');
    await page.goto('http://localhost:3000/phases/quietstar', {
      waitUntil: 'networkidle',
      timeout: 30000
    });

    // Wait for the page to fully load
    await page.waitForTimeout(3000);

    // Capture the full page screenshot
    console.log('📸 Capturing full page screenshot...');
    const screenshotPath = path.join(__dirname, 'quietstar-audit-after.png');
    await page.screenshot({
      path: screenshotPath,
      fullPage: true
    });

    console.log(`✅ Screenshot saved: ${screenshotPath}`);

    // Test the configuration controls
    console.log('🔧 Testing configuration controls...');

    // Change convergence threshold
    const thresholdSlider = page.locator('input[type="range"]').first();
    await thresholdSlider.fill('0.90');
    await page.waitForTimeout(1000);

    // Toggle features
    const grokfastToggle = page.locator('input[type="checkbox"]').nth(2);
    await grokfastToggle.uncheck();
    await page.waitForTimeout(1000);
    await grokfastToggle.check();
    await page.waitForTimeout(1000);

    // Change cognitive strategy
    const strategySelect = page.locator('select').first();
    await strategySelect.selectOption('first_principles');
    await page.waitForTimeout(1000);

    // Capture interactive state screenshot
    const interactiveScreenshot = path.join(__dirname, 'quietstar-interactive-test.png');
    await page.screenshot({
      path: interactiveScreenshot,
      fullPage: true
    });

    console.log(`✅ Interactive test screenshot saved: ${interactiveScreenshot}`);

    // Test component presence
    console.log('🧩 Verifying component presence...');

    const components = {
      'Prompt Baking Progress': 'text=Prompt Baking Accelerator',
      'Configuration Panel': 'text=Configuration',
      'Thought Generator': 'text=Thought Generation',
      'Attention Visualizer': 'text=Attention Weight Visualization',
      'Baking Metrics': 'text=Baking Metrics'
    };

    const componentResults = {};

    for (const [componentName, selector] of Object.entries(components)) {
      try {
        await page.waitForSelector(selector, { timeout: 5000 });
        componentResults[componentName] = '✅ Present';
        console.log(`  ✅ ${componentName}: Found`);
      } catch (e) {
        componentResults[componentName] = '❌ Missing';
        console.log(`  ❌ ${componentName}: Not found`);
      }
    }

    // Check for API endpoints
    console.log('🔗 Testing API connectivity...');

    let apiStatus = 'Unknown';
    try {
      const response = await page.request.get('http://localhost:8001/api/phases/quietstar/status');
      if (response.ok()) {
        apiStatus = '✅ Connected';
        console.log('  ✅ Python API: Connected');
      } else {
        apiStatus = '⚠️ Not responding';
        console.log('  ⚠️ Python API: Not responding');
      }
    } catch (e) {
      apiStatus = '❌ Failed';
      console.log('  ❌ Python API: Connection failed');
    }

    // Generate audit report
    const auditReport = {
      timestamp: new Date().toISOString(),
      url: 'http://localhost:3000/phases/quietstar',
      screenshots: [
        screenshotPath,
        interactiveScreenshot
      ],
      components: componentResults,
      api_status: apiStatus,
      summary: {
        total_components: Object.keys(components).length,
        found_components: Object.values(componentResults).filter(v => v.includes('✅')).length,
        missing_components: Object.values(componentResults).filter(v => v.includes('❌')).length
      }
    };

    // Save audit report
    const reportPath = path.join(__dirname, 'quietstar-audit-report.json');
    fs.writeFileSync(reportPath, JSON.stringify(auditReport, null, 2));

    console.log(`📊 Audit report saved: ${reportPath}`);
    console.log('\n🎯 AUDIT SUMMARY:');
    console.log(`   Components Found: ${auditReport.summary.found_components}/${auditReport.summary.total_components}`);
    console.log(`   API Status: ${apiStatus}`);
    console.log(`   Screenshots: ${auditReport.screenshots.length} captured`);

  } catch (error) {
    console.error('❌ Error during audit:', error);
  } finally {
    await browser.close();
    console.log('🏁 Playwright audit completed!');
  }
}

// Run the audit
captureQuietStarAudit().catch(console.error);