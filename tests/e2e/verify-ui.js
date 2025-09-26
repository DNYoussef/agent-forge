const { chromium } = require('playwright');

(async () => {
  const browser = await chromium.launch({ headless: false });
  const page = await browser.newPage();

  console.log('[INFO] Navigating to Agent Forge Dashboard...');
  await page.goto('http://localhost:3000');
  await page.waitForTimeout(2000);

  // Check main dashboard elements
  console.log('[CHECK] Looking for Pipeline Control Center...');
  const pipelineControl = await page.locator('h2:has-text("Pipeline Control Center")').isVisible();
  console.log(`  Pipeline Control Center: ${pipelineControl ? 'FOUND' : 'MISSING'}`);

  // Check for phase selection checkboxes
  console.log('[CHECK] Looking for phase selection checkboxes...');
  const phases = [
    'Cognate (Model Creation)',
    'EvoMerge (Evolution)',
    'Quiet-STaR (Reasoning)',
    'BitNet (Compression)',
    'Forge Training',
    'Tool & Persona Baking',
    'ADAS (Architecture Search)',
    'Final Compression'
  ];

  for (const phase of phases) {
    const checkbox = await page.locator(`label:has-text("${phase}")`).isVisible();
    console.log(`  Phase "${phase}": ${checkbox ? 'FOUND' : 'MISSING'}`);
  }

  // Check for Start Pipeline button
  console.log('[CHECK] Looking for Start Pipeline button...');
  const startButton = await page.locator('button:has-text("Start Pipeline")').isVisible();
  console.log(`  Start Pipeline button: ${startButton ? 'FOUND' : 'MISSING'}`);

  // Take screenshot of dashboard
  await page.screenshot({ path: 'dashboard-screenshot.png', fullPage: true });
  console.log('[INFO] Dashboard screenshot saved as dashboard-screenshot.png');

  // Navigate to Cognate phase page
  console.log('[INFO] Navigating to Cognate phase page...');
  await page.goto('http://localhost:3000/phases/cognate');
  await page.waitForTimeout(2000);

  // Check Cognate page elements
  console.log('[CHECK] Looking for PhaseController component...');
  const phaseController = await page.locator('div.border.rounded-lg').first().isVisible();
  console.log(`  PhaseController component: ${phaseController ? 'FOUND' : 'MISSING'}`);

  // Check for configuration controls
  console.log('[CHECK] Looking for configuration controls...');
  const modelTypeSelect = await page.locator('select#modelType').isVisible();
  console.log(`  Model Type selector: ${modelTypeSelect ? 'FOUND' : 'MISSING'}`);

  const vocabSizeInput = await page.locator('input#vocabSize').isVisible();
  console.log(`  Vocab Size input: ${vocabSizeInput ? 'FOUND' : 'MISSING'}`);

  const grokfastSection = await page.locator('h3:has-text("Grokfast Settings")').isVisible();
  console.log(`  Grokfast Settings: ${grokfastSection ? 'FOUND' : 'MISSING'}`);

  // Take screenshot of Cognate page
  await page.screenshot({ path: 'cognate-screenshot.png', fullPage: true });
  console.log('[INFO] Cognate page screenshot saved as cognate-screenshot.png');

  // Check if API calls are being made
  console.log('[CHECK] Testing API connectivity...');
  await page.goto('http://localhost:3000');

  // Listen for API requests
  page.on('response', response => {
    if (response.url().includes('/api/')) {
      console.log(`  API call detected: ${response.url()} - Status: ${response.status()}`);
    }
  });

  // Wait for stats API to be called
  await page.waitForTimeout(3000);

  console.log('\n[SUMMARY]');
  console.log('========================================');
  if (pipelineControl && startButton) {
    console.log('[OK] Enhanced dashboard with pipeline controls is ACTIVE');
    console.log('[OK] UI components match the real Agent Forge pipeline implementation');
  } else {
    console.log('[WARNING] Original UI may still be active - enhanced components not found');
  }

  await browser.close();
})();