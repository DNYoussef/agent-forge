const { chromium } = require('playwright');
const fs = require('fs');
const path = require('path');

async function captureAllScreenshots() {
  const browser = await chromium.launch({ headless: false });
  const page = await browser.newPage();

  // Set viewport for consistent screenshots
  await page.setViewportSize({ width: 1920, height: 1080 });

  console.log('📸 Capturing Agent Forge UI Screenshots...\n');

  // Create screenshots directory if it doesn't exist
  const screenshotsDir = path.join(__dirname, 'docs', 'screenshots');
  if (!fs.existsSync(screenshotsDir)) {
    fs.mkdirSync(screenshotsDir, { recursive: true });
  }

  const pages = [
    { name: 'dashboard', url: 'http://localhost:3000', title: 'Agent Forge Dashboard' },
    { name: 'phase1-cognate', url: 'http://localhost:3000/phases/cognate', title: 'Phase 1: Cognate' },
    { name: 'phase2-evomerge', url: 'http://localhost:3000/phases/evomerge', title: 'Phase 2: EvoMerge' },
    { name: 'phase3-quietstar', url: 'http://localhost:3000/phases/quietstar', title: 'Phase 3: Quiet-STaR' },
    { name: 'phase4-bitnet', url: 'http://localhost:3000/phases/bitnet', title: 'Phase 4: BitNet' },
    { name: 'phase5-forge', url: 'http://localhost:3000/phases/forge', title: 'Phase 5: Forge Training' },
    { name: 'phase6-baking', url: 'http://localhost:3000/phases/baking', title: 'Phase 6: Tool & Persona Baking' },
    { name: 'phase7-adas', url: 'http://localhost:3000/phases/adas', title: 'Phase 7: ADAS' },
    { name: 'phase8-final', url: 'http://localhost:3000/phases/final', title: 'Phase 8: Final Compression' }
  ];

  for (const pageInfo of pages) {
    console.log(`Capturing ${pageInfo.title}...`);

    try {
      await page.goto(pageInfo.url, { waitUntil: 'networkidle', timeout: 30000 });
      await page.waitForTimeout(2000); // Wait for animations to settle

      const screenshotPath = path.join(screenshotsDir, `${pageInfo.name}.png`);
      await page.screenshot({
        path: screenshotPath,
        fullPage: false // Just viewport for consistent sizing
      });

      console.log(`  ✅ Saved: docs/screenshots/${pageInfo.name}.png`);
    } catch (error) {
      console.log(`  ❌ Error capturing ${pageInfo.title}: ${error.message}`);
    }
  }

  console.log('\n✨ All screenshots captured!');
  await browser.close();
}

captureAllScreenshots().catch(console.error);