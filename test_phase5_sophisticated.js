const { chromium } = require('playwright');

(async () => {
  const browser = await chromium.launch({
    headless: false,
    args: ['--window-size=1920,1080']
  });
  const context = await browser.newContext({
    viewport: { width: 1920, height: 1080 },
    deviceScaleFactor: 1
  });
  const page = await context.newPage();

  try {
    console.log('Navigating to Phase 5 Forge page...');
    await page.goto('http://localhost:3000/phases/forge', { waitUntil: 'networkidle' });
    await page.waitForTimeout(3000);

    // Capture homepage with 10-level progression
    console.log('Capturing Phase 5 homepage with START button and 10-level system...');
    await page.screenshot({
      path: 'phase5-homepage-10levels.png',
      fullPage: true
    });

    // Click START button
    const startButton = await page.locator('button:has-text("START PHASE 5 TRAINING")').first();
    if (await startButton.isVisible()) {
      console.log('Clicking START button...');
      await startButton.click();
      await page.waitForTimeout(2000);

      // Capture training interface
      console.log('Capturing sophisticated training interface...');
      await page.screenshot({
        path: 'phase5-training-interface.png',
        fullPage: true
      });

      // Test Assessment tab
      const assessmentTab = await page.locator('button:has-text("Assessment")').first();
      if (await assessmentTab.isVisible()) {
        await assessmentTab.click();
        await page.waitForTimeout(1000);
        console.log('Capturing Assessment interface...');
        await page.screenshot({
          path: 'phase5-assessment-tab.png',
          fullPage: true
        });
      }

      // Test Training Loop tab
      const trainingTab = await page.locator('button:has-text("Training Loop")').first();
      if (await trainingTab.isVisible()) {
        await trainingTab.click();
        await page.waitForTimeout(1000);
        console.log('Capturing Training Loop interface...');
        await page.screenshot({
          path: 'phase5-training-loop-tab.png',
          fullPage: true
        });
      }

      // Test Self-Modeling tab
      const selfModelTab = await page.locator('button:has-text("Self-Modeling")').first();
      if (await selfModelTab.isVisible()) {
        await selfModelTab.click();
        await page.waitForTimeout(1000);
        console.log('Capturing Self-Modeling interface...');
        await page.screenshot({
          path: 'phase5-self-modeling-tab.png',
          fullPage: true
        });
      }

      // Test Sleep/Dream tab
      const sleepTab = await page.locator('button:has-text("Sleep/Dream")').first();
      if (await sleepTab.isVisible()) {
        await sleepTab.click();
        await page.waitForTimeout(1000);
        console.log('Capturing Sleep/Dream interface...');
        await page.screenshot({
          path: 'phase5-sleep-dream-tab.png',
          fullPage: true
        });
      }

      // Test Weight Space tab
      const weightTab = await page.locator('button:has-text("Weight Space")').first();
      if (await weightTab.isVisible()) {
        await weightTab.click();
        await page.waitForTimeout(3000); // Extra time for 3D rendering
        console.log('Capturing Weight Space Visualization...');
        await page.screenshot({
          path: 'phase5-weight-space-tab.png',
          fullPage: true
        });
      }
    }

    console.log('Phase 5 sophisticated interface testing complete!');
    console.log('Screenshots saved:');
    console.log('- phase5-homepage-10levels.png');
    console.log('- phase5-training-interface.png');
    console.log('- phase5-assessment-tab.png');
    console.log('- phase5-training-loop-tab.png');
    console.log('- phase5-self-modeling-tab.png');
    console.log('- phase5-sleep-dream-tab.png');
    console.log('- phase5-weight-space-tab.png');

  } catch (error) {
    console.error('Error:', error);
  } finally {
    await browser.close();
  }
})();