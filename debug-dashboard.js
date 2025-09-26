const { chromium } = require('playwright');

(async () => {
  const browser = await chromium.launch({ headless: false });
  const context = await browser.newContext();
  const page = await context.newPage();

  page.on('console', msg => console.log('BROWSER LOG:', msg.text()));
  page.on('pageerror', error => console.error('PAGE ERROR:', error));

  try {
    await page.goto('http://localhost:3000', { waitUntil: 'networkidle', timeout: 30000 });
    
    await page.screenshot({ path: 'C:/Users/17175/Desktop/agent-forge/debug-screenshot.png', fullPage: true });
    console.log('Screenshot saved');

    const errors = await page.evaluate(() => {
      return window.__errors || [];
    });
    console.log('Runtime errors:', errors);

    const content = await page.content();
    console.log('Page loaded successfully');
    
    await page.waitForTimeout(5000);
  } catch (error) {
    console.error('Error during test:', error.message);
    await page.screenshot({ path: 'C:/Users/17175/Desktop/agent-forge/error-screenshot.png' });
  } finally {
    await browser.close();
  }
})();
