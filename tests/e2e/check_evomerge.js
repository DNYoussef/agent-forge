const { chromium } = require('playwright');

async function checkEvoMerge() {
    console.log('🔍 Checking EvoMerge UI...\n');

    const browser = await chromium.launch({
        headless: false,
        args: ['--start-maximized']
    });

    const context = await browser.newContext({
        viewport: { width: 1920, height: 1080 }
    });

    const page = await context.newPage();

    try {
        // Go directly to EvoMerge page
        console.log('✅ Navigating to EvoMerge page...');
        await page.goto('http://localhost:3000/phases/evomerge', {
            waitUntil: 'domcontentloaded',
            timeout: 10000
        });

        await page.waitForTimeout(2000);

        // Take screenshot
        await page.screenshot({
            path: 'evomerge-current.png',
            fullPage: true
        });
        console.log('   Screenshot saved: evomerge-current.png');

        // Check for tournament elements
        const hasTitle = await page.locator('text=/Phase 2.*EvoMerge/i').count() > 0;
        console.log(`   ✅ EvoMerge title found: ${hasTitle}`);

        const hasTournament = await page.locator('text=/tournament/i').count() > 0;
        console.log(`   ✅ Tournament text found: ${hasTournament}`);

        const hasPopulation = await page.locator('text=/Population/i').count() > 0;
        console.log(`   ✅ Population text found: ${hasPopulation}`);

        // Look for specific tournament rules
        const hasWinners = await page.locator('text=/Top 2/i').count() > 0;
        console.log(`   ✅ "Top 2" winners text found: ${hasWinners}`);

        const hasLosers = await page.locator('text=/Bottom 6/i').count() > 0;
        console.log(`   ✅ "Bottom 6" losers text found: ${hasLosers}`);

        console.log('\n✨ Check complete! Review evomerge-current.png');

    } catch (error) {
        console.error('❌ Error:', error.message);
        await page.screenshot({
            path: 'evomerge-error.png',
            fullPage: true
        });
    } finally {
        await browser.close();
    }
}

checkEvoMerge().catch(console.error);