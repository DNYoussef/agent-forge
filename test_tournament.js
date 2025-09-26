const { chromium } = require('playwright');

async function testTournamentPage() {
    console.log('🎯 Testing Tournament Page...\n');

    const browser = await chromium.launch({
        headless: false,
        args: ['--start-maximized']
    });

    const context = await browser.newContext({
        viewport: { width: 1920, height: 1080 }
    });

    const page = await context.newPage();

    try {
        // Test the tournament-specific page
        console.log('✅ Testing tournament-page route...');
        await page.goto('http://localhost:3000/phases/evomerge/tournament-page', {
            waitUntil: 'domcontentloaded',
            timeout: 10000
        });

        await page.waitForTimeout(2000);

        // Take screenshot
        await page.screenshot({
            path: 'tournament-page-test.png',
            fullPage: true
        });
        console.log('   Screenshot saved: tournament-page-test.png');

        // Check for tournament-specific elements
        const hasTitle = await page.locator('text=/Tournament Evolution/i').count() > 0;
        console.log(`   ✅ Tournament Evolution title found: ${hasTitle}`);

        const hasRules = await page.locator('text=/Tournament Selection Algorithm/i').count() > 0;
        console.log(`   ✅ Tournament rules section found: ${hasRules}`);

        const hasWinners = await page.locator('text=/Top 2 Winners/i').count() > 0;
        console.log(`   ✅ "Top 2 Winners" found: ${hasWinners}`);

        const hasLosers = await page.locator('text=/Bottom 6.*Chaos/i').count() > 0;
        console.log(`   ✅ "Bottom 6 Chaos" found: ${hasLosers}`);

        const hasEvolutionTree = await page.locator('text=/Evolution Tree/i').count() > 0;
        console.log(`   ✅ Evolution tree found: ${hasEvolutionTree}`);

        console.log('\n✨ Tournament page test complete!');

    } catch (error) {
        console.error('❌ Error:', error.message);
        await page.screenshot({
            path: 'tournament-error.png',
            fullPage: true
        });
    } finally {
        await browser.close();
    }
}

testTournamentPage().catch(console.error);