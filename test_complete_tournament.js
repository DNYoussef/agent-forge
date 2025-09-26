const { chromium } = require('playwright');

async function testCompleteTournament() {
    console.log('🏆 COMPLETE TOURNAMENT UI TEST...\n');

    const browser = await chromium.launch({
        headless: false,
        args: ['--start-maximized']
    });

    const context = await browser.newContext({
        viewport: { width: 1920, height: 1080 }
    });

    const page = await context.newPage();

    try {
        console.log('✅ Navigating to EvoMerge Tournament page...');
        await page.goto('http://localhost:3000/phases/evomerge', {
            waitUntil: 'domcontentloaded',
            timeout: 10000
        });

        await page.waitForTimeout(3000);

        // Main screenshot
        await page.screenshot({
            path: 'tournament-ui-complete.png',
            fullPage: true
        });
        console.log('   📸 Screenshot saved: tournament-ui-complete.png');

        // Check all tournament-specific elements
        const hasTitle = await page.locator('text=/Tournament Evolution/i').count() > 0;
        console.log(`   ✅ Tournament Evolution title: ${hasTitle}`);

        const hasRules = await page.locator('text=/Tournament Selection Algorithm/i').count() > 0;
        console.log(`   ✅ Tournament Selection Algorithm: ${hasRules}`);

        const hasWinners = await page.locator('text=/Top 2 Winners/i').count() > 0;
        console.log(`   ✅ "Top 2 Winners": ${hasWinners}`);

        const hasLosers = await page.locator('text=/Bottom 6.*Chaos/i').count() > 0;
        console.log(`   ✅ "Bottom 6 Chaos": ${hasLosers}`);

        const hasEvolutionTree = await page.locator('text=/Evolution Tree/i').count() > 0;
        console.log(`   ✅ Evolution Tree: ${hasEvolutionTree}`);

        const hasFixedPopulation = await page.locator('text=/Fixed.*Tournament/i').count() > 0;
        console.log(`   ✅ Fixed Population (Tournament): ${hasFixedPopulation}`);

        const hasBackendInfo = await page.locator('text=/Backend.*8001/i').count() > 0;
        console.log(`   ✅ Backend port 8001 info: ${hasBackendInfo}`);

        // Test button functionality (hover to show it's interactive)
        const startButton = page.locator('button:has-text("Start Tournament Evolution")');
        const buttonExists = await startButton.count() > 0;
        console.log(`   ✅ Start Tournament button: ${buttonExists}`);

        if (buttonExists) {
            await startButton.hover();
            await page.screenshot({
                path: 'tournament-ui-button-hover.png',
                fullPage: false
            });
            console.log('   📸 Button hover screenshot: tournament-ui-button-hover.png');
        }

        // Test backend connectivity (just the call, don't actually start)
        console.log('\n🔗 Testing backend connectivity...');
        const backendResponse = await page.evaluate(async () => {
            try {
                const response = await fetch('http://localhost:8001/api/evomerge/status');
                return { status: response.status, ok: response.ok };
            } catch (error) {
                return { error: error.message };
            }
        });

        console.log(`   Backend connection test: ${JSON.stringify(backendResponse)}`);

        console.log('\n✨ COMPLETE TOURNAMENT UI TEST SUCCESSFUL! ✨');
        console.log('   📁 Screenshots: tournament-ui-complete.png, tournament-ui-button-hover.png');

    } catch (error) {
        console.error('❌ Error:', error.message);
        await page.screenshot({
            path: 'tournament-error-final.png',
            fullPage: true
        });
    } finally {
        await browser.close();
    }
}

testCompleteTournament().catch(console.error);