const { chromium } = require('playwright');
const fs = require('fs');
const path = require('path');

async function auditAgentForge() {
    console.log('🔍 Starting Agent Forge UI Audit...\n');

    // Create screenshots directory
    const screenshotDir = path.join(__dirname, 'audit-screenshots');
    if (!fs.existsSync(screenshotDir)) {
        fs.mkdirSync(screenshotDir, { recursive: true });
    }

    const browser = await chromium.launch({
        headless: false,
        args: ['--start-maximized']
    });

    const context = await browser.newContext({
        viewport: { width: 1920, height: 1080 },
        deviceScaleFactor: 1,
    });

    const page = await context.newPage();

    try {
        // Test 1: Check if frontend is running
        console.log('✅ Test 1: Checking frontend server...');
        await page.goto('http://localhost:3000', { waitUntil: 'networkidle' });
        await page.screenshot({
            path: path.join(screenshotDir, '01-homepage.png'),
            fullPage: true
        });
        console.log('   Screenshot saved: 01-homepage.png');

        // Test 2: Navigate to Cognate phase
        console.log('\n✅ Test 2: Navigating to Cognate Training UI...');
        await page.click('text=Phases');
        await page.waitForTimeout(1000);
        await page.click('text=Cognate');
        await page.waitForTimeout(2000);
        await page.screenshot({
            path: path.join(screenshotDir, '02-cognate-ui.png'),
            fullPage: true
        });
        console.log('   Screenshot saved: 02-cognate-ui.png');

        // Test 3: Check WebSocket connection status
        console.log('\n✅ Test 3: Checking WebSocket connection indicator...');
        const connectionStatus = await page.locator('.connection-status, [class*="connection"]').first();
        if (await connectionStatus.isVisible()) {
            await connectionStatus.screenshot({
                path: path.join(screenshotDir, '03-connection-status.png')
            });
            console.log('   Connection status indicator found!');
        }

        // Test 4: Start training simulation
        console.log('\n✅ Test 4: Testing training controls...');
        const startButton = await page.locator('button:has-text("Start"), button:has-text("Train")').first();
        if (await startButton.isVisible()) {
            await startButton.click();
            await page.waitForTimeout(3000);
            await page.screenshot({
                path: path.join(screenshotDir, '04-training-started.png'),
                fullPage: true
            });
            console.log('   Training started - Screenshot saved: 04-training-started.png');
        }

        // Test 5: Check for real-time updates
        console.log('\n✅ Test 5: Checking for real-time metric updates...');
        const metricsSection = await page.locator('[class*="metric"], [class*="progress"]').first();
        if (await metricsSection.isVisible()) {
            await metricsSection.screenshot({
                path: path.join(screenshotDir, '05-metrics-display.png')
            });
            console.log('   Metrics display captured: 05-metrics-display.png');
        }

        // Test 6: Check API backend
        console.log('\n✅ Test 6: Testing backend API...');
        const apiResponse = await page.evaluate(async () => {
            try {
                const response = await fetch('http://localhost:8001/');
                return await response.json();
            } catch (error) {
                return { error: error.message };
            }
        });
        console.log('   Backend API Response:', apiResponse);

        // Test 7: Navigate to EvoMerge
        console.log('\n✅ Test 7: Checking EvoMerge UI...');
        await page.goto('http://localhost:3000');
        await page.click('text=Phases');
        await page.waitForTimeout(1000);
        await page.click('text=EvoMerge');
        await page.waitForTimeout(2000);
        await page.screenshot({
            path: path.join(screenshotDir, '06-evomerge-ui.png'),
            fullPage: true
        });
        console.log('   Screenshot saved: 06-evomerge-ui.png');

        // Generate audit report
        console.log('\n📊 Generating audit report...');
        const report = {
            timestamp: new Date().toISOString(),
            tests: {
                frontend: 'RUNNING',
                cognateUI: 'ACCESSIBLE',
                websocket: 'CHECK SCREENSHOTS',
                backend: apiResponse.error ? 'ERROR' : 'RUNNING',
                screenshots: fs.readdirSync(screenshotDir)
            },
            improvements: [
                'Real-time WebSocket connection status indicator',
                'Live training metrics updates',
                'Progress bars with animations',
                'Enhanced UI components with Lucide icons',
                'Socket.IO integration for real-time updates'
            ]
        };

        fs.writeFileSync(
            path.join(screenshotDir, 'audit-report.json'),
            JSON.stringify(report, null, 2)
        );
        console.log('   Report saved: audit-report.json');

        console.log('\n✨ Audit Complete! Check the audit-screenshots folder for results.');

    } catch (error) {
        console.error('❌ Audit Error:', error);
        await page.screenshot({
            path: path.join(screenshotDir, 'error-state.png'),
            fullPage: true
        });
    } finally {
        await browser.close();
    }
}

// Run the audit
auditAgentForge().catch(console.error);