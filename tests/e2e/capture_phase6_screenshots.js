const { chromium } = require('playwright');
const path = require('path');

(async () => {
    const browser = await chromium.launch({ headless: false });
    const context = await browser.newContext({
        viewport: { width: 1920, height: 1080 }
    });
    const page = await context.newPage();
    
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
    
    try {
        // Go to main page
        console.log('Navigating to dashboard homepage...');
        await page.goto('http://localhost:3000', { waitUntil: 'networkidle', timeout: 60000 });
        await page.waitForTimeout(2000);
        await page.screenshot({ path: `phase6-audit-${timestamp}-homepage.png`, fullPage: true });
        
        // Navigate to Phase 6 - Prompt Baking
        console.log('Navigating to Phase 6 - Baking...');
        await page.goto('http://localhost:3000/phases/baking', { waitUntil: 'networkidle', timeout: 60000 });
        await page.waitForTimeout(3000);
        await page.screenshot({ path: `phase6-audit-${timestamp}-baking-main.png`, fullPage: true });
        
        // Look for any prompt baking specific UI elements
        const promptElements = await page.$$('[data-testid*="prompt"], [class*="prompt"], [id*="prompt"]');
        console.log(`Found ${promptElements.length} prompt-related elements`);
        
        // Look for tool integration elements
        const toolElements = await page.$$('[data-testid*="tool"], [class*="tool"], [id*="tool"]');
        console.log(`Found ${toolElements.length} tool-related elements`);
        
        // Look for benchmark/success rate elements
        const benchmarkElements = await page.$$('[data-testid*="benchmark"], [class*="benchmark"], [id*="benchmark"], [class*="success"], [class*="rate"]');
        console.log(`Found ${benchmarkElements.length} benchmark-related elements`);
        
        // Check for any MCP server mentions
        const mcpElements = await page.$$('[data-testid*="mcp"], [class*="mcp"], [id*="mcp"]');
        console.log(`Found ${mcpElements.length} MCP-related elements`);
        
        // Check for SWE-bench mentions
        const sweElements = await page.$$text('SWE');
        console.log(`Found ${sweElements.length} SWE-bench mentions`);
        
        console.log('Screenshots captured successfully');
        
    } catch (error) {
        console.error('Error capturing screenshots:', error);
    } finally {
        await browser.close();
    }
})();
