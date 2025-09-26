const { chromium } = require('playwright');

async function auditAllPhases() {
  const browser = await chromium.launch({ headless: false });
  const page = await browser.newPage();

  const phases = [
    { id: 1, name: 'cognate', title: 'Cognate (Model Creation)', expectedControls: ['modelType', 'vocabSize', 'learningRate', 'grokfast'] },
    { id: 2, name: 'evomerge', title: 'EvoMerge (Evolution)', expectedControls: ['generations', 'populationSize', 'mutationRate', 'techniques'] },
    { id: 3, name: 'quietstar', title: 'Quiet-STaR (Reasoning)', expectedControls: ['thinkingTokens', 'mixingHead', 'rewardFunction'] },
    { id: 4, name: 'bitnet', title: 'BitNet (Compression)', expectedControls: ['quantizationBits', 'optimizationProfile', 'compressionRatio'] },
    { id: 5, name: 'forge', title: 'Forge Training', expectedControls: ['epochs', 'batchSize', 'optimizer', 'scheduler'] },
    { id: 6, name: 'baking', title: 'Tool & Persona Baking', expectedControls: ['toolSelection', 'personaConfig', 'integrationMode'] },
    { id: 7, name: 'adas', title: 'ADAS (Architecture Search)', expectedControls: ['searchSpace', 'searchStrategy', 'hardwareTarget'] },
    { id: 8, name: 'final', title: 'Final Compression', expectedControls: ['seedLMConfig', 'vptqSettings', 'deploymentTarget'] }
  ];

  const auditResults = [];

  for (const phase of phases) {
    console.log(`\n[PHASE ${phase.id}] Auditing ${phase.title}...`);
    console.log('=' .repeat(60));

    await page.goto(`http://localhost:3000/phases/${phase.name}`);
    await page.waitForTimeout(2000);

    const result = {
      phase: phase.name,
      phaseId: phase.id,
      title: phase.title,
      hasPhaseController: false,
      hasConfigControls: false,
      has3DAnimation: false,
      hasStartButton: false,
      hasMetricsDisplay: false,
      missingControls: [],
      foundControls: [],
      apiEndpointExists: false
    };

    // Check for PhaseController component
    result.hasPhaseController = await page.locator('.border.rounded-lg').first().isVisible().catch(() => false);
    console.log(`  PhaseController: ${result.hasPhaseController ? 'FOUND' : 'MISSING'}`);

    // Check for Start button
    result.hasStartButton = await page.locator('button:has-text("Start")').isVisible().catch(() => false);
    console.log(`  Start Button: ${result.hasStartButton ? 'FOUND' : 'MISSING'}`);

    // Check for 3D animation (Canvas element)
    result.has3DAnimation = await page.locator('canvas').isVisible().catch(() => false);
    console.log(`  3D Animation: ${result.has3DAnimation ? 'YES' : 'NO'}`);

    // Check for specific configuration controls
    for (const control of phase.expectedControls) {
      const found = await page.locator(`[id="${control}"], [name="${control}"], label:has-text("${control}")`)
        .isVisible()
        .catch(() => false);

      if (found) {
        result.foundControls.push(control);
      } else {
        result.missingControls.push(control);
      }
    }

    result.hasConfigControls = result.foundControls.length > 0;

    console.log(`  Configuration Controls: ${result.hasConfigControls ? 'PARTIAL' : 'NONE'}`);
    if (result.foundControls.length > 0) {
      console.log(`    Found: ${result.foundControls.join(', ')}`);
    }
    if (result.missingControls.length > 0) {
      console.log(`    Missing: ${result.missingControls.join(', ')}`);
    }

    // Check if API endpoint exists
    try {
      const response = await page.evaluate(async (phaseName) => {
        const res = await fetch(`/api/phases/${phaseName}`);
        return res.status;
      }, phase.name);
      result.apiEndpointExists = response < 500;
      console.log(`  API Endpoint: ${result.apiEndpointExists ? 'EXISTS' : 'NOT FOUND'}`);
    } catch (e) {
      result.apiEndpointExists = false;
      console.log(`  API Endpoint: ERROR`);
    }

    // Take screenshot
    await page.screenshot({
      path: `phase-${phase.id}-${phase.name}-audit.png`,
      fullPage: true
    });

    auditResults.push(result);

    // Summary for this phase
    if (result.has3DAnimation && !result.hasConfigControls) {
      console.log(`  [WARNING] Phase ${phase.id} only has 3D animation - needs functional controls!`);
    } else if (result.hasPhaseController && result.hasConfigControls) {
      console.log(`  [OK] Phase ${phase.id} appears to have functional controls`);
    }
  }

  // Generate summary report
  console.log('\n\n' + '='.repeat(80));
  console.log('AUDIT SUMMARY REPORT');
  console.log('='.repeat(80));

  const functionalPhases = auditResults.filter(r => r.hasPhaseController && r.hasConfigControls);
  const animationOnlyPhases = auditResults.filter(r => r.has3DAnimation && !r.hasConfigControls);
  const needsWork = auditResults.filter(r => !r.hasPhaseController || !r.hasConfigControls);

  console.log(`\nFunctional Phases (${functionalPhases.length}/8):`);
  functionalPhases.forEach(p => console.log(`  ✓ Phase ${p.phaseId}: ${p.title}`));

  console.log(`\nAnimation-Only Phases (${animationOnlyPhases.length}/8):`);
  animationOnlyPhases.forEach(p => console.log(`  ✗ Phase ${p.phaseId}: ${p.title}`));

  console.log(`\nPhases Needing Implementation (${needsWork.length}/8):`);
  needsWork.forEach(p => {
    console.log(`  Phase ${p.phaseId}: ${p.title}`);
    console.log(`    - Needs: PhaseController, Config Controls, API Connection`);
    console.log(`    - Missing Controls: ${p.missingControls.join(', ')}`);
  });

  // Save detailed report
  const fs = require('fs');
  fs.writeFileSync('phase-audit-report.json', JSON.stringify(auditResults, null, 2));
  console.log('\nDetailed report saved to phase-audit-report.json');

  await browser.close();
  return auditResults;
}

// Run the audit
auditAllPhases().then(results => {
  console.log('\n[COMPLETE] Audit finished. Check screenshots and report.');
}).catch(console.error);