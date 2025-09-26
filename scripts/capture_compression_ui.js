const { chromium } = require('playwright');
const path = require('path');
const fs = require('fs').promises;

async function captureCompressionUI() {
  console.log('📸 Starting Phase 8 Compression UI screenshot capture...');

  // Ensure screenshots directory exists
  const screenshotsDir = path.join(__dirname, '..', 'screenshots', 'phase8-compression');
  await fs.mkdir(screenshotsDir, { recursive: true });

  const browser = await chromium.launch({
    headless: false, // Set to true for headless mode
    args: ['--start-maximized']
  });

  const context = await browser.newContext({
    viewport: { width: 1920, height: 1080 },
    deviceScaleFactor: 2, // High-quality screenshots
  });

  const page = await context.newPage();

  try {
    // Navigate to compression dashboard
    console.log('🌐 Navigating to compression dashboard...');
    await page.goto('http://localhost:3000/phases/compression', {
      waitUntil: 'networkidle'
    });

    // Wait for 3D scene to load
    await page.waitForTimeout(3000);

    // Capture initial state
    console.log('📷 Capturing initial state...');
    await page.screenshot({
      path: path.join(screenshotsDir, '01-compression-idle.png'),
      fullPage: false
    });

    // Start compression
    console.log('▶️ Starting compression pipeline...');
    await page.click('button:has-text("Start")');
    await page.waitForTimeout(2000);

    // Capture SeedLM stage
    console.log('📷 Capturing SeedLM compression...');
    await page.screenshot({
      path: path.join(screenshotsDir, '02-seedlm-active.png'),
      fullPage: false
    });

    // Wait for transition to VPTQ
    await page.waitForTimeout(3000);

    // Capture VPTQ stage
    console.log('📷 Capturing VPTQ quantization...');
    await page.screenshot({
      path: path.join(screenshotsDir, '03-vptq-active.png'),
      fullPage: false
    });

    // Wait for transition to Hypercompression
    await page.waitForTimeout(3000);

    // Capture Hypercompression stage
    console.log('📷 Capturing Hypercompression with Grokfast...');
    await page.screenshot({
      path: path.join(screenshotsDir, '04-hypercompression-active.png'),
      fullPage: false
    });

    // Adjust camera angle using mouse drag
    console.log('🎥 Adjusting camera angle...');
    await page.mouse.move(960, 540);
    await page.mouse.down();
    await page.mouse.move(1200, 400, { steps: 10 });
    await page.mouse.up();
    await page.waitForTimeout(1000);

    // Capture from different angle
    console.log('📷 Capturing from different angle...');
    await page.screenshot({
      path: path.join(screenshotsDir, '05-angle-view.png'),
      fullPage: false
    });

    // Zoom in using wheel
    console.log('🔍 Zooming in...');
    await page.mouse.wheel(0, -200);
    await page.waitForTimeout(1000);

    // Capture zoomed view
    console.log('📷 Capturing zoomed view...');
    await page.screenshot({
      path: path.join(screenshotsDir, '06-zoomed-view.png'),
      fullPage: false
    });

    // Toggle stats
    console.log('📊 Toggling stats display...');
    await page.click('button:has-text("Stats")');
    await page.waitForTimeout(1000);

    // Capture with stats
    console.log('📷 Capturing with performance stats...');
    await page.screenshot({
      path: path.join(screenshotsDir, '07-with-stats.png'),
      fullPage: false
    });

    // Adjust configuration sliders
    console.log('⚙️ Adjusting compression parameters...');

    // Change SeedLM bits
    const seedlmSlider = await page.$('input[type="range"][min="2"][max="4"]');
    if (seedlmSlider) {
      await seedlmSlider.evaluate(el => el.value = '2');
      await seedlmSlider.dispatchEvent('input');
    }

    // Change VPTQ bits
    const vptqSlider = await page.$('input[type="range"][min="2"][max="8"]');
    if (vptqSlider) {
      await vptqSlider.evaluate(el => el.value = '4');
      await vptqSlider.dispatchEvent('input');
    }

    await page.waitForTimeout(1000);

    // Capture with adjusted parameters
    console.log('📷 Capturing with adjusted parameters...');
    await page.screenshot({
      path: path.join(screenshotsDir, '08-adjusted-params.png'),
      fullPage: false
    });

    // Reset view
    console.log('🔄 Resetting view...');
    await page.click('button:has-text("Reset")');
    await page.waitForTimeout(2000);

    // Capture full page
    console.log('📷 Capturing full page...');
    await page.screenshot({
      path: path.join(screenshotsDir, '09-full-page.png'),
      fullPage: true
    });

    // Create animated GIF sequence (captures)
    console.log('🎬 Creating animation sequence...');
    const animationFrames = [];

    // Start new compression for animation
    await page.click('button:has-text("Start")');

    for (let i = 0; i < 20; i++) {
      await page.waitForTimeout(500);
      const framePath = path.join(screenshotsDir, `animation-frame-${String(i).padStart(2, '0')}.png`);
      await page.screenshot({ path: framePath });
      animationFrames.push(framePath);
    }

    console.log(`✅ Successfully captured ${animationFrames.length + 9} screenshots!`);
    console.log(`📁 Screenshots saved to: ${screenshotsDir}`);

    // Generate summary HTML
    const summaryHtml = `
<!DOCTYPE html>
<html>
<head>
  <title>Phase 8 Compression UI Screenshots</title>
  <style>
    body {
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
      background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
      color: white;
      padding: 2rem;
    }
    h1 { text-align: center; margin-bottom: 2rem; }
    .gallery {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
      gap: 2rem;
      margin-bottom: 2rem;
    }
    .screenshot {
      background: rgba(255, 255, 255, 0.1);
      border-radius: 12px;
      overflow: hidden;
      transition: transform 0.3s;
    }
    .screenshot:hover { transform: scale(1.05); }
    .screenshot img {
      width: 100%;
      height: auto;
      display: block;
    }
    .screenshot h3 {
      padding: 1rem;
      margin: 0;
      background: rgba(0, 0, 0, 0.3);
    }
    .features {
      background: rgba(255, 255, 255, 0.1);
      border-radius: 12px;
      padding: 2rem;
      margin-bottom: 2rem;
    }
    .features h2 { color: #fbbf24; }
    .features ul { line-height: 1.8; }
  </style>
</head>
<body>
  <h1>🎯 Phase 8: 3-Stage Compression Pipeline UI</h1>

  <div class="features">
    <h2>✨ Key Features Visualized</h2>
    <ul>
      <li><strong>SeedLM Chamber:</strong> LFSR cipher wheel, pseudo-random projection matrix, block compression</li>
      <li><strong>VPTQ Chamber:</strong> Codebook constellation, K-means++ optimization, vector quantization</li>
      <li><strong>Hypercompression Chamber:</strong> Ergodic trajectories (sinusoidal, spiral, chaotic), phase space visualization</li>
      <li><strong>Weight Flow:</strong> Particle system showing compression stages (Blue→Purple→Green→Gold)</li>
      <li><strong>Grokfast Accelerator:</strong> Energy field visualization with optimization waves</li>
      <li><strong>3D Metrics:</strong> Holographic display panels with real-time compression ratios</li>
      <li><strong>Interactive Controls:</strong> Parameter adjustment, camera controls, stage progression</li>
    </ul>
  </div>

  <div class="gallery">
    <div class="screenshot">
      <img src="01-compression-idle.png" alt="Initial State">
      <h3>1. Initial Idle State</h3>
    </div>
    <div class="screenshot">
      <img src="02-seedlm-active.png" alt="SeedLM Active">
      <h3>2. SeedLM Compression Active</h3>
    </div>
    <div class="screenshot">
      <img src="03-vptq-active.png" alt="VPTQ Active">
      <h3>3. VPTQ Quantization Active</h3>
    </div>
    <div class="screenshot">
      <img src="04-hypercompression-active.png" alt="Hypercompression Active">
      <h3>4. Hypercompression with Grokfast</h3>
    </div>
    <div class="screenshot">
      <img src="05-angle-view.png" alt="Different Angle">
      <h3>5. Alternative Camera Angle</h3>
    </div>
    <div class="screenshot">
      <img src="06-zoomed-view.png" alt="Zoomed View">
      <h3>6. Zoomed Detail View</h3>
    </div>
    <div class="screenshot">
      <img src="07-with-stats.png" alt="Performance Stats">
      <h3>7. With Performance Stats</h3>
    </div>
    <div class="screenshot">
      <img src="08-adjusted-params.png" alt="Adjusted Parameters">
      <h3>8. Adjusted Compression Parameters</h3>
    </div>
  </div>

  <div class="features">
    <h2>🚀 Technical Implementation</h2>
    <ul>
      <li>Built with React Three Fiber for declarative 3D scenes</li>
      <li>Custom shaders for holographic effects and energy fields</li>
      <li>Particle systems for weight flow visualization</li>
      <li>Real-time WebSocket integration with Python backend</li>
      <li>Accurate representation of SeedLM, VPTQ, and Hypercompression algorithms</li>
      <li>Grokfast acceleration visualization with EMA gradient flow</li>
    </ul>
  </div>
</body>
</html>
    `;

    await fs.writeFile(
      path.join(screenshotsDir, 'index.html'),
      summaryHtml.trim()
    );

    console.log('📄 Generated summary at: index.html');

  } catch (error) {
    console.error('❌ Error capturing screenshots:', error);
  } finally {
    await browser.close();
  }
}

// Run the capture
captureCompressionUI().catch(console.error);