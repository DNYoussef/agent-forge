/**
 * QuietStarDashboard Component
 * Main dashboard for Phase 3 Quiet Star reasoning visualization
 */

import React, { useState, useEffect, useCallback } from 'react';
import { PhaseController } from './PhaseController';
import { ReasoningTokenVisualizer } from './ReasoningTokenVisualizer';
import { ParallelThoughtsViewer } from './ParallelThoughtsViewer';
import { CurriculumProgressTracker } from './CurriculumProgressTracker';
import { ConvergenceMetrics } from './ConvergenceMetrics';
import {
  PhaseStatus,
  QuietStarConfig,
  QuietStarMetrics,
  WebSocketMessage,
  ThoughtStream,
  CurriculumStage,
  ConvergenceMetrics as ConvergenceMetricsType
} from '../types/phases';

interface QuietStarDashboardProps {
  initialConfig?: Partial<QuietStarConfig>;
  onConfigChange?: (config: QuietStarConfig) => void;
}

export const QuietStarDashboard: React.FC<QuietStarDashboardProps> = ({
  initialConfig = {},
  onConfigChange
}) => {
  const [status, setStatus] = useState<PhaseStatus>('idle');
  const [config, setConfig] = useState<QuietStarConfig>({
    parallelStreams: 4,
    maxThoughtLength: 512,
    temperature: 0.7,
    curriculumEnabled: true,
    visualizationMode: '3d',
    realTimeUpdates: true,
    ...initialConfig
  });

  const [metrics, setMetrics] = useState<QuietStarMetrics>({
    streams: [],
    curriculum: [],
    convergence: {
      internalizationRate: 0,
      thoughtQuality: 0,
      parallelismEfficiency: 0,
      curriculumProgress: 0,
      tokenGeneration: {
        rate: 0,
        accuracy: 0,
        diversity: 0
      },
      learning: {
        convergenceSpeed: 0,
        stabilityScore: 0,
        adaptationRate: 0
      }
    },
    totalTokensGenerated: 0,
    averageThoughtLength: 0,
    activeStreams: 0
  });

  const [websocket, setWebsocket] = useState<WebSocket | null>(null);
  const [selectedStream, setSelectedStream] = useState<number>(0);
  const [showAdvancedControls, setShowAdvancedControls] = useState(false);

  // WebSocket connection management
  useEffect(() => {
    if (config.realTimeUpdates && status === 'running') {
      const ws = new WebSocket('ws://localhost:8080/quiet-star');

      ws.onopen = () => {
        console.log('Connected to Quiet Star WebSocket');
        setWebsocket(ws);
      };

      ws.onmessage = (event) => {
        const message: WebSocketMessage = JSON.parse(event.data);
        handleWebSocketMessage(message);
      };

      ws.onclose = () => {
        console.log('Disconnected from Quiet Star WebSocket');
        setWebsocket(null);
      };

      ws.onerror = (error) => {
        console.error('WebSocket error:', error);
      };

      return () => {
        ws.close();
      };
    }
  }, [config.realTimeUpdates, status]);

  const handleWebSocketMessage = useCallback((message: WebSocketMessage) => {
    switch (message.type) {
      case 'reasoning_token':
        setMetrics(prev => ({
          ...prev,
          streams: prev.streams.map(stream =>
            stream.id === message.data.streamId
              ? { ...stream, tokens: [...stream.tokens, message.data.token] }
              : stream
          ),
          totalTokensGenerated: prev.totalTokensGenerated + 1
        }));
        break;

      case 'stream_update':
        setMetrics(prev => ({
          ...prev,
          streams: prev.streams.map(stream =>
            stream.id === message.data.streamId
              ? { ...stream, ...message.data.updates }
              : stream
          )
        }));
        break;

      case 'convergence_update':
        setMetrics(prev => ({
          ...prev,
          convergence: { ...prev.convergence, ...message.data }
        }));
        break;

      case 'curriculum_progress':
        setMetrics(prev => ({
          ...prev,
          curriculum: prev.curriculum.map(stage =>
            stage.id === message.data.stageId
              ? { ...stage, ...message.data.updates }
              : stage
          )
        }));
        break;
    }
  }, []);

  const handleStart = useCallback(async () => {
    setStatus('running');

    // Initialize streams
    const initialStreams: ThoughtStream[] = Array.from({ length: config.parallelStreams }, (_, i) => ({
      id: i,
      tokens: [],
      isActive: true,
      temperature: config.temperature + (i * 0.1), // Slight variation per stream
      length: 0,
      convergenceScore: 0
    }));

    // Initialize curriculum stages
    const initialCurriculum: CurriculumStage[] = [
      {
        id: 0,
        name: 'Foundation',
        description: 'Basic reasoning patterns',
        progress: 0,
        isActive: true,
        thoughtLength: { min: 16, max: 64, current: 32 },
        metrics: { accuracy: 0, efficiency: 0, internalization: 0 }
      },
      {
        id: 1,
        name: 'Structured Thinking',
        description: 'Organized thought processes',
        progress: 0,
        isActive: false,
        thoughtLength: { min: 32, max: 128, current: 64 },
        metrics: { accuracy: 0, efficiency: 0, internalization: 0 }
      },
      {
        id: 2,
        name: 'Complex Reasoning',
        description: 'Multi-step logical chains',
        progress: 0,
        isActive: false,
        thoughtLength: { min: 64, max: 256, current: 128 },
        metrics: { accuracy: 0, efficiency: 0, internalization: 0 }
      },
      {
        id: 3,
        name: 'Abstract Concepts',
        description: 'High-level abstraction',
        progress: 0,
        isActive: false,
        thoughtLength: { min: 128, max: 384, current: 256 },
        metrics: { accuracy: 0, efficiency: 0, internalization: 0 }
      },
      {
        id: 4,
        name: 'Meta-Reasoning',
        description: 'Thinking about thinking',
        progress: 0,
        isActive: false,
        thoughtLength: { min: 256, max: 512, current: 384 },
        metrics: { accuracy: 0, efficiency: 0, internalization: 0 }
      },
      {
        id: 5,
        name: 'Expert Integration',
        description: 'Seamless thought integration',
        progress: 0,
        isActive: false,
        thoughtLength: { min: 384, max: 512, current: 512 },
        metrics: { accuracy: 0, efficiency: 0, internalization: 0 }
      }
    ];

    setMetrics(prev => ({
      ...prev,
      streams: initialStreams,
      curriculum: initialCurriculum,
      activeStreams: config.parallelStreams
    }));

    // Start the Quiet Star engine (simulated API call)
    try {
      await fetch('/api/quiet-star/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(config)
      });
    } catch (error) {
      console.error('Failed to start Quiet Star engine:', error);
      setStatus('error');
    }
  }, [config]);

  const handlePause = useCallback(async () => {
    setStatus('paused');
    try {
      await fetch('/api/quiet-star/pause', { method: 'POST' });
    } catch (error) {
      console.error('Failed to pause Quiet Star engine:', error);
    }
  }, []);

  const handleResume = useCallback(async () => {
    setStatus('running');
    try {
      await fetch('/api/quiet-star/resume', { method: 'POST' });
    } catch (error) {
      console.error('Failed to resume Quiet Star engine:', error);
    }
  }, []);

  const handleStop = useCallback(async () => {
    setStatus('idle');
    websocket?.close();
    try {
      await fetch('/api/quiet-star/stop', { method: 'POST' });
    } catch (error) {
      console.error('Failed to stop Quiet Star engine:', error);
    }
  }, [websocket]);

  const handleConfigUpdate = useCallback((newConfig: Partial<QuietStarConfig>) => {
    const updatedConfig = { ...config, ...newConfig };
    setConfig(updatedConfig);
    onConfigChange?.(updatedConfig);
  }, [config, onConfigChange]);

  return (
    <div className="quiet-star-dashboard" data-testid="quiet-star-dashboard">
      <div className="dashboard-header">
        <h1>Phase 3: Quiet Star Reasoning</h1>
        <div className="header-controls">
          <button
            className="toggle-advanced"
            onClick={() => setShowAdvancedControls(!showAdvancedControls)}
            data-testid="toggle-advanced"
          >
            {showAdvancedControls ? 'Hide' : 'Show'} Advanced Controls
          </button>
        </div>
      </div>

      <PhaseController
        status={status}
        onStart={handleStart}
        onPause={handlePause}
        onResume={handleResume}
        onStop={handleStop}
      />

      {showAdvancedControls && (
        <div className="config-panel" data-testid="config-panel">
          <h3>Configuration</h3>
          <div className="config-controls">
            <div className="control-group">
              <label htmlFor="parallel-streams">Parallel Streams:</label>
              <input
                id="parallel-streams"
                type="range"
                min="1"
                max="8"
                value={config.parallelStreams}
                onChange={(e) => handleConfigUpdate({ parallelStreams: parseInt(e.target.value) })}
              />
              <span>{config.parallelStreams}</span>
            </div>

            <div className="control-group">
              <label htmlFor="temperature">Temperature:</label>
              <input
                id="temperature"
                type="range"
                min="0.1"
                max="2.0"
                step="0.1"
                value={config.temperature}
                onChange={(e) => handleConfigUpdate({ temperature: parseFloat(e.target.value) })}
              />
              <span>{config.temperature}</span>
            </div>

            <div className="control-group">
              <label htmlFor="max-thought-length">Max Thought Length:</label>
              <input
                id="max-thought-length"
                type="range"
                min="64"
                max="1024"
                step="64"
                value={config.maxThoughtLength}
                onChange={(e) => handleConfigUpdate({ maxThoughtLength: parseInt(e.target.value) })}
              />
              <span>{config.maxThoughtLength}</span>
            </div>

            <div className="control-group">
              <label>
                <input
                  type="checkbox"
                  checked={config.curriculumEnabled}
                  onChange={(e) => handleConfigUpdate({ curriculumEnabled: e.target.checked })}
                />
                Enable Curriculum Learning
              </label>
            </div>

            <div className="control-group">
              <label>
                <input
                  type="checkbox"
                  checked={config.realTimeUpdates}
                  onChange={(e) => handleConfigUpdate({ realTimeUpdates: e.target.checked })}
                />
                Real-time Updates
              </label>
            </div>

            <div className="control-group">
              <label htmlFor="visualization-mode">Visualization Mode:</label>
              <select
                id="visualization-mode"
                value={config.visualizationMode}
                onChange={(e) => handleConfigUpdate({ visualizationMode: e.target.value as '2d' | '3d' })}
              >
                <option value="2d">2D</option>
                <option value="3d">3D</option>
              </select>
            </div>
          </div>
        </div>
      )}

      <div className="metrics-summary" data-testid="metrics-summary">
        <div className="metric-card">
          <h4>Active Streams</h4>
          <div className="metric-value">{metrics.activeStreams}</div>
        </div>
        <div className="metric-card">
          <h4>Total Tokens</h4>
          <div className="metric-value">{metrics.totalTokensGenerated.toLocaleString()}</div>
        </div>
        <div className="metric-card">
          <h4>Avg Thought Length</h4>
          <div className="metric-value">{Math.round(metrics.averageThoughtLength)}</div>
        </div>
        <div className="metric-card">
          <h4>Internalization Rate</h4>
          <div className="metric-value">{(metrics.convergence.internalizationRate * 100).toFixed(1)}%</div>
        </div>
      </div>

      <div className="main-content">
        <div className="left-panel">
          <ReasoningTokenVisualizer
            streams={metrics.streams}
            selectedStream={selectedStream}
            onStreamSelect={setSelectedStream}
            visualizationMode={config.visualizationMode}
          />

          <ParallelThoughtsViewer
            streams={metrics.streams}
            activeStreams={metrics.activeStreams}
            visualizationMode={config.visualizationMode}
          />
        </div>

        <div className="right-panel">
          {config.curriculumEnabled && (
            <CurriculumProgressTracker
              stages={metrics.curriculum}
              currentStage={metrics.curriculum.find(s => s.isActive)?.id || 0}
            />
          )}

          <ConvergenceMetrics
            metrics={metrics.convergence}
            historicalData={[]} // TODO: Implement historical data tracking
          />
        </div>
      </div>

      <style>{`
        .quiet-star-dashboard {
          padding: 1rem;
          background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
          min-height: 100vh;
          color: white;
        }

        .dashboard-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 2rem;
          padding: 1rem;
          background: rgba(255, 255, 255, 0.1);
          border-radius: 12px;
          backdrop-filter: blur(10px);
        }

        .dashboard-header h1 {
          margin: 0;
          font-size: 2.5rem;
          font-weight: 300;
          text-shadow: 0 2px 4px rgba(0,0,0,0.3);
        }

        .header-controls {
          display: flex;
          gap: 1rem;
        }

        .toggle-advanced {
          padding: 0.75rem 1.5rem;
          background: rgba(255, 255, 255, 0.2);
          border: 1px solid rgba(255, 255, 255, 0.3);
          border-radius: 8px;
          color: white;
          cursor: pointer;
          transition: all 0.3s ease;
          backdrop-filter: blur(5px);
        }

        .toggle-advanced:hover {
          background: rgba(255, 255, 255, 0.3);
          transform: translateY(-2px);
        }

        .config-panel {
          background: rgba(255, 255, 255, 0.1);
          border-radius: 12px;
          padding: 1.5rem;
          margin-bottom: 2rem;
          backdrop-filter: blur(10px);
        }

        .config-panel h3 {
          margin: 0 0 1rem 0;
          font-size: 1.5rem;
          font-weight: 400;
        }

        .config-controls {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
          gap: 1rem;
        }

        .control-group {
          display: flex;
          flex-direction: column;
          gap: 0.5rem;
        }

        .control-group label {
          font-weight: 500;
          display: flex;
          align-items: center;
          gap: 0.5rem;
        }

        .control-group input[type="range"] {
          width: 100%;
          appearance: none;
          height: 6px;
          background: rgba(255, 255, 255, 0.3);
          border-radius: 3px;
          outline: none;
        }

        .control-group input[type="range"]::-webkit-slider-thumb {
          appearance: none;
          width: 18px;
          height: 18px;
          background: white;
          border-radius: 50%;
          cursor: pointer;
          box-shadow: 0 2px 4px rgba(0,0,0,0.3);
        }

        .control-group select {
          padding: 0.5rem;
          background: rgba(255, 255, 255, 0.2);
          border: 1px solid rgba(255, 255, 255, 0.3);
          border-radius: 6px;
          color: white;
          backdrop-filter: blur(5px);
        }

        .control-group select option {
          background: #764ba2;
          color: white;
        }

        .metrics-summary {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
          gap: 1rem;
          margin-bottom: 2rem;
        }

        .metric-card {
          background: rgba(255, 255, 255, 0.1);
          border-radius: 12px;
          padding: 1.5rem;
          text-align: center;
          backdrop-filter: blur(10px);
          border: 1px solid rgba(255, 255, 255, 0.2);
        }

        .metric-card h4 {
          margin: 0 0 0.5rem 0;
          font-size: 0.9rem;
          font-weight: 500;
          opacity: 0.8;
          text-transform: uppercase;
          letter-spacing: 1px;
        }

        .metric-value {
          font-size: 2rem;
          font-weight: 700;
          color: #fff;
          text-shadow: 0 2px 4px rgba(0,0,0,0.3);
        }

        .main-content {
          display: grid;
          grid-template-columns: 2fr 1fr;
          gap: 2rem;
          min-height: 600px;
        }

        .left-panel, .right-panel {
          display: flex;
          flex-direction: column;
          gap: 2rem;
        }

        @media (max-width: 1200px) {
          .main-content {
            grid-template-columns: 1fr;
          }

          .dashboard-header {
            flex-direction: column;
            gap: 1rem;
            text-align: center;
          }

          .config-controls {
            grid-template-columns: 1fr;
          }
        }

        @media (max-width: 768px) {
          .metrics-summary {
            grid-template-columns: repeat(2, 1fr);
          }

          .dashboard-header h1 {
            font-size: 2rem;
          }
        }
      `}</style>
    </div>
  );
};