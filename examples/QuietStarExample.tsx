/**
 * Quiet Star Dashboard Example
 * Complete example of how to use the Phase 3 Quiet Star visualization components
 */

import React, { useState, useEffect } from 'react';
import {
  QuietStarDashboard,
  QuietStarConfig,
  QuietStarMetrics,
  ReasoningToken,
  ThoughtStream,
  CurriculumStage,
  ConvergenceMetrics
} from '../src/ui/components';
import {
  QuietStarWebSocket,
  MockQuietStarWebSocket,
  useQuietStarWebSocket
} from '../src/ui/services';

// Example implementation showing how to integrate the Quiet Star dashboard
export const QuietStarExample: React.FC = () => {
  const [config, setConfig] = useState<QuietStarConfig>({
    parallelStreams: 4,
    maxThoughtLength: 512,
    temperature: 0.7,
    curriculumEnabled: true,
    visualizationMode: '3d',
    realTimeUpdates: true
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

  // WebSocket integration using the custom hook
  const { connect, disconnect, connectionState, isConnected } = useQuietStarWebSocket(
    {
      url: 'ws://localhost:8080/quiet-star',
      reconnectInterval: 5000,
      maxReconnectAttempts: 10,
      heartbeatInterval: 30000
    },
    {
      onReasoningToken: (token: ReasoningToken) => {
        setMetrics(prev => ({
          ...prev,
          streams: prev.streams.map(stream =>
            stream.id === token.streamId
              ? { ...stream, tokens: [...stream.tokens, token] }
              : stream
          ),
          totalTokensGenerated: prev.totalTokensGenerated + 1,
          averageThoughtLength: calculateAverageThoughtLength(prev.streams)
        }));
      },
      onStreamUpdate: (streamId: number, updates: Partial<ThoughtStream>) => {
        setMetrics(prev => ({
          ...prev,
          streams: prev.streams.map(stream =>
            stream.id === streamId
              ? { ...stream, ...updates }
              : stream
          )
        }));
      },
      onConvergenceUpdate: (convergenceUpdates: Partial<ConvergenceMetrics>) => {
        setMetrics(prev => ({
          ...prev,
          convergence: { ...prev.convergence, ...convergenceUpdates }
        }));
      },
      onCurriculumProgress: (stageId: number, updates: Partial<CurriculumStage>) => {
        setMetrics(prev => ({
          ...prev,
          curriculum: prev.curriculum.map(stage =>
            stage.id === stageId
              ? { ...stage, ...updates }
              : stage
          )
        }));
      },
      onConnectionChange: (connected: boolean) => {
        console.log('WebSocket connection changed:', connected);
      },
      onError: (error: Error) => {
        console.error('WebSocket error:', error);
      }
    }
  );

  const calculateAverageThoughtLength = (streams: ThoughtStream[]): number => {
    const totalTokens = streams.reduce((sum, stream) => sum + stream.tokens.length, 0);
    const activeStreamCount = streams.filter(s => s.isActive).length;
    return activeStreamCount > 0 ? totalTokens / activeStreamCount : 0;
  };

  // Initialize demo data
  useEffect(() => {
    initializeDemoData();
  }, []);

  const initializeDemoData = () => {
    // Initialize streams
    const initialStreams: ThoughtStream[] = Array.from({ length: config.parallelStreams }, (_, i) => ({
      id: i,
      tokens: [],
      isActive: true,
      temperature: config.temperature + (i * 0.1),
      length: 0,
      convergenceScore: Math.random() * 0.5
    }));

    // Initialize curriculum stages
    const initialCurriculum: CurriculumStage[] = [
      {
        id: 0,
        name: 'Foundation',
        description: 'Basic reasoning patterns',
        progress: 85,
        isActive: false,
        thoughtLength: { min: 16, max: 64, current: 45 },
        metrics: { accuracy: 78, efficiency: 82, internalization: 65 }
      },
      {
        id: 1,
        name: 'Structured Thinking',
        description: 'Organized thought processes',
        progress: 92,
        isActive: false,
        thoughtLength: { min: 32, max: 128, current: 87 },
        metrics: { accuracy: 85, efficiency: 88, internalization: 74 }
      },
      {
        id: 2,
        name: 'Complex Reasoning',
        description: 'Multi-step logical chains',
        progress: 67,
        isActive: true,
        thoughtLength: { min: 64, max: 256, current: 156 },
        metrics: { accuracy: 72, efficiency: 69, internalization: 58 }
      },
      {
        id: 3,
        name: 'Abstract Concepts',
        description: 'High-level abstraction',
        progress: 23,
        isActive: false,
        thoughtLength: { min: 128, max: 384, current: 189 },
        metrics: { accuracy: 45, efficiency: 41, internalization: 32 }
      },
      {
        id: 4,
        name: 'Meta-Reasoning',
        description: 'Thinking about thinking',
        progress: 8,
        isActive: false,
        thoughtLength: { min: 256, max: 512, current: 278 },
        metrics: { accuracy: 28, efficiency: 25, internalization: 18 }
      },
      {
        id: 5,
        name: 'Expert Integration',
        description: 'Seamless thought integration',
        progress: 0,
        isActive: false,
        thoughtLength: { min: 384, max: 512, current: 384 },
        metrics: { accuracy: 0, efficiency: 0, internalization: 0 }
      }
    ];

    // Initialize convergence metrics
    const initialConvergence: ConvergenceMetrics = {
      internalizationRate: 0.68,
      thoughtQuality: 0.74,
      parallelismEfficiency: 0.71,
      curriculumProgress: 0.46,
      tokenGeneration: {
        rate: 165,
        accuracy: 0.82,
        diversity: 0.67
      },
      learning: {
        convergenceSpeed: 0.58,
        stabilityScore: 0.75,
        adaptationRate: 0.63
      }
    };

    setMetrics({
      streams: initialStreams,
      curriculum: initialCurriculum,
      convergence: initialConvergence,
      totalTokensGenerated: 8547,
      averageThoughtLength: 89,
      activeStreams: config.parallelStreams
    });
  };

  const handleConfigChange = (newConfig: QuietStarConfig) => {
    setConfig(newConfig);

    // If real-time updates enabled, connect to WebSocket
    if (newConfig.realTimeUpdates && !isConnected) {
      connect();
    } else if (!newConfig.realTimeUpdates && isConnected) {
      disconnect();
    }
  };

  return (
    <div className="quiet-star-example">
      <div className="example-header">
        <h1>Phase 3: Quiet Star Reasoning Visualization</h1>
        <p>
          This example demonstrates the complete Quiet Star dashboard with real-time
          reasoning token visualization, parallel thought streams, curriculum learning
          progress, and convergence metrics.
        </p>

        <div className="connection-status">
          <span className={`status-indicator ${connectionState}`}>
            {connectionState.toUpperCase()}
          </span>
          <div className="connection-controls">
            {!isConnected ? (
              <button onClick={connect} className="connect-btn">
                Connect to Real-time Stream
              </button>
            ) : (
              <button onClick={disconnect} className="disconnect-btn">
                Disconnect
              </button>
            )}
          </div>
        </div>
      </div>

      <QuietStarDashboard
        initialConfig={config}
        onConfigChange={handleConfigChange}
      />

      <div className="example-info">
        <h3>Implementation Details</h3>
        <div className="info-grid">
          <div className="info-card">
            <h4>Components Used</h4>
            <ul>
              <li>QuietStarDashboard - Main container</li>
              <li>ReasoningTokenVisualizer - Token stream display</li>
              <li>ParallelThoughtsViewer - 3D parallel visualization</li>
              <li>CurriculumProgressTracker - Learning stages</li>
              <li>ConvergenceMetrics - Performance tracking</li>
              <li>ThreeJSVisualization - Advanced 3D rendering</li>
            </ul>
          </div>

          <div className="info-card">
            <h4>WebSocket Integration</h4>
            <ul>
              <li>Real-time reasoning token updates</li>
              <li>Stream convergence monitoring</li>
              <li>Curriculum progress tracking</li>
              <li>Automatic reconnection</li>
              <li>Message queuing</li>
              <li>Mock server for development</li>
            </ul>
          </div>

          <div className="info-card">
            <h4>Visualization Features</h4>
            <ul>
              <li>2D and 3D rendering modes</li>
              <li>Interactive camera controls</li>
              <li>Thought token highlighting</li>
              <li>Stream connection visualization</li>
              <li>Performance metrics charts</li>
              <li>Responsive design</li>
            </ul>
          </div>

          <div className="info-card">
            <h4>Enhanced Quiet Star</h4>
            <ul>
              <li>4-stream parallel reasoning</li>
              <li>Fast curriculum learning (6 stages)</li>
              <li>Internalization progress tracking</li>
              <li>Token generation optimization</li>
              <li>Convergence rate monitoring</li>
              <li>Quality assessment metrics</li>
            </ul>
          </div>
        </div>

        <div className="code-example">
          <h4>Basic Usage</h4>
          <pre><code>{`import { QuietStarDashboard } from '../src/ui/components';

<QuietStarDashboard
  initialConfig={{
    parallelStreams: 4,
    maxThoughtLength: 512,
    temperature: 0.7,
    curriculumEnabled: true,
    visualizationMode: '3d',
    realTimeUpdates: true
  }}
  onConfigChange={(config) => console.log('Config updated:', config)}
/>`}</code></pre>
        </div>
      </div>

      <style>{`
        .quiet-star-example {
          min-height: 100vh;
          background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
          padding: 2rem;
          color: white;
        }

        .example-header {
          text-align: center;
          margin-bottom: 3rem;
          max-width: 800px;
          margin-left: auto;
          margin-right: auto;
        }

        .example-header h1 {
          font-size: 3rem;
          margin-bottom: 1rem;
          text-shadow: 0 2px 4px rgba(0,0,0,0.3);
        }

        .example-header p {
          font-size: 1.2rem;
          opacity: 0.9;
          line-height: 1.6;
          margin-bottom: 2rem;
        }

        .connection-status {
          display: flex;
          align-items: center;
          justify-content: center;
          gap: 1rem;
          background: rgba(255, 255, 255, 0.1);
          padding: 1rem;
          border-radius: 8px;
          backdrop-filter: blur(10px);
        }

        .status-indicator {
          padding: 0.5rem 1rem;
          border-radius: 6px;
          font-weight: 600;
          font-size: 0.9rem;
        }

        .status-indicator.open {
          background: rgba(46, 204, 113, 0.3);
          color: #2ecc71;
        }

        .status-indicator.connecting {
          background: rgba(241, 196, 15, 0.3);
          color: #f1c40f;
        }

        .status-indicator.closed {
          background: rgba(231, 76, 60, 0.3);
          color: #e74c3c;
        }

        .connect-btn, .disconnect-btn {
          padding: 0.75rem 1.5rem;
          border: none;
          border-radius: 6px;
          font-weight: 600;
          cursor: pointer;
          transition: all 0.3s ease;
        }

        .connect-btn {
          background: #2ecc71;
          color: white;
        }

        .connect-btn:hover {
          background: #27ae60;
          transform: translateY(-2px);
        }

        .disconnect-btn {
          background: #e74c3c;
          color: white;
        }

        .disconnect-btn:hover {
          background: #c0392b;
          transform: translateY(-2px);
        }

        .example-info {
          margin-top: 4rem;
          background: rgba(255, 255, 255, 0.1);
          border-radius: 12px;
          padding: 2rem;
          backdrop-filter: blur(10px);
        }

        .example-info h3 {
          text-align: center;
          margin-bottom: 2rem;
          font-size: 2rem;
        }

        .info-grid {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
          gap: 1.5rem;
          margin-bottom: 2rem;
        }

        .info-card {
          background: rgba(255, 255, 255, 0.05);
          border-radius: 8px;
          padding: 1.5rem;
          border: 1px solid rgba(255, 255, 255, 0.2);
        }

        .info-card h4 {
          margin: 0 0 1rem 0;
          color: #3498db;
          font-size: 1.2rem;
        }

        .info-card ul {
          list-style: none;
          padding: 0;
          margin: 0;
        }

        .info-card li {
          padding: 0.25rem 0;
          opacity: 0.9;
          position: relative;
          padding-left: 1rem;
        }

        .info-card li::before {
          content: '▸';
          position: absolute;
          left: 0;
          color: #3498db;
        }

        .code-example {
          background: rgba(0, 0, 0, 0.3);
          border-radius: 8px;
          padding: 1.5rem;
          border: 1px solid rgba(255, 255, 255, 0.2);
        }

        .code-example h4 {
          margin: 0 0 1rem 0;
          color: #3498db;
        }

        .code-example pre {
          margin: 0;
          overflow-x: auto;
        }

        .code-example code {
          font-family: 'Courier New', monospace;
          font-size: 0.9rem;
          line-height: 1.4;
          color: #f8f8f2;
        }

        @media (max-width: 768px) {
          .quiet-star-example {
            padding: 1rem;
          }

          .example-header h1 {
            font-size: 2rem;
          }

          .connection-status {
            flex-direction: column;
            gap: 0.5rem;
          }

          .info-grid {
            grid-template-columns: 1fr;
          }
        }
      `}</style>
    </div>
  );
};

export default QuietStarExample;