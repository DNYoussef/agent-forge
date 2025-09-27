/**
 * ConvergenceMetrics Component
 * Displays internalization progress and convergence rates with Chart.js integration
 */

import React, { useRef, useEffect, useState, useMemo } from 'react';
import { ConvergenceMetrics as ConvergenceMetricsType } from '../types/phases';

interface ConvergenceMetricsProps {
  metrics: ConvergenceMetricsType;
  historicalData: Array<{
    timestamp: number;
    metrics: ConvergenceMetricsType;
  }>;
}

interface ChartData {
  labels: string[];
  datasets: Array<{
    label: string;
    data: number[];
    borderColor: string;
    backgroundColor: string;
    tension: number;
  }>;
}

export const ConvergenceMetrics: React.FC<ConvergenceMetricsProps> = ({
  metrics,
  historicalData
}) => {
  const chartRef = useRef<HTMLCanvasElement>(null);
  const radarChartRef = useRef<HTMLCanvasElement>(null);
  const [selectedTimeRange, setSelectedTimeRange] = useState<'1h' | '6h' | '24h' | 'all'>('1h');
  const [showDetailedMetrics, setShowDetailedMetrics] = useState(false);

  // Generate mock historical data if none provided
  const processedHistoricalData = useMemo(() => {
    if (historicalData.length > 0) return historicalData;

    // Generate mock data for demonstration
    const mockData = [];
    const now = Date.now();
    const timeRanges = {
      '1h': 60,
      '6h': 360,
      '24h': 1440,
      'all': 7200
    };

    const minutes = timeRanges[selectedTimeRange];
    for (let i = minutes; i >= 0; i -= 5) {
      const timestamp = now - (i * 60 * 1000);
      const progress = (minutes - i) / minutes;

      mockData.push({
        timestamp,
        metrics: {
          internalizationRate: Math.min(0.95, 0.1 + progress * 0.7 + Math.sin(progress * 10) * 0.1),
          thoughtQuality: Math.min(0.9, 0.2 + progress * 0.6 + Math.cos(progress * 8) * 0.08),
          parallelismEfficiency: Math.min(0.85, 0.3 + progress * 0.5 + Math.sin(progress * 6) * 0.05),
          curriculumProgress: Math.min(1.0, progress * 0.8),
          tokenGeneration: {
            rate: 150 + progress * 50 + Math.sin(progress * 12) * 20,
            accuracy: Math.min(0.95, 0.7 + progress * 0.2 + Math.cos(progress * 15) * 0.05),
            diversity: Math.min(0.9, 0.4 + progress * 0.4 + Math.sin(progress * 9) * 0.1)
          },
          learning: {
            convergenceSpeed: Math.min(0.8, 0.1 + progress * 0.6 + Math.cos(progress * 7) * 0.1),
            stabilityScore: Math.min(0.9, 0.5 + progress * 0.3 + Math.sin(progress * 11) * 0.05),
            adaptationRate: Math.min(0.85, 0.2 + progress * 0.5 + Math.cos(progress * 13) * 0.1)
          }
        }
      });
    }

    return mockData;
  }, [historicalData, selectedTimeRange]);

  // Chart.js implementation using Canvas API
  useEffect(() => {
    const canvas = chartRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const drawLineChart = () => {
      const { width, height } = canvas;
      const padding = 60;
      const chartWidth = width - padding * 2;
      const chartHeight = height - padding * 2;

      ctx.clearRect(0, 0, width, height);

      // Background
      ctx.fillStyle = 'rgba(0, 0, 0, 0.3)';
      ctx.fillRect(0, 0, width, height);

      // Chart area
      ctx.fillStyle = 'rgba(255, 255, 255, 0.05)';
      ctx.fillRect(padding, padding, chartWidth, chartHeight);

      // Grid lines
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.1)';
      ctx.lineWidth = 1;

      // Vertical grid lines
      for (let i = 0; i <= 10; i++) {
        const x = padding + (i / 10) * chartWidth;
        ctx.beginPath();
        ctx.moveTo(x, padding);
        ctx.lineTo(x, padding + chartHeight);
        ctx.stroke();
      }

      // Horizontal grid lines
      for (let i = 0; i <= 10; i++) {
        const y = padding + (i / 10) * chartHeight;
        ctx.beginPath();
        ctx.moveTo(padding, y);
        ctx.lineTo(padding + chartWidth, y);
        ctx.stroke();
      }

      // Data lines
      const dataLines = [
        { key: 'internalizationRate', color: '#3498db', label: 'Internalization Rate' },
        { key: 'thoughtQuality', color: '#2ecc71', label: 'Thought Quality' },
        { key: 'parallelismEfficiency', color: '#f39c12', label: 'Parallelism Efficiency' },
        { key: 'curriculumProgress', color: '#e74c3c', label: 'Curriculum Progress' }
      ];

      dataLines.forEach(line => {
        ctx.strokeStyle = line.color;
        ctx.lineWidth = 2;
        ctx.beginPath();

        processedHistoricalData.forEach((point, index) => {
          const x = padding + (index / (processedHistoricalData.length - 1)) * chartWidth;
          const value = point.metrics[line.key as keyof ConvergenceMetricsType] as number;
          const y = padding + chartHeight - (value * chartHeight);

          if (index === 0) {
            ctx.moveTo(x, y);
          } else {
            ctx.lineTo(x, y);
          }
        });

        ctx.stroke();

        // Add glow effect
        ctx.shadowColor = line.color;
        ctx.shadowBlur = 10;
        ctx.stroke();
        ctx.shadowBlur = 0;
      });

      // Labels
      ctx.fillStyle = 'white';
      ctx.font = '12px monospace';
      ctx.textAlign = 'center';

      // X-axis labels (time)
      for (let i = 0; i <= 5; i++) {
        const x = padding + (i / 5) * chartWidth;
        const timeIndex = Math.floor((i / 5) * (processedHistoricalData.length - 1));
        const time = new Date(processedHistoricalData[timeIndex]?.timestamp || 0);
        ctx.fillText(time.toLocaleTimeString(), x, height - 20);
      }

      // Y-axis labels (percentage)
      ctx.textAlign = 'right';
      for (let i = 0; i <= 10; i++) {
        const y = padding + chartHeight - (i / 10) * chartHeight;
        ctx.fillText(`${i * 10}%`, padding - 10, y + 4);
      }

      // Legend
      ctx.textAlign = 'left';
      dataLines.forEach((line, index) => {
        const legendY = 20 + index * 20;
        ctx.fillStyle = line.color;
        ctx.fillRect(width - 200, legendY - 10, 15, 15);
        ctx.fillStyle = 'white';
        ctx.fillText(line.label, width - 180, legendY);
      });

      // Title
      ctx.fillStyle = 'white';
      ctx.font = '16px sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('Convergence Metrics Over Time', width / 2, 30);
    };

    drawLineChart();
  }, [processedHistoricalData]);

  // Radar chart for current metrics
  useEffect(() => {
    const canvas = radarChartRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const drawRadarChart = () => {
      const { width, height } = canvas;
      const centerX = width / 2;
      const centerY = height / 2;
      const radius = Math.min(width, height) / 2 - 40;

      ctx.clearRect(0, 0, width, height);

      // Background
      ctx.fillStyle = 'rgba(0, 0, 0, 0.3)';
      ctx.fillRect(0, 0, width, height);

      const radarMetrics = [
        { label: 'Internalization', value: metrics.internalizationRate, angle: 0 },
        { label: 'Quality', value: metrics.thoughtQuality, angle: Math.PI / 3 },
        { label: 'Efficiency', value: metrics.parallelismEfficiency, angle: 2 * Math.PI / 3 },
        { label: 'Generation Rate', value: metrics.tokenGeneration.rate / 200, angle: Math.PI },
        { label: 'Accuracy', value: metrics.tokenGeneration.accuracy, angle: 4 * Math.PI / 3 },
        { label: 'Stability', value: metrics.learning.stabilityScore, angle: 5 * Math.PI / 3 }
      ];

      // Draw radar grid
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.2)';
      ctx.lineWidth = 1;

      // Concentric circles
      for (let i = 1; i <= 5; i++) {
        ctx.beginPath();
        ctx.arc(centerX, centerY, (radius * i) / 5, 0, Math.PI * 2);
        ctx.stroke();
      }

      // Radial lines
      radarMetrics.forEach(metric => {
        ctx.beginPath();
        ctx.moveTo(centerX, centerY);
        ctx.lineTo(
          centerX + Math.cos(metric.angle - Math.PI / 2) * radius,
          centerY + Math.sin(metric.angle - Math.PI / 2) * radius
        );
        ctx.stroke();
      });

      // Draw data polygon
      ctx.strokeStyle = '#3498db';
      ctx.fillStyle = 'rgba(52, 152, 219, 0.2)';
      ctx.lineWidth = 3;
      ctx.beginPath();

      radarMetrics.forEach((metric, index) => {
        const value = Math.min(1, Math.max(0, metric.value));
        const x = centerX + Math.cos(metric.angle - Math.PI / 2) * radius * value;
        const y = centerY + Math.sin(metric.angle - Math.PI / 2) * radius * value;

        if (index === 0) {
          ctx.moveTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }
      });

      ctx.closePath();
      ctx.fill();
      ctx.stroke();

      // Draw data points
      radarMetrics.forEach(metric => {
        const value = Math.min(1, Math.max(0, metric.value));
        const x = centerX + Math.cos(metric.angle - Math.PI / 2) * radius * value;
        const y = centerY + Math.sin(metric.angle - Math.PI / 2) * radius * value;

        ctx.fillStyle = '#3498db';
        ctx.beginPath();
        ctx.arc(x, y, 4, 0, Math.PI * 2);
        ctx.fill();

        // Add glow
        ctx.shadowColor = '#3498db';
        ctx.shadowBlur = 8;
        ctx.fill();
        ctx.shadowBlur = 0;
      });

      // Labels
      ctx.fillStyle = 'white';
      ctx.font = '12px sans-serif';
      ctx.textAlign = 'center';

      radarMetrics.forEach(metric => {
        const labelRadius = radius + 25;
        const x = centerX + Math.cos(metric.angle - Math.PI / 2) * labelRadius;
        const y = centerY + Math.sin(metric.angle - Math.PI / 2) * labelRadius;

        ctx.fillText(metric.label, x, y);
        ctx.fillText(`${(metric.value * 100).toFixed(1)}%`, x, y + 15);
      });

      // Title
      ctx.font = '16px sans-serif';
      ctx.fillText('Current Performance Radar', centerX, 25);
    };

    drawRadarChart();
  }, [metrics]);

  const getConvergenceStatus = () => {
    const overallScore = (
      metrics.internalizationRate +
      metrics.thoughtQuality +
      metrics.parallelismEfficiency +
      metrics.learning.stabilityScore
    ) / 4;

    if (overallScore >= 0.9) return { status: 'Excellent', color: '#2ecc71' };
    if (overallScore >= 0.7) return { status: 'Good', color: '#3498db' };
    if (overallScore >= 0.5) return { status: 'Improving', color: '#f39c12' };
    return { status: 'Learning', color: '#e74c3c' };
  };

  const convergenceStatus = getConvergenceStatus();

  return (
    <div className="convergence-metrics" data-testid="convergence-metrics">
      <div className="metrics-header">
        <h3>Convergence & Internalization</h3>
        <div className="header-controls">
          <div className="time-range-selector">
            {(['1h', '6h', '24h', 'all'] as const).map(range => (
              <button
                key={range}
                className={`time-range-btn ${selectedTimeRange === range ? 'active' : ''}`}
                onClick={() => setSelectedTimeRange(range)}
              >
                {range}
              </button>
            ))}
          </div>

          <button
            className="toggle-details"
            onClick={() => setShowDetailedMetrics(!showDetailedMetrics)}
          >
            {showDetailedMetrics ? 'Hide' : 'Show'} Details
          </button>
        </div>
      </div>

      <div className="convergence-status" data-testid="convergence-status">
        <div className="status-indicator">
          <div
            className="status-dot"
            style={{ backgroundColor: convergenceStatus.color }}
          />
          <span className="status-text">{convergenceStatus.status}</span>
        </div>
        <div className="overall-score">
          {((metrics.internalizationRate + metrics.thoughtQuality + metrics.parallelismEfficiency + metrics.learning.stabilityScore) / 4 * 100).toFixed(1)}%
        </div>
      </div>

      <div className="charts-container">
        <div className="line-chart-container">
          <canvas
            ref={chartRef}
            width={600}
            height={300}
            className="convergence-chart"
          />
        </div>

        <div className="radar-chart-container">
          <canvas
            ref={radarChartRef}
            width={350}
            height={300}
            className="radar-chart"
          />
        </div>
      </div>

      <div className="key-metrics-grid" data-testid="key-metrics">
        <div className="metric-card primary">
          <h4>Internalization Rate</h4>
          <div className="metric-value">{(metrics.internalizationRate * 100).toFixed(1)}%</div>
          <div className="metric-bar">
            <div
              className="metric-fill internalization"
              style={{ width: `${metrics.internalizationRate * 100}%` }}
            />
          </div>
          <p className="metric-description">
            How well the model integrates reasoning without explicit thought tokens
          </p>
        </div>

        <div className="metric-card primary">
          <h4>Thought Quality</h4>
          <div className="metric-value">{(metrics.thoughtQuality * 100).toFixed(1)}%</div>
          <div className="metric-bar">
            <div
              className="metric-fill quality"
              style={{ width: `${metrics.thoughtQuality * 100}%` }}
            />
          </div>
          <p className="metric-description">
            Average quality and coherence of generated thought sequences
          </p>
        </div>

        <div className="metric-card primary">
          <h4>Parallel Efficiency</h4>
          <div className="metric-value">{(metrics.parallelismEfficiency * 100).toFixed(1)}%</div>
          <div className="metric-bar">
            <div
              className="metric-fill efficiency"
              style={{ width: `${metrics.parallelismEfficiency * 100}%` }}
            />
          </div>
          <p className="metric-description">
            Effectiveness of parallel thought stream coordination
          </p>
        </div>

        <div className="metric-card primary">
          <h4>Curriculum Progress</h4>
          <div className="metric-value">{(metrics.curriculumProgress * 100).toFixed(1)}%</div>
          <div className="metric-bar">
            <div
              className="metric-fill curriculum"
              style={{ width: `${metrics.curriculumProgress * 100}%` }}
            />
          </div>
          <p className="metric-description">
            Overall progress through Fast Quiet-STaR curriculum stages
          </p>
        </div>
      </div>

      {showDetailedMetrics && (
        <div className="detailed-metrics" data-testid="detailed-metrics">
          <div className="metrics-section">
            <h4>Token Generation</h4>
            <div className="sub-metrics">
              <div className="sub-metric">
                <label>Generation Rate</label>
                <value>{metrics.tokenGeneration.rate.toFixed(1)} tokens/sec</value>
                <div className="sub-bar">
                  <div
                    className="sub-fill rate"
                    style={{ width: `${(metrics.tokenGeneration.rate / 200) * 100}%` }}
                  />
                </div>
              </div>
              <div className="sub-metric">
                <label>Accuracy</label>
                <value>{(metrics.tokenGeneration.accuracy * 100).toFixed(1)}%</value>
                <div className="sub-bar">
                  <div
                    className="sub-fill accuracy"
                    style={{ width: `${metrics.tokenGeneration.accuracy * 100}%` }}
                  />
                </div>
              </div>
              <div className="sub-metric">
                <label>Diversity</label>
                <value>{(metrics.tokenGeneration.diversity * 100).toFixed(1)}%</value>
                <div className="sub-bar">
                  <div
                    className="sub-fill diversity"
                    style={{ width: `${metrics.tokenGeneration.diversity * 100}%` }}
                  />
                </div>
              </div>
            </div>
          </div>

          <div className="metrics-section">
            <h4>Learning Dynamics</h4>
            <div className="sub-metrics">
              <div className="sub-metric">
                <label>Convergence Speed</label>
                <value>{(metrics.learning.convergenceSpeed * 100).toFixed(1)}%</value>
                <div className="sub-bar">
                  <div
                    className="sub-fill convergence"
                    style={{ width: `${metrics.learning.convergenceSpeed * 100}%` }}
                  />
                </div>
              </div>
              <div className="sub-metric">
                <label>Stability Score</label>
                <value>{(metrics.learning.stabilityScore * 100).toFixed(1)}%</value>
                <div className="sub-bar">
                  <div
                    className="sub-fill stability"
                    style={{ width: `${metrics.learning.stabilityScore * 100}%` }}
                  />
                </div>
              </div>
              <div className="sub-metric">
                <label>Adaptation Rate</label>
                <value>{(metrics.learning.adaptationRate * 100).toFixed(1)}%</value>
                <div className="sub-bar">
                  <div
                    className="sub-fill adaptation"
                    style={{ width: `${metrics.learning.adaptationRate * 100}%` }}
                  />
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      <style>{`
        .convergence-metrics {
          background: rgba(255, 255, 255, 0.1);
          border-radius: 12px;
          padding: 1.5rem;
          backdrop-filter: blur(10px);
          border: 1px solid rgba(255, 255, 255, 0.2);
          color: white;
        }

        .metrics-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 1.5rem;
          flex-wrap: wrap;
          gap: 1rem;
        }

        .metrics-header h3 {
          margin: 0;
          font-size: 1.5rem;
          font-weight: 500;
        }

        .header-controls {
          display: flex;
          align-items: center;
          gap: 1rem;
          flex-wrap: wrap;
        }

        .time-range-selector {
          display: flex;
          background: rgba(255, 255, 255, 0.1);
          border-radius: 6px;
          overflow: hidden;
        }

        .time-range-btn {
          padding: 0.5rem 1rem;
          background: transparent;
          border: none;
          color: white;
          cursor: pointer;
          transition: all 0.3s ease;
          font-size: 0.9rem;
        }

        .time-range-btn:hover {
          background: rgba(255, 255, 255, 0.1);
        }

        .time-range-btn.active {
          background: rgba(52, 152, 219, 0.6);
        }

        .toggle-details {
          padding: 0.5rem 1rem;
          background: rgba(255, 255, 255, 0.2);
          border: 1px solid rgba(255, 255, 255, 0.3);
          border-radius: 6px;
          color: white;
          cursor: pointer;
          transition: all 0.3s ease;
          backdrop-filter: blur(5px);
        }

        .toggle-details:hover {
          background: rgba(255, 255, 255, 0.3);
        }

        .convergence-status {
          display: flex;
          justify-content: space-between;
          align-items: center;
          background: rgba(255, 255, 255, 0.05);
          border-radius: 8px;
          padding: 1rem;
          margin-bottom: 1.5rem;
        }

        .status-indicator {
          display: flex;
          align-items: center;
          gap: 0.75rem;
        }

        .status-dot {
          width: 12px;
          height: 12px;
          border-radius: 50%;
          animation: pulse 2s infinite;
        }

        .status-text {
          font-size: 1.1rem;
          font-weight: 600;
        }

        .overall-score {
          font-size: 2rem;
          font-weight: 700;
          color: #3498db;
        }

        .charts-container {
          display: grid;
          grid-template-columns: 2fr 1fr;
          gap: 1.5rem;
          margin-bottom: 2rem;
        }

        .line-chart-container, .radar-chart-container {
          background: rgba(0, 0, 0, 0.2);
          border-radius: 8px;
          padding: 0.5rem;
        }

        .convergence-chart, .radar-chart {
          width: 100%;
          height: auto;
          border-radius: 6px;
        }

        .key-metrics-grid {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
          gap: 1rem;
          margin-bottom: 2rem;
        }

        .metric-card {
          background: rgba(255, 255, 255, 0.05);
          border-radius: 8px;
          padding: 1.25rem;
          border: 1px solid rgba(255, 255, 255, 0.1);
        }

        .metric-card.primary {
          border-color: rgba(52, 152, 219, 0.3);
        }

        .metric-card h4 {
          margin: 0 0 0.75rem 0;
          font-size: 1rem;
          font-weight: 600;
        }

        .metric-value {
          font-size: 2rem;
          font-weight: 700;
          margin-bottom: 0.75rem;
          color: #3498db;
        }

        .metric-bar {
          height: 6px;
          background: rgba(255, 255, 255, 0.2);
          border-radius: 3px;
          overflow: hidden;
          margin-bottom: 0.75rem;
        }

        .metric-fill {
          height: 100%;
          transition: width 0.8s ease;
          border-radius: 3px;
        }

        .metric-fill.internalization { background: linear-gradient(90deg, #3498db, #5dade2); }
        .metric-fill.quality { background: linear-gradient(90deg, #2ecc71, #58d68d); }
        .metric-fill.efficiency { background: linear-gradient(90deg, #f39c12, #f8c471); }
        .metric-fill.curriculum { background: linear-gradient(90deg, #e74c3c, #ec7063); }

        .metric-description {
          font-size: 0.85rem;
          opacity: 0.8;
          line-height: 1.4;
          margin: 0;
        }

        .detailed-metrics {
          display: grid;
          grid-template-columns: 1fr 1fr;
          gap: 2rem;
          padding-top: 1.5rem;
          border-top: 1px solid rgba(255, 255, 255, 0.2);
        }

        .metrics-section h4 {
          margin: 0 0 1rem 0;
          font-size: 1.1rem;
          color: #3498db;
        }

        .sub-metrics {
          display: flex;
          flex-direction: column;
          gap: 1rem;
        }

        .sub-metric {
          display: flex;
          flex-direction: column;
          gap: 0.25rem;
        }

        .sub-metric label {
          font-size: 0.9rem;
          opacity: 0.8;
        }

        .sub-metric value {
          font-size: 1.1rem;
          font-weight: 600;
          color: white;
        }

        .sub-bar {
          height: 4px;
          background: rgba(255, 255, 255, 0.2);
          border-radius: 2px;
          overflow: hidden;
        }

        .sub-fill {
          height: 100%;
          transition: width 0.6s ease;
          border-radius: 2px;
        }

        .sub-fill.rate { background: #9b59b6; }
        .sub-fill.accuracy { background: #e67e22; }
        .sub-fill.diversity { background: #1abc9c; }
        .sub-fill.convergence { background: #f1c40f; }
        .sub-fill.stability { background: #e74c3c; }
        .sub-fill.adaptation { background: #3498db; }

        @keyframes pulse {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.5; }
        }

        @media (max-width: 1200px) {
          .charts-container {
            grid-template-columns: 1fr;
          }

          .detailed-metrics {
            grid-template-columns: 1fr;
          }
        }

        @media (max-width: 768px) {
          .metrics-header {
            flex-direction: column;
            align-items: stretch;
          }

          .header-controls {
            justify-content: space-between;
          }

          .convergence-status {
            flex-direction: column;
            gap: 1rem;
            text-align: center;
          }

          .key-metrics-grid {
            grid-template-columns: 1fr;
          }
        }
      `}</style>
    </div>
  );
};