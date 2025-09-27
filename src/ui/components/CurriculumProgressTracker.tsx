/**
 * CurriculumProgressTracker Component
 * Tracks Fast Quiet-STaR 6-stage curriculum learning progress
 */

import React, { useState, useEffect } from 'react';
import { CurriculumStage } from '../types/phases';

interface CurriculumProgressTrackerProps {
  stages: CurriculumStage[];
  currentStage: number;
  onStageSelect?: (stageId: number) => void;
}

export const CurriculumProgressTracker: React.FC<CurriculumProgressTrackerProps> = ({
  stages,
  currentStage,
  onStageSelect
}) => {
  const [expandedStage, setExpandedStage] = useState<number | null>(currentStage);
  const [animationPhase, setAnimationPhase] = useState(0);

  // Animation for progress bars
  useEffect(() => {
    const interval = setInterval(() => {
      setAnimationPhase(prev => (prev + 1) % 360);
    }, 50);

    return () => clearInterval(interval);
  }, []);

  const getStageColor = (stage: CurriculumStage) => {
    if (stage.isActive) return '#3498db'; // Blue for active
    if (stage.progress >= 100) return '#2ecc71'; // Green for completed
    if (stage.progress > 0) return '#f39c12'; // Orange for in progress
    return '#95a5a6'; // Gray for not started
  };

  const getProgressGradient = (stage: CurriculumStage) => {
    const color = getStageColor(stage);
    const shimmer = stage.isActive ? Math.sin(animationPhase * 0.1) * 0.2 + 0.8 : 1;
    return `linear-gradient(90deg, ${color}${Math.floor(shimmer * 255).toString(16).padStart(2, '0')} ${stage.progress}%, transparent ${stage.progress}%)`;
  };

  const calculateOverallProgress = () => {
    const totalProgress = stages.reduce((sum, stage) => sum + stage.progress, 0);
    return totalProgress / stages.length;
  };

  const getStageDescription = (stage: CurriculumStage) => {
    const descriptions = {
      0: "Learning basic reasoning patterns and token generation fundamentals",
      1: "Developing structured thought processes with improved coherence",
      2: "Mastering multi-step logical reasoning chains and complex arguments",
      3: "Abstract concept formation and high-level pattern recognition",
      4: "Meta-cognitive reasoning - thinking about the thinking process itself",
      5: "Seamless integration of reasoning into natural language generation"
    };
    return descriptions[stage.id as keyof typeof descriptions] || stage.description;
  };

  const getThoughtLengthColor = (current: number, min: number, max: number) => {
    const ratio = (current - min) / (max - min);
    if (ratio < 0.3) return '#e74c3c'; // Red for short thoughts
    if (ratio < 0.7) return '#f39c12'; // Orange for medium thoughts
    return '#2ecc71'; // Green for long thoughts
  };

  return (
    <div className="curriculum-progress-tracker" data-testid="curriculum-progress-tracker">
      <div className="tracker-header">
        <h3>Curriculum Learning Progress</h3>
        <div className="overall-progress">
          <div className="progress-circle">
            <svg width="60" height="60" viewBox="0 0 60 60">
              <circle
                cx="30"
                cy="30"
                r="25"
                fill="none"
                stroke="rgba(255,255,255,0.2)"
                strokeWidth="4"
              />
              <circle
                cx="30"
                cy="30"
                r="25"
                fill="none"
                stroke="url(#progressGradient)"
                strokeWidth="4"
                strokeDasharray={`${(calculateOverallProgress() / 100) * 157} 157`}
                strokeLinecap="round"
                transform="rotate(-90 30 30)"
                style={{
                  transition: 'stroke-dasharray 0.5s ease'
                }}
              />
              <defs>
                <linearGradient id="progressGradient" x1="0%" y1="0%" x2="100%" y2="0%">
                  <stop offset="0%" stopColor="#3498db" />
                  <stop offset="50%" stopColor="#9b59b6" />
                  <stop offset="100%" stopColor="#e91e63" />
                </linearGradient>
              </defs>
            </svg>
            <div className="progress-text">
              {Math.round(calculateOverallProgress())}%
            </div>
          </div>
          <div className="progress-info">
            <div className="current-stage">Stage {currentStage + 1}</div>
            <div className="stage-name">{stages[currentStage]?.name || 'Unknown'}</div>
          </div>
        </div>
      </div>

      <div className="stages-container">
        {stages.map((stage, index) => (
          <div
            key={stage.id}
            className={`stage-card ${stage.isActive ? 'active' : ''} ${stage.progress >= 100 ? 'completed' : ''}`}
            onClick={() => {
              setExpandedStage(expandedStage === stage.id ? null : stage.id);
              onStageSelect?.(stage.id);
            }}
            data-testid={`stage-${stage.id}`}
          >
            <div className="stage-header">
              <div className="stage-number">
                <div className="number-circle" style={{ backgroundColor: getStageColor(stage) }}>
                  {stage.progress >= 100 ? '✓' : index + 1}
                </div>
              </div>

              <div className="stage-info">
                <h4 className="stage-title">{stage.name}</h4>
                <div className="stage-progress-bar">
                  <div
                    className="progress-fill"
                    style={{
                      background: getProgressGradient(stage),
                      transition: 'all 0.3s ease'
                    }}
                  />
                  <span className="progress-percentage">{stage.progress.toFixed(1)}%</span>
                </div>
              </div>

              <div className="stage-status">
                {stage.isActive && (
                  <div className="status-indicator active">
                    <div className="pulse-dot" />
                    ACTIVE
                  </div>
                )}
                {!stage.isActive && stage.progress >= 100 && (
                  <div className="status-indicator completed">COMPLETED</div>
                )}
                {!stage.isActive && stage.progress > 0 && stage.progress < 100 && (
                  <div className="status-indicator paused">PAUSED</div>
                )}
                {stage.progress === 0 && (
                  <div className="status-indicator pending">PENDING</div>
                )}
              </div>
            </div>

            {expandedStage === stage.id && (
              <div className="stage-details" data-testid={`stage-details-${stage.id}`}>
                <p className="stage-description">{getStageDescription(stage)}</p>

                <div className="metrics-grid">
                  <div className="metric-card">
                    <h5>Thought Length Range</h5>
                    <div className="thought-length-bar">
                      <div className="length-scale">
                        <div className="scale-min">{stage.thoughtLength.min}</div>
                        <div className="scale-max">{stage.thoughtLength.max}</div>
                      </div>
                      <div className="length-indicator">
                        <div
                          className="current-length"
                          style={{
                            left: `${((stage.thoughtLength.current - stage.thoughtLength.min) /
                                    (stage.thoughtLength.max - stage.thoughtLength.min)) * 100}%`,
                            backgroundColor: getThoughtLengthColor(
                              stage.thoughtLength.current,
                              stage.thoughtLength.min,
                              stage.thoughtLength.max
                            )
                          }}
                        />
                      </div>
                      <div className="current-value">{stage.thoughtLength.current} tokens</div>
                    </div>
                  </div>

                  <div className="metric-card">
                    <h5>Performance Metrics</h5>
                    <div className="performance-bars">
                      <div className="perf-metric">
                        <label>Accuracy</label>
                        <div className="perf-bar">
                          <div
                            className="perf-fill accuracy"
                            style={{ width: `${stage.metrics.accuracy}%` }}
                          />
                          <span>{stage.metrics.accuracy.toFixed(1)}%</span>
                        </div>
                      </div>
                      <div className="perf-metric">
                        <label>Efficiency</label>
                        <div className="perf-bar">
                          <div
                            className="perf-fill efficiency"
                            style={{ width: `${stage.metrics.efficiency}%` }}
                          />
                          <span>{stage.metrics.efficiency.toFixed(1)}%</span>
                        </div>
                      </div>
                      <div className="perf-metric">
                        <label>Internalization</label>
                        <div className="perf-bar">
                          <div
                            className="perf-fill internalization"
                            style={{ width: `${stage.metrics.internalization}%` }}
                          />
                          <span>{stage.metrics.internalization.toFixed(1)}%</span>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>

                <div className="stage-timeline">
                  <h5>Learning Timeline</h5>
                  <div className="timeline-stages">
                    {['Foundation', 'Development', 'Mastery', 'Integration'].map((phase, phaseIndex) => (
                      <div
                        key={phase}
                        className={`timeline-phase ${phaseIndex <= (stage.progress / 25) ? 'completed' : ''}`}
                      >
                        <div className="phase-dot" />
                        <span className="phase-label">{phase}</span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            )}
          </div>
        ))}
      </div>

      <div className="curriculum-summary" data-testid="curriculum-summary">
        <h4>Fast Quiet-STaR Curriculum</h4>
        <div className="summary-stats">
          <div className="stat">
            <label>Completed Stages</label>
            <value>{stages.filter(s => s.progress >= 100).length} / {stages.length}</value>
          </div>
          <div className="stat">
            <label>Average Progress</label>
            <value>{Math.round(calculateOverallProgress())}%</value>
          </div>
          <div className="stat">
            <label>Current Focus</label>
            <value>{stages.find(s => s.isActive)?.name || 'None'}</value>
          </div>
          <div className="stat">
            <label>Next Milestone</label>
            <value>
              {(() => {
                const nextStage = stages.find(s => s.progress < 100 && !s.isActive);
                return nextStage ? nextStage.name : 'Completed';
              })()}
            </value>
          </div>
        </div>
      </div>

      <style>{`
        .curriculum-progress-tracker {
          background: rgba(255, 255, 255, 0.1);
          border-radius: 12px;
          padding: 1.5rem;
          backdrop-filter: blur(10px);
          border: 1px solid rgba(255, 255, 255, 0.2);
          color: white;
        }

        .tracker-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 2rem;
          flex-wrap: wrap;
          gap: 1rem;
        }

        .tracker-header h3 {
          margin: 0;
          font-size: 1.5rem;
          font-weight: 500;
        }

        .overall-progress {
          display: flex;
          align-items: center;
          gap: 1rem;
        }

        .progress-circle {
          position: relative;
          display: flex;
          align-items: center;
          justify-content: center;
        }

        .progress-text {
          position: absolute;
          font-size: 1.2rem;
          font-weight: 600;
          color: white;
        }

        .progress-info {
          text-align: left;
        }

        .current-stage {
          font-size: 0.9rem;
          opacity: 0.8;
          margin-bottom: 0.25rem;
        }

        .stage-name {
          font-size: 1.1rem;
          font-weight: 600;
        }

        .stages-container {
          display: flex;
          flex-direction: column;
          gap: 1rem;
          margin-bottom: 2rem;
        }

        .stage-card {
          background: rgba(255, 255, 255, 0.05);
          border-radius: 8px;
          border: 1px solid rgba(255, 255, 255, 0.1);
          cursor: pointer;
          transition: all 0.3s ease;
          overflow: hidden;
        }

        .stage-card:hover {
          background: rgba(255, 255, 255, 0.1);
          transform: translateY(-2px);
          box-shadow: 0 4px 12px rgba(0,0,0,0.2);
        }

        .stage-card.active {
          border-color: #3498db;
          box-shadow: 0 0 20px rgba(52, 152, 219, 0.3);
        }

        .stage-card.completed {
          border-color: #2ecc71;
        }

        .stage-header {
          display: flex;
          align-items: center;
          padding: 1rem;
          gap: 1rem;
        }

        .stage-number {
          flex-shrink: 0;
        }

        .number-circle {
          width: 40px;
          height: 40px;
          border-radius: 50%;
          display: flex;
          align-items: center;
          justify-content: center;
          color: white;
          font-weight: 600;
          font-size: 1.1rem;
        }

        .stage-info {
          flex: 1;
          min-width: 0;
        }

        .stage-title {
          margin: 0 0 0.5rem 0;
          font-size: 1.1rem;
          font-weight: 600;
        }

        .stage-progress-bar {
          position: relative;
          height: 8px;
          background: rgba(255, 255, 255, 0.2);
          border-radius: 4px;
          overflow: hidden;
        }

        .progress-fill {
          height: 100%;
          transition: width 0.5s ease;
        }

        .progress-percentage {
          position: absolute;
          right: 0;
          top: -25px;
          font-size: 0.8rem;
          color: white;
        }

        .stage-status {
          flex-shrink: 0;
        }

        .status-indicator {
          padding: 0.25rem 0.75rem;
          border-radius: 12px;
          font-size: 0.7rem;
          font-weight: 600;
          letter-spacing: 0.5px;
          display: flex;
          align-items: center;
          gap: 0.5rem;
        }

        .status-indicator.active {
          background: rgba(52, 152, 219, 0.3);
          color: #3498db;
        }

        .status-indicator.completed {
          background: rgba(46, 204, 113, 0.3);
          color: #2ecc71;
        }

        .status-indicator.paused {
          background: rgba(243, 156, 18, 0.3);
          color: #f39c12;
        }

        .status-indicator.pending {
          background: rgba(149, 165, 166, 0.3);
          color: #95a5a6;
        }

        .pulse-dot {
          width: 8px;
          height: 8px;
          background: #3498db;
          border-radius: 50%;
          animation: pulse 2s infinite;
        }

        .stage-details {
          padding: 0 1rem 1rem 1rem;
          border-top: 1px solid rgba(255, 255, 255, 0.1);
          margin-top: 1rem;
        }

        .stage-description {
          margin: 1rem 0;
          opacity: 0.9;
          line-height: 1.5;
        }

        .metrics-grid {
          display: grid;
          grid-template-columns: 1fr 1fr;
          gap: 1rem;
          margin: 1rem 0;
        }

        .metric-card {
          background: rgba(255, 255, 255, 0.05);
          border-radius: 6px;
          padding: 1rem;
        }

        .metric-card h5 {
          margin: 0 0 0.75rem 0;
          font-size: 0.9rem;
          opacity: 0.8;
        }

        .thought-length-bar {
          display: flex;
          flex-direction: column;
          gap: 0.5rem;
        }

        .length-scale {
          display: flex;
          justify-content: space-between;
          font-size: 0.8rem;
          opacity: 0.7;
        }

        .length-indicator {
          position: relative;
          height: 6px;
          background: rgba(255, 255, 255, 0.2);
          border-radius: 3px;
        }

        .current-length {
          position: absolute;
          top: -4px;
          width: 8px;
          height: 14px;
          border-radius: 2px;
          transform: translateX(-50%);
        }

        .current-value {
          text-align: center;
          font-size: 0.8rem;
          font-weight: 600;
        }

        .performance-bars {
          display: flex;
          flex-direction: column;
          gap: 0.5rem;
        }

        .perf-metric {
          display: flex;
          flex-direction: column;
          gap: 0.25rem;
        }

        .perf-metric label {
          font-size: 0.8rem;
          opacity: 0.8;
        }

        .perf-bar {
          position: relative;
          height: 6px;
          background: rgba(255, 255, 255, 0.2);
          border-radius: 3px;
          overflow: hidden;
        }

        .perf-fill {
          height: 100%;
          transition: width 0.5s ease;
        }

        .perf-fill.accuracy { background: #e74c3c; }
        .perf-fill.efficiency { background: #f39c12; }
        .perf-fill.internalization { background: #2ecc71; }

        .perf-bar span {
          position: absolute;
          right: 0;
          top: -20px;
          font-size: 0.7rem;
        }

        .stage-timeline {
          margin-top: 1rem;
        }

        .stage-timeline h5 {
          margin: 0 0 0.75rem 0;
          font-size: 0.9rem;
          opacity: 0.8;
        }

        .timeline-stages {
          display: flex;
          justify-content: space-between;
          position: relative;
        }

        .timeline-stages::before {
          content: '';
          position: absolute;
          top: 8px;
          left: 8px;
          right: 8px;
          height: 2px;
          background: rgba(255, 255, 255, 0.2);
        }

        .timeline-phase {
          display: flex;
          flex-direction: column;
          align-items: center;
          gap: 0.5rem;
          position: relative;
        }

        .phase-dot {
          width: 16px;
          height: 16px;
          border-radius: 50%;
          background: rgba(255, 255, 255, 0.3);
          border: 2px solid rgba(255, 255, 255, 0.5);
          transition: all 0.3s ease;
        }

        .timeline-phase.completed .phase-dot {
          background: #2ecc71;
          border-color: #2ecc71;
        }

        .phase-label {
          font-size: 0.7rem;
          opacity: 0.8;
          white-space: nowrap;
        }

        .curriculum-summary {
          background: rgba(255, 255, 255, 0.05);
          border-radius: 8px;
          padding: 1rem;
          border: 1px solid rgba(255, 255, 255, 0.1);
        }

        .curriculum-summary h4 {
          margin: 0 0 1rem 0;
          font-size: 1.1rem;
        }

        .summary-stats {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
          gap: 1rem;
        }

        .stat {
          display: flex;
          flex-direction: column;
          gap: 0.25rem;
        }

        .stat label {
          font-size: 0.8rem;
          opacity: 0.7;
        }

        .stat value {
          font-size: 1rem;
          font-weight: 600;
          color: #3498db;
        }

        @keyframes pulse {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.5; }
        }

        @media (max-width: 768px) {
          .tracker-header {
            flex-direction: column;
            text-align: center;
          }

          .metrics-grid {
            grid-template-columns: 1fr;
          }

          .summary-stats {
            grid-template-columns: repeat(2, 1fr);
          }
        }
      `}</style>
    </div>
  );
};