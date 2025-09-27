/**
 * ReasoningTokenVisualizer Component
 * Displays reasoning token streams with <|startofthought|> and <|endofthought|> visualization
 */

import React, { useRef, useEffect, useState, useMemo } from 'react';
import { ThoughtStream, ReasoningToken } from '../types/phases';

interface ReasoningTokenVisualizerProps {
  streams: ThoughtStream[];
  selectedStream: number;
  onStreamSelect: (streamId: number) => void;
  visualizationMode: '2d' | '3d';
}

export const ReasoningTokenVisualizer: React.FC<ReasoningTokenVisualizerProps> = ({
  streams,
  selectedStream,
  onStreamSelect,
  visualizationMode
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationRef = useRef<number>();
  const [autoScroll, setAutoScroll] = useState(true);
  const [showTokenDetails, setShowTokenDetails] = useState(false);
  const [selectedToken, setSelectedToken] = useState<ReasoningToken | null>(null);

  const selectedStreamData = useMemo(() =>
    streams.find(s => s.id === selectedStream) || streams[0],
    [streams, selectedStream]
  );

  const thoughtGroups = useMemo(() => {
    if (!selectedStreamData) return [];

    const groups: Array<{
      id: string;
      startToken: ReasoningToken;
      endToken?: ReasoningToken;
      contentTokens: ReasoningToken[];
      isComplete: boolean;
    }> = [];

    let currentGroup: any = null;

    selectedStreamData.tokens.forEach(token => {
      if (token.type === 'thought_start') {
        if (currentGroup && !currentGroup.isComplete) {
          groups.push(currentGroup);
        }
        currentGroup = {
          id: `thought-${token.id}`,
          startToken: token,
          contentTokens: [],
          isComplete: false
        };
      } else if (token.type === 'thought_end' && currentGroup) {
        currentGroup.endToken = token;
        currentGroup.isComplete = true;
        groups.push(currentGroup);
        currentGroup = null;
      } else if (token.type === 'thought_content' && currentGroup) {
        currentGroup.contentTokens.push(token);
      }
    });

    if (currentGroup && !currentGroup.isComplete) {
      groups.push(currentGroup);
    }

    return groups;
  }, [selectedStreamData]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !selectedStreamData) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const draw = () => {
      const { width, height } = canvas;
      ctx.clearRect(0, 0, width, height);

      if (visualizationMode === '2d') {
        draw2D(ctx, width, height);
      } else {
        draw3D(ctx, width, height);
      }
    };

    const draw2D = (ctx: CanvasRenderingContext2D, width: number, height: number) => {
      const tokens = selectedStreamData.tokens;
      const tokenHeight = 30;
      const tokenWidth = Math.max(80, width / Math.max(tokens.length, 20));
      const startY = height / 2;

      // Draw timeline
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.3)';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(0, startY);
      ctx.lineTo(width, startY);
      ctx.stroke();

      // Draw thought groups
      thoughtGroups.forEach((group, groupIndex) => {
        const groupColor = `hsl(${(groupIndex * 60) % 360}, 70%, 60%)`;

        // Draw thought boundary
        if (group.isComplete && group.endToken) {
          const startX = group.startToken.position * tokenWidth;
          const endX = group.endToken.position * tokenWidth;

          ctx.fillStyle = groupColor + '20';
          ctx.fillRect(startX, startY - tokenHeight/2, endX - startX, tokenHeight);

          ctx.strokeStyle = groupColor;
          ctx.lineWidth = 2;
          ctx.strokeRect(startX, startY - tokenHeight/2, endX - startX, tokenHeight);
        }

        // Draw tokens
        [...(group.startToken ? [group.startToken] : []),
         ...group.contentTokens,
         ...(group.endToken ? [group.endToken] : [])].forEach(token => {
          const x = token.position * tokenWidth;
          const y = startY;

          // Token background
          ctx.fillStyle = getTokenColor(token.type, token.confidence);
          ctx.beginPath();
          ctx.arc(x, y, 8, 0, Math.PI * 2);
          ctx.fill();

          // Token border
          ctx.strokeStyle = 'white';
          ctx.lineWidth = 2;
          ctx.stroke();

          // Confidence indicator
          const confidenceHeight = token.confidence * 20;
          ctx.fillStyle = `rgba(255, 255, 255, ${token.confidence})`;
          ctx.fillRect(x - 2, y - confidenceHeight/2, 4, confidenceHeight);
        });
      });

      // Draw stream info
      ctx.fillStyle = 'white';
      ctx.font = '12px monospace';
      ctx.fillText(`Stream ${selectedStream} | Tokens: ${tokens.length} | Active: ${selectedStreamData.isActive}`, 10, 20);
      ctx.fillText(`Temperature: ${selectedStreamData.temperature.toFixed(2)} | Convergence: ${(selectedStreamData.convergenceScore * 100).toFixed(1)}%`, 10, 35);
    };

    const draw3D = (ctx: CanvasRenderingContext2D, width: number, height: number) => {
      const time = Date.now() * 0.001;
      const centerX = width / 2;
      const centerY = height / 2;

      // Draw 3D thought flows
      thoughtGroups.forEach((group, groupIndex) => {
        const angle = (groupIndex / thoughtGroups.length) * Math.PI * 2;
        const radius = 100 + Math.sin(time + groupIndex) * 20;

        const x = centerX + Math.cos(angle) * radius;
        const y = centerY + Math.sin(angle) * radius;

        // Draw thought spiral
        ctx.strokeStyle = `hsl(${(groupIndex * 60) % 360}, 70%, 60%)`;
        ctx.lineWidth = 3;
        ctx.beginPath();

        for (let i = 0; i <= group.contentTokens.length; i++) {
          const t = i / Math.max(group.contentTokens.length, 1);
          const spiralAngle = angle + t * Math.PI * 4;
          const spiralRadius = radius * (0.8 + t * 0.4);
          const spiralX = centerX + Math.cos(spiralAngle) * spiralRadius;
          const spiralY = centerY + Math.sin(spiralAngle) * spiralRadius + Math.sin(time * 2 + t * 10) * 10;

          if (i === 0) {
            ctx.moveTo(spiralX, spiralY);
          } else {
            ctx.lineTo(spiralX, spiralY);
          }
        }
        ctx.stroke();

        // Draw thought center
        ctx.fillStyle = group.isComplete ?
          `hsl(${(groupIndex * 60) % 360}, 70%, 60%)` :
          'rgba(255, 255, 255, 0.5)';
        ctx.beginPath();
        ctx.arc(x, y, group.isComplete ? 12 : 8, 0, Math.PI * 2);
        ctx.fill();

        // Draw thought content visualization
        if (group.contentTokens.length > 0) {
          ctx.fillStyle = 'white';
          ctx.font = '10px monospace';
          ctx.textAlign = 'center';
          ctx.fillText(group.contentTokens.length.toString(), x, y + 3);
        }
      });

      // Draw connection lines between thoughts
      for (let i = 0; i < thoughtGroups.length - 1; i++) {
        const group1 = thoughtGroups[i];
        const group2 = thoughtGroups[i + 1];

        if (group1.isComplete && group2.isComplete) {
          const angle1 = (i / thoughtGroups.length) * Math.PI * 2;
          const angle2 = ((i + 1) / thoughtGroups.length) * Math.PI * 2;
          const radius = 100;

          const x1 = centerX + Math.cos(angle1) * radius;
          const y1 = centerY + Math.sin(angle1) * radius;
          const x2 = centerX + Math.cos(angle2) * radius;
          const y2 = centerY + Math.sin(angle2) * radius;

          ctx.strokeStyle = 'rgba(255, 255, 255, 0.3)';
          ctx.lineWidth = 1;
          ctx.setLineDash([5, 5]);
          ctx.beginPath();
          ctx.moveTo(x1, y1);
          ctx.lineTo(x2, y2);
          ctx.stroke();
          ctx.setLineDash([]);
        }
      }
    };

    const animate = () => {
      draw();
      if (visualizationMode === '3d') {
        animationRef.current = requestAnimationFrame(animate);
      }
    };

    animate();

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [selectedStreamData, visualizationMode, thoughtGroups, selectedStream]);

  const getTokenColor = (type: ReasoningToken['type'], confidence: number) => {
    const alpha = Math.max(0.6, confidence);
    switch (type) {
      case 'thought_start':
        return `rgba(46, 204, 113, ${alpha})`; // Green
      case 'thought_end':
        return `rgba(231, 76, 60, ${alpha})`; // Red
      case 'thought_content':
        return `rgba(52, 152, 219, ${alpha})`; // Blue
      default:
        return `rgba(149, 165, 166, ${alpha})`; // Gray
    }
  };

  const handleCanvasClick = (event: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas || !selectedStreamData) return;

    const rect = canvas.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;

    // Find clicked token (simplified hit detection)
    const tokenWidth = Math.max(80, canvas.width / Math.max(selectedStreamData.tokens.length, 20));
    const clickedTokenIndex = Math.floor(x / tokenWidth);
    const token = selectedStreamData.tokens[clickedTokenIndex];

    if (token) {
      setSelectedToken(token);
      setShowTokenDetails(true);
    }
  };

  return (
    <div className="reasoning-token-visualizer" data-testid="reasoning-token-visualizer">
      <div className="visualizer-header">
        <h3>Reasoning Token Stream</h3>
        <div className="header-controls">
          <div className="stream-selector">
            <label htmlFor="stream-select">Stream:</label>
            <select
              id="stream-select"
              value={selectedStream}
              onChange={(e) => onStreamSelect(parseInt(e.target.value))}
            >
              {streams.map(stream => (
                <option key={stream.id} value={stream.id}>
                  Stream {stream.id} ({stream.tokens.length} tokens)
                </option>
              ))}
            </select>
          </div>

          <label>
            <input
              type="checkbox"
              checked={autoScroll}
              onChange={(e) => setAutoScroll(e.target.checked)}
            />
            Auto-scroll
          </label>

          <button
            className="details-toggle"
            onClick={() => setShowTokenDetails(!showTokenDetails)}
          >
            {showTokenDetails ? 'Hide' : 'Show'} Details
          </button>
        </div>
      </div>

      <div className="visualizer-content">
        <canvas
          ref={canvasRef}
          width={800}
          height={400}
          onClick={handleCanvasClick}
          className="token-canvas"
        />

        {showTokenDetails && (
          <div className="token-details" data-testid="token-details">
            <h4>Token Details</h4>
            {selectedToken ? (
              <div className="selected-token">
                <div><strong>ID:</strong> {selectedToken.id}</div>
                <div><strong>Type:</strong> {selectedToken.type}</div>
                <div><strong>Content:</strong> {selectedToken.content}</div>
                <div><strong>Position:</strong> {selectedToken.position}</div>
                <div><strong>Confidence:</strong> {(selectedToken.confidence * 100).toFixed(1)}%</div>
                <div><strong>Stream:</strong> {selectedToken.streamId}</div>
                <div><strong>Timestamp:</strong> {new Date(selectedToken.timestamp).toLocaleTimeString()}</div>
              </div>
            ) : (
              <div className="no-selection">Click on a token to view details</div>
            )}
          </div>
        )}
      </div>

      <div className="thought-groups-summary" data-testid="thought-groups">
        <h4>Thought Groups ({thoughtGroups.length})</h4>
        <div className="groups-list">
          {thoughtGroups.map((group, index) => (
            <div key={group.id} className={`thought-group ${group.isComplete ? 'complete' : 'incomplete'}`}>
              <div className="group-header">
                <span className="group-id">Group {index + 1}</span>
                <span className={`status ${group.isComplete ? 'complete' : 'incomplete'}`}>
                  {group.isComplete ? 'Complete' : 'In Progress'}
                </span>
              </div>
              <div className="group-stats">
                <span>Tokens: {group.contentTokens.length}</span>
                {group.startToken && (
                  <span>Start: {group.startToken.confidence.toFixed(2)}</span>
                )}
                {group.endToken && (
                  <span>End: {group.endToken.confidence.toFixed(2)}</span>
                )}
              </div>
            </div>
          ))}
        </div>
      </div>

      <style>{`
        .reasoning-token-visualizer {
          background: rgba(255, 255, 255, 0.1);
          border-radius: 12px;
          padding: 1.5rem;
          backdrop-filter: blur(10px);
          border: 1px solid rgba(255, 255, 255, 0.2);
        }

        .visualizer-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 1.5rem;
          flex-wrap: wrap;
          gap: 1rem;
        }

        .visualizer-header h3 {
          margin: 0;
          font-size: 1.5rem;
          font-weight: 500;
          color: white;
        }

        .header-controls {
          display: flex;
          align-items: center;
          gap: 1rem;
          flex-wrap: wrap;
        }

        .stream-selector {
          display: flex;
          align-items: center;
          gap: 0.5rem;
        }

        .stream-selector label {
          color: white;
          font-size: 0.9rem;
        }

        .stream-selector select {
          padding: 0.5rem;
          background: rgba(255, 255, 255, 0.2);
          border: 1px solid rgba(255, 255, 255, 0.3);
          border-radius: 6px;
          color: white;
          backdrop-filter: blur(5px);
        }

        .stream-selector select option {
          background: #764ba2;
          color: white;
        }

        .header-controls label {
          display: flex;
          align-items: center;
          gap: 0.5rem;
          color: white;
          font-size: 0.9rem;
          cursor: pointer;
        }

        .details-toggle {
          padding: 0.5rem 1rem;
          background: rgba(255, 255, 255, 0.2);
          border: 1px solid rgba(255, 255, 255, 0.3);
          border-radius: 6px;
          color: white;
          cursor: pointer;
          transition: all 0.3s ease;
          backdrop-filter: blur(5px);
        }

        .details-toggle:hover {
          background: rgba(255, 255, 255, 0.3);
        }

        .visualizer-content {
          display: grid;
          grid-template-columns: 1fr auto;
          gap: 1.5rem;
          margin-bottom: 1.5rem;
        }

        .token-canvas {
          background: rgba(0, 0, 0, 0.3);
          border-radius: 8px;
          cursor: crosshair;
          width: 100%;
          height: 400px;
        }

        .token-details {
          min-width: 250px;
          background: rgba(255, 255, 255, 0.1);
          border-radius: 8px;
          padding: 1rem;
          border: 1px solid rgba(255, 255, 255, 0.2);
        }

        .token-details h4 {
          margin: 0 0 1rem 0;
          color: white;
          font-size: 1.1rem;
        }

        .selected-token div {
          margin-bottom: 0.5rem;
          color: white;
          font-size: 0.9rem;
        }

        .no-selection {
          color: rgba(255, 255, 255, 0.7);
          font-style: italic;
          text-align: center;
          padding: 2rem 0;
        }

        .thought-groups-summary {
          border-top: 1px solid rgba(255, 255, 255, 0.2);
          padding-top: 1.5rem;
        }

        .thought-groups-summary h4 {
          margin: 0 0 1rem 0;
          color: white;
          font-size: 1.1rem;
        }

        .groups-list {
          display: grid;
          grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
          gap: 1rem;
        }

        .thought-group {
          background: rgba(255, 255, 255, 0.1);
          border-radius: 8px;
          padding: 1rem;
          border: 1px solid rgba(255, 255, 255, 0.2);
        }

        .thought-group.complete {
          border-color: rgba(46, 204, 113, 0.5);
        }

        .thought-group.incomplete {
          border-color: rgba(241, 196, 15, 0.5);
        }

        .group-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 0.5rem;
        }

        .group-id {
          color: white;
          font-weight: 600;
        }

        .status {
          padding: 0.25rem 0.5rem;
          border-radius: 4px;
          font-size: 0.8rem;
          font-weight: 500;
        }

        .status.complete {
          background: rgba(46, 204, 113, 0.3);
          color: #2ecc71;
        }

        .status.incomplete {
          background: rgba(241, 196, 15, 0.3);
          color: #f1c40f;
        }

        .group-stats {
          display: flex;
          flex-direction: column;
          gap: 0.25rem;
          font-size: 0.8rem;
          color: rgba(255, 255, 255, 0.8);
        }

        @media (max-width: 1200px) {
          .visualizer-content {
            grid-template-columns: 1fr;
          }

          .token-details {
            min-width: auto;
          }
        }

        @media (max-width: 768px) {
          .visualizer-header {
            flex-direction: column;
            align-items: stretch;
          }

          .header-controls {
            justify-content: space-between;
          }

          .groups-list {
            grid-template-columns: 1fr;
          }
        }
      `}</style>
    </div>
  );
};