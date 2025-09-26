'use client';

import { useState, useEffect } from 'react';
import { Play, Pause, Square, RotateCcw, Settings } from 'lucide-react';

interface PhaseControllerProps {
  phaseName: string;
  phaseId: number;
  apiEndpoint: string;
  onStatusChange?: (status: string) => void;
}

export default function PhaseController({
  phaseName,
  phaseId,
  apiEndpoint,
  onStatusChange
}: PhaseControllerProps) {
  const [status, setStatus] = useState<'idle' | 'running' | 'paused' | 'stopped' | 'completed'>('idle');
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [progress, setProgress] = useState(0);

  useEffect(() => {
    if (sessionId && status === 'running') {
      const interval = setInterval(async () => {
        try {
          const response = await fetch(`${apiEndpoint}?sessionId=${sessionId}`);
          const data = await response.json();

          if (data.metrics?.grokProgress) {
            setProgress(data.metrics.grokProgress);
          } else if (data.currentGeneration && data.config?.generations) {
            setProgress((data.currentGeneration / data.config.generations) * 100);
          } else if (data.progress) {
            setProgress(data.progress);
          }

          if (data.status !== status) {
            setStatus(data.status);
            onStatusChange?.(data.status);
          }

          if (data.status === 'completed' || data.status === 'stopped') {
            clearInterval(interval);
          }
        } catch (error) {
          console.error('Failed to fetch status:', error);
        }
      }, 1000);

      return () => clearInterval(interval);
    }
  }, [sessionId, status, apiEndpoint, onStatusChange]);

  const handleStart = async () => {
    setLoading(true);
    try {
      const response = await fetch(apiEndpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          action: 'start',
          config: {} // Use default config
        })
      });

      const data = await response.json();
      if (data.success) {
        setSessionId(data.sessionId);
        setStatus('running');
        setProgress(0);
        onStatusChange?.('running');
      }
    } catch (error) {
      console.error('Failed to start phase:', error);
    }
    setLoading(false);
  };

  const handlePause = async () => {
    if (!sessionId) return;

    setLoading(true);
    try {
      await fetch(apiEndpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action: 'pause', sessionId })
      });
      setStatus('paused');
      onStatusChange?.('paused');
    } catch (error) {
      console.error('Failed to pause phase:', error);
    }
    setLoading(false);
  };

  const handleStop = async () => {
    if (!sessionId) return;

    setLoading(true);
    try {
      await fetch(apiEndpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action: 'stop', sessionId })
      });
      setStatus('stopped');
      setProgress(0);
      onStatusChange?.('stopped');
    } catch (error) {
      console.error('Failed to stop phase:', error);
    }
    setLoading(false);
  };

  const handleResume = async () => {
    if (!sessionId) return;

    setLoading(true);
    try {
      await fetch(apiEndpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action: 'resume', sessionId })
      });
      setStatus('running');
      onStatusChange?.('running');
    } catch (error) {
      console.error('Failed to resume phase:', error);
    }
    setLoading(false);
  };

  const getStatusColor = () => {
    switch (status) {
      case 'running': return 'text-green-400';
      case 'paused': return 'text-yellow-400';
      case 'stopped': return 'text-red-400';
      case 'completed': return 'text-blue-400';
      default: return 'text-gray-400';
    }
  };

  return (
    <div className="bg-white/5 backdrop-blur-lg rounded-2xl p-6 border border-white/10">
      <div className="flex items-center justify-between mb-4">
        <div>
          <h3 className="text-xl font-bold">Phase {phaseId}: {phaseName}</h3>
          <p className={`text-sm ${getStatusColor()}`}>
            Status: {status.charAt(0).toUpperCase() + status.slice(1)}
          </p>
        </div>
        <Settings className="w-6 h-6 text-gray-400 cursor-pointer hover:text-white" />
      </div>

      {/* Progress Bar */}
      <div className="mb-4">
        <div className="flex justify-between text-sm text-gray-400 mb-1">
          <span>Progress</span>
          <span>{Math.round(progress)}%</span>
        </div>
        <div className="w-full bg-gray-700 rounded-full h-2">
          <div
            className="bg-gradient-to-r from-blue-500 to-purple-500 h-2 rounded-full transition-all duration-300"
            style={{ width: `${progress}%` }}
          />
        </div>
      </div>

      {/* Control Buttons */}
      <div className="flex gap-2">
        {status === 'idle' || status === 'stopped' || status === 'completed' ? (
          <button
            onClick={handleStart}
            disabled={loading}
            className="flex-1 bg-green-600 hover:bg-green-700 text-white px-4 py-2 rounded-lg flex items-center justify-center gap-2 transition-colors disabled:opacity-50"
          >
            <Play className="w-4 h-4" />
            Start
          </button>
        ) : status === 'running' ? (
          <>
            <button
              onClick={handlePause}
              disabled={loading}
              className="flex-1 bg-yellow-600 hover:bg-yellow-700 text-white px-4 py-2 rounded-lg flex items-center justify-center gap-2 transition-colors disabled:opacity-50"
            >
              <Pause className="w-4 h-4" />
              Pause
            </button>
            <button
              onClick={handleStop}
              disabled={loading}
              className="flex-1 bg-red-600 hover:bg-red-700 text-white px-4 py-2 rounded-lg flex items-center justify-center gap-2 transition-colors disabled:opacity-50"
            >
              <Square className="w-4 h-4" />
              Stop
            </button>
          </>
        ) : status === 'paused' ? (
          <>
            <button
              onClick={handleResume}
              disabled={loading}
              className="flex-1 bg-green-600 hover:bg-green-700 text-white px-4 py-2 rounded-lg flex items-center justify-center gap-2 transition-colors disabled:opacity-50"
            >
              <Play className="w-4 h-4" />
              Resume
            </button>
            <button
              onClick={handleStop}
              disabled={loading}
              className="flex-1 bg-red-600 hover:bg-red-700 text-white px-4 py-2 rounded-lg flex items-center justify-center gap-2 transition-colors disabled:opacity-50"
            >
              <Square className="w-4 h-4" />
              Stop
            </button>
          </>
        ) : null}

        {status === 'completed' && (
          <button
            onClick={() => {
              setStatus('idle');
              setProgress(0);
              setSessionId(null);
            }}
            className="bg-gray-600 hover:bg-gray-700 text-white px-4 py-2 rounded-lg flex items-center justify-center gap-2 transition-colors"
          >
            <RotateCcw className="w-4 h-4" />
            Reset
          </button>
        )}
      </div>

      {/* Session Info */}
      {sessionId && (
        <div className="mt-4 text-xs text-gray-500">
          Session: {sessionId}
        </div>
      )}
    </div>
  );
}