'use client';

import { useState, useEffect } from 'react';
import { Target, Zap, Settings, Activity, AlertTriangle, CheckCircle, TrendingUp, Database } from 'lucide-react';

interface AssessmentInterfaceProps {
  config: any;
  setConfig: (config: any) => void;
  metrics: any;
  setMetrics: (metrics: any) => void;
}

export default function AssessmentInterface({ config, setConfig, metrics, setMetrics }: AssessmentInterfaceProps) {
  const [isAssessing, setIsAssessing] = useState(false);
  const [frontierModelStatus, setFrontierModelStatus] = useState<{[key: string]: 'idle' | 'active' | 'error'}>({});

  // Simulate frontier model activity
  useEffect(() => {
    if (isAssessing) {
      const interval = setInterval(() => {
        setMetrics((prev: any) => ({
          ...prev,
          currentSuccessRate: Math.min(prev.currentSuccessRate + (Math.random() - 0.4) * 0.05, 1.0),
          edgeOfChaosDetected: Math.random() > 0.7,
          questionsGenerated: prev.questionsGenerated + Math.floor(Math.random() * 3) + 1,
          difficultyLevel: Math.max(1, Math.min(10, prev.difficultyLevel + (Math.random() - 0.5) * 0.5))
        }));
      }, 2000);

      return () => clearInterval(interval);
    }
  }, [isAssessing, setMetrics]);

  const handleStartAssessment = () => {
    setIsAssessing(!isAssessing);
    if (!isAssessing) {
      // Initialize frontier models
      config.frontierModels.forEach((model: string) => {
        setFrontierModelStatus(prev => ({ ...prev, [model]: 'active' }));
      });
    } else {
      setFrontierModelStatus({});
    }
  };

  const renderOpenRouterConfig = () => (
    <div className="bg-white/5 backdrop-blur-lg rounded-xl p-4 border border-white/10 mb-4">
      <h3 className="text-lg font-bold mb-3 flex items-center gap-2">
        <Database className="w-5 h-5 text-blue-400" />
        OpenRouter Frontier Models
      </h3>

      <div className="mb-4">
        <label className="text-sm text-gray-400 block mb-2">OpenRouter API Key</label>
        <input
          type="password"
          value={config.openRouterApiKey}
          onChange={(e) => setConfig({ ...config, openRouterApiKey: e.target.value })}
          placeholder="sk-or-..."
          className="w-full bg-white/10 border border-white/20 rounded px-3 py-2 text-white placeholder-gray-500"
        />
      </div>

      <div className="grid grid-cols-3 gap-3">
        {config.frontierModels.map((model: string) => {
          const status = frontierModelStatus[model] || 'idle';
          return (
            <div key={model} className={`
              p-3 rounded-lg border-2 transition-all
              ${status === 'active' ? 'border-green-400 bg-green-400/10' :
                status === 'error' ? 'border-red-400 bg-red-400/10' :
                'border-gray-600 bg-white/5'
              }
            `}>
              <div className="flex items-center gap-2 mb-1">
                <div className={`w-2 h-2 rounded-full ${
                  status === 'active' ? 'bg-green-400 animate-pulse' :
                  status === 'error' ? 'bg-red-400' :
                  'bg-gray-500'
                }`} />
                <span className="text-sm font-medium">{model}</span>
              </div>
              <div className="text-xs text-gray-400">
                {status === 'active' ? 'Generating questions...' :
                 status === 'error' ? 'Connection failed' :
                 'Standby'}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );

  const renderEdgeOfChaosController = () => (
    <div className="bg-white/5 backdrop-blur-lg rounded-xl p-4 border border-white/10 mb-4">
      <h3 className="text-lg font-bold mb-3 flex items-center gap-2">
        <Target className="w-5 h-5 text-orange-400" />
        Edge-of-Chaos Detection
      </h3>

      <div className="grid grid-cols-2 gap-4 mb-4">
        <div>
          <label className="text-sm text-gray-400 block mb-1">Target Success Rate</label>
          <div className="flex items-center gap-2">
            <input
              type="range"
              min="0.5"
              max="0.9"
              step="0.01"
              value={config.targetSuccessRate}
              onChange={(e) => setConfig({ ...config, targetSuccessRate: parseFloat(e.target.value) })}
              className="flex-1"
            />
            <span className="text-sm font-mono text-orange-400 min-w-[4rem]">
              {(config.targetSuccessRate * 100).toFixed(0)}%
            </span>
          </div>
        </div>

        <div>
          <label className="text-sm text-gray-400 block mb-1">Chaos Threshold</label>
          <div className="flex items-center gap-2">
            <input
              type="range"
              min="0.6"
              max="0.8"
              step="0.01"
              value={config.edgeOfChaosThreshold}
              onChange={(e) => setConfig({ ...config, edgeOfChaosThreshold: parseFloat(e.target.value) })}
              className="flex-1"
            />
            <span className="text-sm font-mono text-orange-400 min-w-[4rem]">
              {(config.edgeOfChaosThreshold * 100).toFixed(0)}%
            </span>
          </div>
        </div>
      </div>

      <div className="bg-black/20 rounded-lg p-3">
        <div className="flex justify-between items-center mb-2">
          <span className="text-sm text-gray-400">Current Success Rate</span>
          <div className="flex items-center gap-2">
            {metrics.edgeOfChaosDetected && (
              <AlertTriangle className="w-4 h-4 text-orange-400" />
            )}
            <span className={`text-sm font-mono ${
              metrics.currentSuccessRate >= config.targetSuccessRate * 0.9 &&
              metrics.currentSuccessRate <= config.targetSuccessRate * 1.1
                ? 'text-green-400' : 'text-orange-400'
            }`}>
              {(metrics.currentSuccessRate * 100).toFixed(1)}%
            </span>
          </div>
        </div>

        <div className="w-full bg-gray-700 rounded-full h-2 mb-2">
          <div
            className="bg-gradient-to-r from-red-400 via-orange-400 to-green-400 h-2 rounded-full transition-all duration-300"
            style={{ width: `${metrics.currentSuccessRate * 100}%` }}
          />
        </div>

        <div className="text-xs text-gray-500 flex justify-between">
          <span>Chaos Zone: {(config.edgeOfChaosThreshold * 100).toFixed(0)}%</span>
          <span>Target: {(config.targetSuccessRate * 100).toFixed(0)}%</span>
        </div>
      </div>
    </div>
  );

  const renderAssessmentMetrics = () => (
    <div className="bg-white/5 backdrop-blur-lg rounded-xl p-4 border border-white/10">
      <h3 className="text-lg font-bold mb-3 flex items-center gap-2">
        <Activity className="w-5 h-5 text-purple-400" />
        Assessment Metrics
      </h3>

      <div className="grid grid-cols-2 gap-4">
        <div className="bg-black/20 rounded-lg p-3">
          <div className="text-xs text-gray-400 mb-1">Questions Generated</div>
          <div className="text-2xl font-bold text-purple-400">{metrics.questionsGenerated}</div>
          <div className="text-xs text-gray-500">Across all frontier models</div>
        </div>

        <div className="bg-black/20 rounded-lg p-3">
          <div className="text-xs text-gray-400 mb-1">Difficulty Level</div>
          <div className="text-2xl font-bold text-blue-400">{metrics.difficultyLevel.toFixed(1)}</div>
          <div className="text-xs text-gray-500">Auto-adaptive scaling</div>
        </div>

        <div className="bg-black/20 rounded-lg p-3">
          <div className="text-xs text-gray-400 mb-1">Chaos Detection</div>
          <div className={`text-2xl font-bold ${metrics.edgeOfChaosDetected ? 'text-orange-400' : 'text-green-400'}`}>
            {metrics.edgeOfChaosDetected ? 'ACTIVE' : 'STABLE'}
          </div>
          <div className="text-xs text-gray-500">Learning zone optimal</div>
        </div>

        <div className="bg-black/20 rounded-lg p-3">
          <div className="text-xs text-gray-400 mb-1">Model Consensus</div>
          <div className="text-2xl font-bold text-cyan-400">
            {Object.values(frontierModelStatus).filter(s => s === 'active').length}/{config.frontierModels.length}
          </div>
          <div className="text-xs text-gray-500">Active frontier models</div>
        </div>
      </div>
    </div>
  );

  return (
    <div className="space-y-4">
      {renderOpenRouterConfig()}
      {renderEdgeOfChaosController()}
      {renderAssessmentMetrics()}

      <div className="bg-white/5 backdrop-blur-lg rounded-xl p-4 border border-white/10">
        <button
          onClick={handleStartAssessment}
          className={`
            w-full px-4 py-3 rounded-lg font-semibold transition-all flex items-center justify-center gap-2
            ${isAssessing
              ? 'bg-red-600 hover:bg-red-700 text-white'
              : 'bg-blue-500 hover:bg-blue-600 text-white'
            }
          `}
        >
          <Target className="w-5 h-5" />
          {isAssessing ? 'Stop Assessment' : 'Start Frontier Assessment'}
        </button>
      </div>
    </div>
  );
}