'use client';

import { useState, useEffect } from 'react';
import { Brain, Thermometer, Target, Lightbulb, TrendingUp, Settings, Gauge, Zap } from 'lucide-react';

interface SelfModelingInterfaceProps {
  config: any;
  setConfig: (config: any) => void;
  metrics: any;
  setMetrics: (metrics: any) => void;
}

export default function SelfModelingInterface({ config, setConfig, metrics, setMetrics }: SelfModelingInterfaceProps) {
  const [currentTemperature, setCurrentTemperature] = useState(1.0);
  const [predictionGaps, setPredictionGaps] = useState<{[key: string]: number}>({});
  const [isModeling, setIsModeling] = useState(false);

  // Simulate self-modeling activity
  useEffect(() => {
    if (isModeling) {
      const interval = setInterval(() => {
        setMetrics((prev: any) => ({
          ...prev,
          selfPredictionAccuracy: Math.min(Math.max(prev.selfPredictionAccuracy + (Math.random() - 0.45) * 0.1, 0), 1),
          temperatureRange: config.temperatureRanges[Math.floor(Math.random() * config.temperatureRanges.length)],
          grockingSelfDetected: Math.random() > 0.8,
          personaDevelopmentScore: Math.min(prev.personaDevelopmentScore + Math.random() * 5, 100)
        }));

        // Update prediction gaps
        const gaps = ['attention', 'activation', 'gradient', 'loss'];
        const newGaps: {[key: string]: number} = {};
        gaps.forEach(gap => {
          newGaps[gap] = Math.random();
        });
        setPredictionGaps(newGaps);
      }, 2000);

      return () => clearInterval(interval);
    }
  }, [isModeling, setMetrics, config.temperatureRanges]);

  const renderTemperatureCurriculum = () => (
    <div className="bg-white/5 backdrop-blur-lg rounded-xl p-4 border border-white/10 mb-4">
      <h3 className="text-lg font-bold mb-3 flex items-center gap-2">
        <Thermometer className="w-5 h-5 text-red-400" />
        Temperature Curriculum System
      </h3>

      <div className="mb-4">
        <label className="text-sm text-gray-400 block mb-2">Temperature Ranges</label>
        <div className="grid grid-cols-5 gap-2">
          {config.temperatureRanges.map((temp: number, index: number) => (
            <div key={index} className="relative">
              <input
                type="number"
                step="0.1"
                min="0.1"
                max="3.0"
                value={temp}
                onChange={(e) => {
                  const newRanges = [...config.temperatureRanges];
                  newRanges[index] = parseFloat(e.target.value);
                  setConfig({ ...config, temperatureRanges: newRanges });
                }}
                className="w-full bg-white/10 border border-white/20 rounded px-2 py-1 text-white text-sm"
              />
              <div className="text-xs text-gray-500 mt-1 text-center">T{index + 1}</div>
            </div>
          ))}
        </div>
      </div>

      <div className="mb-4">
        <div className="flex justify-between items-center mb-2">
          <label className="text-sm text-gray-400">Current Temperature</label>
          <span className="text-sm font-mono text-red-400">{currentTemperature.toFixed(2)}</span>
        </div>
        <input
          type="range"
          min="0.1"
          max="3.0"
          step="0.1"
          value={currentTemperature}
          onChange={(e) => setCurrentTemperature(parseFloat(e.target.value))}
          className="w-full"
        />
        <div className="flex justify-between text-xs text-gray-500 mt-1">
          <span>Conservative (0.1)</span>
          <span>Balanced (1.0)</span>
          <span>Creative (3.0)</span>
        </div>
      </div>

      <div className="bg-black/20 rounded-lg p-3">
        <div className="text-sm text-gray-400 mb-2">Temperature Impact Visualization</div>
        <div className="grid grid-cols-3 gap-2 text-xs">
          <div className="text-center">
            <div className="text-blue-400 font-mono">{(currentTemperature * 0.3).toFixed(2)}</div>
            <div className="text-gray-500">Attention Sharpness</div>
          </div>
          <div className="text-center">
            <div className="text-green-400 font-mono">{(currentTemperature * 1.2).toFixed(2)}</div>
            <div className="text-gray-500">Output Diversity</div>
          </div>
          <div className="text-center">
            <div className="text-purple-400 font-mono">{(currentTemperature * 0.8).toFixed(2)}</div>
            <div className="text-gray-500">Exploration Rate</div>
          </div>
        </div>
      </div>
    </div>
  );

  const renderSelfPrediction = () => (
    <div className="bg-white/5 backdrop-blur-lg rounded-xl p-4 border border-white/10 mb-4">
      <h3 className="text-lg font-bold mb-3 flex items-center gap-2">
        <Target className="w-5 h-5 text-blue-400" />
        Self-Prediction & Gap Analysis
      </h3>

      <div className="mb-4">
        <div className="flex justify-between items-center mb-2">
          <label className="text-sm text-gray-400">Prediction Area Width</label>
          <span className="text-sm font-mono text-blue-400">{config.selfPredictionAreaWidth} tokens</span>
        </div>
        <input
          type="range"
          min="10"
          max="100"
          value={config.selfPredictionAreaWidth}
          onChange={(e) => setConfig({ ...config, selfPredictionAreaWidth: parseInt(e.target.value) })}
          className="w-full"
        />
      </div>

      <div className="space-y-2">
        {Object.entries(predictionGaps).map(([type, accuracy]) => (
          <div key={type} className="bg-black/20 rounded-lg p-3">
            <div className="flex justify-between items-center mb-2">
              <span className="text-sm font-medium capitalize">{type} Prediction</span>
              <span className={`text-sm font-mono ${
                accuracy > 0.8 ? 'text-green-400' :
                accuracy > 0.6 ? 'text-yellow-400' :
                'text-red-400'
              }`}>
                {(accuracy * 100).toFixed(1)}%
              </span>
            </div>
            <div className="w-full bg-gray-700 rounded-full h-2">
              <div
                className={`h-2 rounded-full transition-all duration-300 ${
                  accuracy > 0.8 ? 'bg-green-400' :
                  accuracy > 0.6 ? 'bg-yellow-400' :
                  'bg-red-400'
                }`}
                style={{ width: `${accuracy * 100}%` }}
              />
            </div>
          </div>
        ))}
      </div>
    </div>
  );

  const renderPromptBaking = () => (
    <div className="bg-white/5 backdrop-blur-lg rounded-xl p-4 border border-white/10 mb-4">
      <h3 className="text-lg font-bold mb-3 flex items-center gap-2">
        <Lightbulb className="w-5 h-5 text-yellow-400" />
        Prompt Baking & Persona Development
      </h3>

      <div className="mb-4">
        <div className="flex items-center gap-2 mb-2">
          <input
            type="checkbox"
            checked={config.promptBakingEnabled}
            onChange={(e) => setConfig({ ...config, promptBakingEnabled: e.target.checked })}
            className="w-4 h-4"
          />
          <label className="text-sm text-gray-400">Enable Prompt Baking</label>
        </div>
      </div>

      {config.promptBakingEnabled && (
        <div className="space-y-4">
          <div>
            <label className="text-sm text-gray-400 block mb-2">Moral Compass Rules</label>
            <textarea
              rows={3}
              placeholder="Define moral and ethical guidelines for the model..."
              className="w-full bg-white/10 border border-white/20 rounded px-3 py-2 text-white placeholder-gray-500 text-sm"
              onChange={(e) => setConfig({
                ...config,
                moralCompassSettings: { ...config.moralCompassSettings, rules: e.target.value }
              })}
            />
          </div>

          <div>
            <label className="text-sm text-gray-400 block mb-2">Eudaimonia Settings</label>
            <textarea
              rows={3}
              placeholder="Define flourishing and well-being principles..."
              className="w-full bg-white/10 border border-white/20 rounded px-3 py-2 text-white placeholder-gray-500 text-sm"
              onChange={(e) => setConfig({
                ...config,
                eudaimoniaSettings: { ...config.eudaimoniaSettings, principles: e.target.value }
              })}
            />
          </div>

          <div className="flex items-center gap-2">
            <input
              type="checkbox"
              checked={config.personaGuidanceEnabled}
              onChange={(e) => setConfig({ ...config, personaGuidanceEnabled: e.target.checked })}
              className="w-4 h-4"
            />
            <label className="text-sm text-gray-400">Enable Persona Guidance System</label>
          </div>
        </div>
      )}
    </div>
  );

  const renderSelfModelingMetrics = () => (
    <div className="bg-white/5 backdrop-blur-lg rounded-xl p-4 border border-white/10">
      <h3 className="text-lg font-bold mb-3 flex items-center gap-2">
        <Gauge className="w-5 h-5 text-purple-400" />
        Self-Modeling Metrics
      </h3>

      <div className="grid grid-cols-2 gap-4 mb-4">
        <div className="bg-black/20 rounded-lg p-3">
          <div className="text-xs text-gray-400 mb-1">Prediction Accuracy</div>
          <div className="text-2xl font-bold text-blue-400">
            {(metrics.selfPredictionAccuracy * 100).toFixed(1)}%
          </div>
          <div className="text-xs text-gray-500">Self-awareness score</div>
        </div>

        <div className="bg-black/20 rounded-lg p-3">
          <div className="text-xs text-gray-400 mb-1">Temperature Range</div>
          <div className="text-2xl font-bold text-red-400">{metrics.temperatureRange.toFixed(2)}</div>
          <div className="text-xs text-gray-500">Current adaptive range</div>
        </div>

        <div className="bg-black/20 rounded-lg p-3">
          <div className="text-xs text-gray-400 mb-1">Grokking Detection</div>
          <div className={`text-2xl font-bold ${metrics.grockingSelfDetected ? 'text-green-400' : 'text-gray-400'}`}>
            {metrics.grockingSelfDetected ? 'ACTIVE' : 'INACTIVE'}
          </div>
          <div className="text-xs text-gray-500">Self-grokking state</div>
        </div>

        <div className="bg-black/20 rounded-lg p-3">
          <div className="text-xs text-gray-400 mb-1">Persona Development</div>
          <div className="text-2xl font-bold text-yellow-400">{metrics.personaDevelopmentScore.toFixed(1)}</div>
          <div className="text-xs text-gray-500">Character formation</div>
        </div>
      </div>

      <button
        onClick={() => setIsModeling(!isModeling)}
        className={`
          w-full px-4 py-3 rounded-lg font-semibold transition-all flex items-center justify-center gap-2
          ${isModeling
            ? 'bg-red-600 hover:bg-red-700 text-white'
            : 'bg-blue-600 hover:bg-blue-700 text-white'
          }
        `}
      >
        <Brain className="w-5 h-5" />
        {isModeling ? 'Stop Self-Modeling' : 'Start Self-Modeling'}
      </button>
    </div>
  );

  return (
    <div className="space-y-4">
      {renderTemperatureCurriculum()}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <div>
          {renderSelfPrediction()}
        </div>
        <div>
          {renderPromptBaking()}
        </div>
      </div>

      {renderSelfModelingMetrics()}
    </div>
  );
}