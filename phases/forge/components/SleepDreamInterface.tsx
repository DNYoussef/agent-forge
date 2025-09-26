'use client';

import { useState, useEffect } from 'react';
import { Moon, Brain, BarChart3, Clock, Zap, RefreshCw, Activity, Star } from 'lucide-react';

interface SleepDreamInterfaceProps {
  config: any;
  setConfig: (config: any) => void;
  metrics: any;
  setMetrics: (metrics: any) => void;
}

interface DreamCycle {
  id: number;
  phase: 'REM' | 'NREM1' | 'NREM2' | 'NREM3';
  duration: number;
  consolidationScore: number;
  patterns: string[];
  timestamp: Date;
}

export default function SleepDreamInterface({ config, setConfig, metrics, setMetrics }: SleepDreamInterfaceProps) {
  const [isDreaming, setIsDreaming] = useState(false);
  const [currentCycle, setCurrentCycle] = useState<DreamCycle | null>(null);
  const [recentDreams, setRecentDreams] = useState<DreamCycle[]>([]);
  const [sleepPhase, setSleepPhase] = useState<'awake' | 'light' | 'deep' | 'rem'>('awake');

  // Simulate dream cycles
  useEffect(() => {
    if (isDreaming) {
      const interval = setInterval(() => {
        // Update metrics
        setMetrics((prev: any) => ({
          ...prev,
          dreamCyclesCompleted: prev.dreamCyclesCompleted + (Math.random() > 0.7 ? 1 : 0),
          memoryConsolidationScore: Math.min(prev.memoryConsolidationScore + Math.random() * 5, 100),
          sleepEfficiency: Math.min(Math.max(prev.sleepEfficiency + (Math.random() - 0.4) * 10, 0), 100)
        }));

        // Cycle through sleep phases
        const phases: ('light' | 'deep' | 'rem')[] = ['light', 'deep', 'rem'];
        setSleepPhase(prev => {
          if (prev === 'awake') return 'light';
          const currentIndex = phases.indexOf(prev as any);
          return phases[(currentIndex + 1) % phases.length];
        });

        // Generate dream cycles occasionally
        if (Math.random() > 0.8) {
          const dreamPhases: ('REM' | 'NREM1' | 'NREM2' | 'NREM3')[] = ['REM', 'NREM1', 'NREM2', 'NREM3'];
          const patterns = [
            'gradient flow optimization',
            'attention pattern consolidation',
            'memory pathway strengthening',
            'synaptic weight adjustment',
            'neural pathway pruning',
            'activation pattern memories'
          ];

          const newCycle: DreamCycle = {
            id: Date.now(),
            phase: dreamPhases[Math.floor(Math.random() * dreamPhases.length)],
            duration: Math.floor(Math.random() * 120) + 30,
            consolidationScore: Math.random() * 100,
            patterns: patterns.slice(0, Math.floor(Math.random() * 3) + 1),
            timestamp: new Date()
          };

          setCurrentCycle(newCycle);
          setRecentDreams(prev => [newCycle, ...prev.slice(0, 4)]);
        }
      }, config.dreamCycleInterval);

      return () => clearInterval(interval);
    } else {
      setSleepPhase('awake');
      setCurrentCycle(null);
    }
  }, [isDreaming, config.dreamCycleInterval, setMetrics]);

  const renderSleepConfiguration = () => (
    <div className="bg-white/5 backdrop-blur-lg rounded-xl p-4 border border-white/10 mb-4">
      <h3 className="text-lg font-bold mb-3 flex items-center gap-2">
        <Moon className="w-5 h-5 text-blue-400" />
        Sleep/Dream Configuration
      </h3>

      <div className="grid grid-cols-2 gap-4">
        <div>
          <label className="text-sm text-gray-400 block mb-1">Dream Cycle Interval (ms)</label>
          <input
            type="number"
            min="500"
            max="5000"
            step="100"
            value={config.dreamCycleInterval}
            onChange={(e) => setConfig({ ...config, dreamCycleInterval: parseInt(e.target.value) })}
            className="w-full bg-white/10 border border-white/20 rounded px-3 py-2 text-white"
          />
        </div>

        <div>
          <label className="text-sm text-gray-400 block mb-1">Consolidation Depth</label>
          <div className="flex items-center gap-2">
            <input
              type="range"
              min="0.1"
              max="1.0"
              step="0.1"
              value={config.memoryConsolidationDepth}
              onChange={(e) => setConfig({ ...config, memoryConsolidationDepth: parseFloat(e.target.value) })}
              className="flex-1"
            />
            <span className="text-sm font-mono text-blue-400 min-w-[3rem]">
              {config.memoryConsolidationDepth.toFixed(1)}
            </span>
          </div>
        </div>

        <div>
          <label className="text-sm text-gray-400 block mb-1">Sleep Pattern Duration (s)</label>
          <input
            type="number"
            min="60"
            max="3600"
            step="30"
            value={config.sleepPatternDuration}
            onChange={(e) => setConfig({ ...config, sleepPatternDuration: parseInt(e.target.value) })}
            className="w-full bg-white/10 border border-white/20 rounded px-3 py-2 text-white"
          />
        </div>

        <div className="flex items-center">
          <div className={`text-lg font-bold ${
            sleepPhase === 'awake' ? 'text-yellow-400' :
            sleepPhase === 'light' ? 'text-blue-300' :
            sleepPhase === 'deep' ? 'text-purple-400' :
            'text-green-400'
          }`}>
            {sleepPhase.toUpperCase()} PHASE
          </div>
        </div>
      </div>
    </div>
  );

  const renderCurrentDream = () => {
    if (!currentCycle) {
      return (
        <div className="bg-white/5 backdrop-blur-lg rounded-xl p-4 border border-white/10 mb-4">
          <h3 className="text-lg font-bold mb-3 flex items-center gap-2">
            <Brain className="w-5 h-5 text-purple-400" />
            Current Dream State
          </h3>
          <div className="text-center py-8 text-gray-400">
            {isDreaming ? 'Entering sleep state...' : 'Model is awake - no active dreams'}
          </div>
        </div>
      );
    }

    return (
      <div className="bg-white/5 backdrop-blur-lg rounded-xl p-4 border border-white/10 mb-4">
        <h3 className="text-lg font-bold mb-3 flex items-center gap-2">
          <Brain className="w-5 h-5 text-purple-400" />
          Active Dream Cycle
        </h3>

        <div className="bg-black/20 rounded-lg p-4 mb-3">
          <div className="flex justify-between items-start mb-3">
            <div>
              <div className="text-lg font-bold text-purple-400">{currentCycle.phase} Sleep</div>
              <div className="text-sm text-gray-400">
                Duration: {currentCycle.duration}s | Score: {currentCycle.consolidationScore.toFixed(1)}
              </div>
            </div>
            <div className={`px-2 py-1 rounded text-xs font-bold ${
              currentCycle.phase === 'REM' ? 'bg-green-600' :
              currentCycle.phase.startsWith('NREM') ? 'bg-blue-600' :
              'bg-purple-600'
            }`}>
              {currentCycle.phase}
            </div>
          </div>

          <div>
            <div className="text-sm text-gray-400 mb-2">Active Consolidation Patterns:</div>
            <div className="space-y-1">
              {currentCycle.patterns.map((pattern, index) => (
                <div key={index} className="flex items-center gap-2 text-sm">
                  <Star className="w-3 h-3 text-yellow-400" />
                  <span className="text-gray-300">{pattern}</span>
                </div>
              ))}
            </div>
          </div>
        </div>

        <div className="grid grid-cols-3 gap-2">
          <div className="bg-black/20 rounded p-2 text-center">
            <div className="text-xs text-gray-400">Consolidation</div>
            <div className="text-sm font-bold text-green-400">
              {currentCycle.consolidationScore.toFixed(0)}%
            </div>
          </div>
          <div className="bg-black/20 rounded p-2 text-center">
            <div className="text-xs text-gray-400">Phase</div>
            <div className="text-sm font-bold text-blue-400">{currentCycle.phase}</div>
          </div>
          <div className="bg-black/20 rounded p-2 text-center">
            <div className="text-xs text-gray-400">Duration</div>
            <div className="text-sm font-bold text-purple-400">{currentCycle.duration}s</div>
          </div>
        </div>
      </div>
    );
  };

  const renderDreamHistory = () => (
    <div className="bg-white/5 backdrop-blur-lg rounded-xl p-4 border border-white/10 mb-4">
      <h3 className="text-lg font-bold mb-3 flex items-center gap-2">
        <Clock className="w-5 h-5 text-cyan-400" />
        Recent Dream Cycles
      </h3>

      {recentDreams.length === 0 ? (
        <div className="text-center py-4 text-gray-400">No recent dream cycles</div>
      ) : (
        <div className="space-y-2">
          {recentDreams.map((dream) => (
            <div key={dream.id} className="bg-black/20 rounded-lg p-3">
              <div className="flex justify-between items-start mb-2">
                <div className="flex items-center gap-2">
                  <div className={`w-2 h-2 rounded-full ${
                    dream.phase === 'REM' ? 'bg-green-400' :
                    dream.phase === 'NREM1' ? 'bg-blue-300' :
                    dream.phase === 'NREM2' ? 'bg-blue-400' :
                    'bg-purple-400'
                  }`} />
                  <span className="text-sm font-medium">{dream.phase}</span>
                  <span className="text-xs text-gray-400">
                    {dream.timestamp.toLocaleTimeString()}
                  </span>
                </div>
                <span className="text-xs font-mono text-cyan-400">
                  {dream.consolidationScore.toFixed(0)}%
                </span>
              </div>
              <div className="text-xs text-gray-300">
                {dream.patterns.join(', ')}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );

  const renderSleepMetrics = () => (
    <div className="bg-white/5 backdrop-blur-lg rounded-xl p-4 border border-white/10">
      <h3 className="text-lg font-bold mb-3 flex items-center gap-2">
        <BarChart3 className="w-5 h-5 text-green-400" />
        Sleep Quality Metrics
      </h3>

      <div className="grid grid-cols-2 gap-4 mb-4">
        <div className="bg-black/20 rounded-lg p-3">
          <div className="text-xs text-gray-400 mb-1">Dream Cycles Completed</div>
          <div className="text-2xl font-bold text-purple-400">{metrics.dreamCyclesCompleted}</div>
          <div className="text-xs text-gray-500">Total consolidation cycles</div>
        </div>

        <div className="bg-black/20 rounded-lg p-3">
          <div className="text-xs text-gray-400 mb-1">Memory Consolidation</div>
          <div className="text-2xl font-bold text-green-400">{metrics.memoryConsolidationScore.toFixed(1)}%</div>
          <div className="text-xs text-gray-500">Long-term retention</div>
        </div>

        <div className="bg-black/20 rounded-lg p-3">
          <div className="text-xs text-gray-400 mb-1">Sleep Efficiency</div>
          <div className="text-2xl font-bold text-blue-400">{metrics.sleepEfficiency.toFixed(1)}%</div>
          <div className="text-xs text-gray-500">Quality of rest cycles</div>
        </div>

        <div className="bg-black/20 rounded-lg p-3">
          <div className="text-xs text-gray-400 mb-1">Current Phase</div>
          <div className={`text-lg font-bold ${
            sleepPhase === 'awake' ? 'text-yellow-400' :
            sleepPhase === 'light' ? 'text-blue-300' :
            sleepPhase === 'deep' ? 'text-purple-400' :
            'text-green-400'
          }`}>
            {sleepPhase.toUpperCase()}
          </div>
          <div className="text-xs text-gray-500">Sleep state</div>
        </div>
      </div>

      <button
        onClick={() => setIsDreaming(!isDreaming)}
        className={`
          w-full px-4 py-3 rounded-lg font-semibold transition-all flex items-center justify-center gap-2
          ${isDreaming
            ? 'bg-orange-600 hover:bg-orange-700 text-white'
            : 'bg-purple-600 hover:bg-purple-700 text-white'
          }
        `}
      >
        {isDreaming ? (
          <>
            <RefreshCw className="w-5 h-5" />
            Wake Up Model
          </>
        ) : (
          <>
            <Moon className="w-5 h-5" />
            Enter Dream State
          </>
        )}
      </button>
    </div>
  );

  return (
    <div className="space-y-4">
      {renderSleepConfiguration()}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <div>
          {renderCurrentDream()}
        </div>
        <div>
          {renderDreamHistory()}
        </div>
      </div>

      {renderSleepMetrics()}
    </div>
  );
}