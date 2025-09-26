'use client';

import { useState, useEffect } from 'react';
import { ArrowLeft, Flame, TrendingUp, Settings, Activity, Zap, Eye, Brain, Moon, Target } from 'lucide-react';
import Link from 'next/link';
import PhaseController from '@/components/shared/PhaseController';

// Import sophisticated Phase 5 components
import AssessmentInterface from './components/AssessmentInterface';
import TrainingLoopInterface from './components/TrainingLoopInterface';
import SelfModelingInterface from './components/SelfModelingInterface';
import SleepDreamInterface from './components/SleepDreamInterface';
import WeightSpaceSphere from './components/WeightSpaceSphere'; // Changed to use the working sphere

interface Phase5Config {
  // Master Phase 5 Configuration
  currentSubPhase: 'assessment' | 'training' | 'self-modeling' | 'sleep-dream';
  currentLevel: number; // 1-10 level progression
  totalLevels: number;

  // Assessment Phase Config
  openRouterApiKey: string;
  frontierModels: string[];
  targetSuccessRate: number;
  edgeOfChaosThreshold: number;

  // Training Loop Config
  questionsPerLevel: number;
  maxHintsPerQuestion: number;
  strikeSystemEnabled: boolean;
  questionVariations: number;

  // Self-Modeling Config
  temperatureRanges: number[];
  selfPredictionAreaWidth: number;
  promptBakingEnabled: boolean;
  personaGuidanceEnabled: boolean;
  moralCompassSettings: any;
  eudaimoniaSettings: any;

  // Sleep/Dream Config
  dreamCycleInterval: number;
  memoryConsolidationDepth: number;
  sleepPatternDuration: number;

  // Weight Space Proprioception
  proprietyceptionEnabled: boolean;
  geometricRepresentation: '2D' | '3D';
  weightVisualizationMode: 'real-time' | 'checkpoint';
  grokVisualizationEnabled: boolean;

  // GrokFast Integration
  grokfastEnabled: boolean;
  grokfastAlpha: number;
  grokfastLambda: number;

  // For sphere compatibility
  sleepMode?: boolean;
  metrics?: {
    grokfast_active: boolean;
    grokfast_lambda: number;
  };
}

interface Phase5Metrics {
  // Master Progress
  overallProgress: number;
  currentLevelProgress: number;
  subPhaseProgress: number;

  // Assessment Metrics
  currentSuccessRate: number;
  edgeOfChaosDetected: boolean;
  difficultyLevel: number;
  questionsGenerated: number;

  // Training Loop Metrics
  questionsAttempted: number;
  questionsCompleted: number;
  hintsAccumulated: number;
  questionVariationsCreated: number;
  strikeCount: number;

  // Self-Modeling Metrics
  selfPredictionAccuracy: number;
  temperatureRange: number;
  grockingSelfDetected: boolean;
  personaDevelopmentScore: number;

  // Sleep/Dream Metrics
  dreamCyclesCompleted: number;
  memoryConsolidationScore: number;
  sleepEfficiency: number;

  // Weight Space Metrics
  weightSpaceGeometry: any;
  grockingTransitions: number;
  proprietyceptionFeed: boolean;
}

export default function SophisticatedPhase5Page() {
  const [hasStarted, setHasStarted] = useState(false);
  const [config, setConfig] = useState<Phase5Config>({
    // Master Configuration
    currentSubPhase: 'assessment',
    currentLevel: 1,
    totalLevels: 10,

    // Assessment Phase
    openRouterApiKey: '',
    frontierModels: ['gpt-4', 'claude-3-opus', 'gemini-pro'],
    targetSuccessRate: 0.75,
    edgeOfChaosThreshold: 0.75,

    // Training Loop
    questionsPerLevel: 100,
    maxHintsPerQuestion: 5,
    strikeSystemEnabled: true,
    questionVariations: 3,

    // Self-Modeling
    temperatureRanges: [0.1, 0.5, 1.0, 1.5, 2.0],
    selfPredictionAreaWidth: 50,
    promptBakingEnabled: true,
    personaGuidanceEnabled: true,
    moralCompassSettings: {},
    eudaimoniaSettings: {},

    // Sleep/Dream
    dreamCycleInterval: 1000,
    memoryConsolidationDepth: 0.8,
    sleepPatternDuration: 300,

    // Weight Space
    proprietyceptionEnabled: true,
    geometricRepresentation: '3D',
    weightVisualizationMode: 'real-time',
    grokVisualizationEnabled: true,

    // GrokFast
    grokfastEnabled: true,
    grokfastAlpha: 0.98,
    grokfastLambda: 0.05,

    // Sphere compatibility
    sleepMode: false,
    metrics: {
      grokfast_active: true,
      grokfast_lambda: 0.05
    }
  });

  const [metrics, setMetrics] = useState<Phase5Metrics>({
    // Master Progress
    overallProgress: 0,
    currentLevelProgress: 0,
    subPhaseProgress: 0,

    // Assessment
    currentSuccessRate: 0,
    edgeOfChaosDetected: false,
    difficultyLevel: 1,
    questionsGenerated: 0,

    // Training Loop
    questionsAttempted: 0,
    questionsCompleted: 0,
    hintsAccumulated: 0,
    questionVariationsCreated: 0,
    strikeCount: 0,

    // Self-Modeling
    selfPredictionAccuracy: 0,
    temperatureRange: 1.0,
    grockingSelfDetected: false,
    personaDevelopmentScore: 0,

    // Sleep/Dream
    dreamCyclesCompleted: 0,
    memoryConsolidationScore: 0,
    sleepEfficiency: 0,

    // Weight Space
    weightSpaceGeometry: null,
    grockingTransitions: 0,
    proprietyceptionFeed: false,
  });

  const [isTraining, setIsTraining] = useState(false);

  // Level descriptions for guidance
  const levelDescriptions = [
    { level: 1, title: "Foundation Setup", description: "Initialize OpenRouter, configure frontier models, set edge-of-chaos parameters" },
    { level: 2, title: "Assessment Phase", description: "Test model capabilities, detect optimal difficulty, calibrate success rate to 75%" },
    { level: 3, title: "Question Generation", description: "Generate adaptive questions using frontier models, implement 3-strike hint system" },
    { level: 4, title: "Training Loop", description: "Execute systematic training with question variations and progressive difficulty" },
    { level: 5, title: "Self-Modeling", description: "Enable temperature curriculum, self-prediction, and gap analysis" },
    { level: 6, title: "Prompt Baking", description: "Implement moral compass rules, eudaimonia settings, persona development" },
    { level: 7, title: "Weight Proprioception", description: "Activate geometric weight space representation and grokking visualization" },
    { level: 8, title: "Dream Cycles", description: "Initialize sleep/dream memory consolidation with configurable intervals" },
    { level: 9, title: "GrokFast Integration", description: "Enable 50x acceleration with alpha/lambda optimization" },
    { level: 10, title: "Full Orchestration", description: "Complete sophisticated training loop with all systems operational" }
  ];

  // Sub-phase navigation
  const subPhases = [
    { id: 'assessment', name: 'Assessment', icon: Target, description: 'Edge-of-Chaos Detection with Frontier Models' },
    { id: 'training', name: 'Training Loop', icon: Zap, description: '10-Level Adaptive Question System' },
    { id: 'self-modeling', name: 'Self-Modeling', icon: Brain, description: 'Temperature-Aware Self-Prediction' },
    { id: 'sleep-dream', name: 'Sleep/Dream', icon: Moon, description: 'Memory Consolidation Cycles' }
  ];

  // Auto-progress logic
  useEffect(() => {
    if (isTraining && hasStarted) {
      const interval = setInterval(() => {
        setMetrics((prev) => {
          let newProgress = prev.currentLevelProgress + Math.random() * 3;
          let newLevel = config.currentLevel;
          let newOverallProgress = prev.overallProgress;

          // Level progression logic
          if (newProgress >= 100) {
            if (config.currentLevel < config.totalLevels) {
              newLevel = config.currentLevel + 1;
              newProgress = 0;
              setConfig(c => ({ ...c, currentLevel: newLevel }));
            } else {
              newProgress = 100;
            }
          }

          // Overall progress calculation
          newOverallProgress = ((newLevel - 1) * 100 + newProgress) / config.totalLevels;

          return {
            ...prev,
            currentLevelProgress: newProgress,
            overallProgress: newOverallProgress
          };
        });
      }, 1000);

      return () => clearInterval(interval);
    }
  }, [isTraining, hasStarted, config.currentLevel, config.totalLevels]);

  // Update sleep mode based on current sub-phase
  useEffect(() => {
    setConfig(prev => ({
      ...prev,
      sleepMode: prev.currentSubPhase === 'sleep-dream'
    }));
  }, [config.currentSubPhase]);

  // Render Phase 5 Homepage
  const renderHomepage = () => {
    const currentLevelInfo = levelDescriptions.find(l => l.level === config.currentLevel);

    return (
      <div className="max-w-6xl mx-auto space-y-6">
        {/* Hero Section */}
        <div className="text-center bg-white/5 backdrop-blur-lg rounded-3xl p-12 border border-white/10">
          <div className="mb-8">
            <div className="text-8xl font-bold bg-gradient-to-r from-orange-400 via-red-400 to-purple-400 bg-clip-text text-transparent mb-4">
              LEVEL {config.currentLevel}
            </div>
            <h2 className="text-4xl font-bold text-white mb-4">{currentLevelInfo?.title}</h2>
            <p className="text-xl text-gray-300 mb-8 max-w-3xl mx-auto">
              {currentLevelInfo?.description}
            </p>
          </div>

          {/* Giant Start Button */}
          <button
            onClick={() => {
              setHasStarted(true);
              setIsTraining(true);
            }}
            className="bg-gradient-to-r from-orange-500 to-red-500 hover:from-orange-600 hover:to-red-600
                       text-white text-2xl font-bold py-6 px-12 rounded-2xl shadow-2xl
                       transform hover:scale-105 transition-all duration-300
                       flex items-center justify-center gap-4 mx-auto"
          >
            <Flame className="w-8 h-8" />
            START PHASE 5 TRAINING
            <Zap className="w-8 h-8" />
          </button>
        </div>

        {/* 10-Level Overview */}
        <div className="bg-white/5 backdrop-blur-lg rounded-2xl p-6 border border-white/10">
          <h3 className="text-2xl font-bold text-orange-400 mb-6 text-center">
            10-Level Sophisticated Training System
          </h3>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-4">
            {levelDescriptions.map((level) => {
              const isCurrent = level.level === config.currentLevel;
              const isCompleted = level.level < config.currentLevel;

              return (
                <div
                  key={level.level}
                  className={`
                    p-4 rounded-xl border-2 transition-all
                    ${isCurrent ? 'border-orange-500 bg-orange-500/10 scale-105' :
                      isCompleted ? 'border-green-500 bg-green-500/10' :
                      'border-gray-600 bg-white/5'
                    }
                  `}
                >
                  <div className={`
                    text-3xl font-bold mb-2 text-center
                    ${isCurrent ? 'text-orange-400 animate-pulse' :
                      isCompleted ? 'text-green-400' :
                      'text-gray-500'
                    }
                  `}>
                    {level.level}
                  </div>
                  <h4 className={`
                    text-sm font-semibold mb-2 text-center
                    ${isCurrent ? 'text-orange-300' :
                      isCompleted ? 'text-green-300' :
                      'text-gray-400'
                    }
                  `}>
                    {level.title}
                  </h4>
                  <p className={`
                    text-xs leading-tight
                    ${isCurrent ? 'text-gray-200' :
                      isCompleted ? 'text-gray-300' :
                      'text-gray-500'
                    }
                  `}>
                    {level.description}
                  </p>

                  {isCurrent && (
                    <div className="mt-3">
                      <div className="w-full bg-gray-700 rounded-full h-2">
                        <div className="bg-orange-400 h-2 rounded-full w-0 animate-pulse" />
                      </div>
                    </div>
                  )}

                  {isCompleted && (
                    <div className="mt-2 text-center">
                      <div className="w-6 h-6 bg-green-500 rounded-full mx-auto flex items-center justify-center">
                        <svg className="w-4 h-4 text-white" fill="currentColor" viewBox="0 0 20 20">
                          <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                        </svg>
                      </div>
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </div>

        {/* System Features Overview */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          <div className="bg-white/5 backdrop-blur-lg rounded-xl p-6 border border-white/10 text-center">
            <Target className="w-12 h-12 text-blue-400 mx-auto mb-4" />
            <h4 className="text-lg font-semibold text-blue-400 mb-2">OpenRouter Integration</h4>
            <p className="text-sm text-gray-400">Frontier models (GPT-4, Claude, Gemini) generate adaptive questions at edge-of-chaos</p>
          </div>

          <div className="bg-white/5 backdrop-blur-lg rounded-xl p-6 border border-white/10 text-center">
            <Brain className="w-12 h-12 text-purple-400 mx-auto mb-4" />
            <h4 className="text-lg font-semibold text-purple-400 mb-2">Self-Modeling</h4>
            <p className="text-sm text-gray-400">Temperature-aware self-prediction with gap analysis and persona development</p>
          </div>

          <div className="bg-white/5 backdrop-blur-lg rounded-xl p-6 border border-white/10 text-center">
            <Eye className="w-12 h-12 text-cyan-400 mx-auto mb-4" />
            <h4 className="text-lg font-semibold text-cyan-400 mb-2">Weight Proprioception</h4>
            <p className="text-sm text-gray-400">3D geometric mathematical representation of weight space with grokking visualization</p>
          </div>

          <div className="bg-white/5 backdrop-blur-lg rounded-xl p-6 border border-white/10 text-center">
            <Moon className="w-12 h-12 text-indigo-400 mx-auto mb-4" />
            <h4 className="text-lg font-semibold text-indigo-400 mb-2">Dream Cycles</h4>
            <p className="text-sm text-gray-400">Memory consolidation through configurable sleep/dream cycles</p>
          </div>
        </div>

        {/* Quick Config Preview */}
        <div className="bg-white/5 backdrop-blur-lg rounded-xl p-6 border border-white/10">
          <h3 className="text-xl font-bold text-orange-400 mb-4">Quick Configuration Preview</h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
            <div>
              <span className="text-gray-400">Target Success Rate:</span>
              <span className="text-orange-400 ml-2 font-mono">{(config.targetSuccessRate * 100).toFixed(0)}%</span>
            </div>
            <div>
              <span className="text-gray-400">GrokFast Acceleration:</span>
              <span className="text-green-400 ml-2 font-mono">{config.grokfastEnabled ? '50x ENABLED' : 'DISABLED'}</span>
            </div>
            <div>
              <span className="text-gray-400">Weight Visualization:</span>
              <span className="text-blue-400 ml-2 font-mono">{config.geometricRepresentation} MODE</span>
            </div>
          </div>
        </div>
      </div>
    );
  };

  // Level progression indicator
  const renderLevelProgression = () => {
    return (
      <div className="bg-white/5 backdrop-blur-lg rounded-xl p-4 border border-white/10">
        <div className="flex justify-between items-center mb-2">
          <span className="text-sm text-gray-400">Phase 5 Master Progress</span>
          <span className="text-orange-400 font-semibold">Level {config.currentLevel}/{config.totalLevels}</span>
        </div>

        {/* Overall Progress Bar */}
        <div className="w-full bg-gray-700 rounded-full h-3 mb-4">
          <div
            className="bg-gradient-to-r from-orange-400 to-red-400 h-3 rounded-full transition-all duration-500"
            style={{ width: `${metrics.overallProgress}%` }}
          />
        </div>

        {/* Level Grid */}
        <div className="grid grid-cols-10 gap-1">
          {Array.from({ length: config.totalLevels }, (_, i) => {
            const level = i + 1;
            const isCompleted = level < config.currentLevel;
            const isCurrent = level === config.currentLevel;

            return (
              <div
                key={level}
                className={`
                  h-8 rounded text-xs flex items-center justify-center font-semibold transition-all
                  ${isCompleted ? 'bg-green-600 text-white' :
                    isCurrent ? 'bg-orange-500 text-white animate-pulse' :
                    'bg-gray-700 text-gray-400'}
                `}
              >
                {level}
              </div>
            );
          })}
        </div>

        {/* Current Level Progress */}
        <div className="mt-4">
          <div className="flex justify-between text-xs text-gray-400 mb-1">
            <span>Level {config.currentLevel} Progress</span>
            <span>{metrics.currentLevelProgress.toFixed(1)}%</span>
          </div>
          <div className="w-full bg-gray-700 rounded-full h-2">
            <div
              className="bg-orange-400 h-2 rounded-full transition-all duration-300"
              style={{ width: `${metrics.currentLevelProgress}%` }}
            />
          </div>
        </div>
      </div>
    );
  };

  // Sub-phase tabs
  const renderSubPhaseTabs = () => {
    return (
      <div className="flex space-x-2 mb-6">
        {subPhases.map((phase) => {
          const Icon = phase.icon;
          const isActive = config.currentSubPhase === phase.id;

          return (
            <button
              key={phase.id}
              onClick={() => setConfig({ ...config, currentSubPhase: phase.id as any })}
              className={`
                flex items-center gap-2 px-4 py-3 rounded-lg font-medium transition-all
                ${isActive
                  ? 'bg-orange-500 text-white shadow-lg'
                  : 'bg-white/10 text-gray-300 hover:bg-white/20'
                }
              `}
            >
              <Icon className="w-4 h-4" />
              <span>{phase.name}</span>
            </button>
          );
        })}
      </div>
    );
  };

  // Master control panel
  const renderMasterControls = () => {
    return (
      <div className="bg-white/5 backdrop-blur-lg rounded-xl p-4 border border-white/10">
        <h3 className="text-lg font-bold mb-3 flex items-center gap-2">
          <Settings className="w-5 h-5 text-orange-400" />
          Phase 5 Master Controls
        </h3>

        <div className="grid grid-cols-2 gap-4">
          {/* GrokFast Controls */}
          <div className="border border-white/10 rounded-lg p-3">
            <h4 className="text-sm font-semibold text-orange-400 mb-2">GrokFast Acceleration</h4>
            <div className="flex items-center gap-2 mb-2">
              <input
                type="checkbox"
                checked={config.grokfastEnabled}
                onChange={(e) => setConfig({...config, grokfastEnabled: e.target.checked})}
                className="w-4 h-4"
              />
              <label className="text-xs text-gray-400">50x Acceleration</label>
            </div>
            {config.grokfastEnabled && (
              <div className="space-y-2">
                <div>
                  <label className="text-xs text-gray-500 block">Alpha: {config.grokfastAlpha}</label>
                  <input
                    type="range"
                    min="0.9"
                    max="0.99"
                    step="0.01"
                    value={config.grokfastAlpha}
                    onChange={(e) => setConfig({...config, grokfastAlpha: parseFloat(e.target.value)})}
                    className="w-full"
                  />
                </div>
                <div>
                  <label className="text-xs text-gray-500 block">Lambda: {config.grokfastLambda}</label>
                  <input
                    type="range"
                    min="0.01"
                    max="0.25"
                    step="0.01"
                    value={config.grokfastLambda}
                    onChange={(e) => setConfig({...config, grokfastLambda: parseFloat(e.target.value)})}
                    className="w-full"
                  />
                </div>
              </div>
            )}
          </div>

          {/* Weight Space Proprioception */}
          <div className="border border-white/10 rounded-lg p-3">
            <h4 className="text-sm font-semibold text-blue-400 mb-2">Weight Space Proprioception</h4>
            <div className="flex items-center gap-2 mb-2">
              <input
                type="checkbox"
                checked={config.proprietyceptionEnabled}
                onChange={(e) => setConfig({...config, proprietyceptionEnabled: e.target.checked})}
                className="w-4 h-4"
              />
              <label className="text-xs text-gray-400">Geometric Representation</label>
            </div>
            {config.proprietyceptionEnabled && (
              <div className="space-y-2">
                <div>
                  <label className="text-xs text-gray-500 block">Visualization Mode</label>
                  <select
                    value={config.geometricRepresentation}
                    onChange={(e) => setConfig({...config, geometricRepresentation: e.target.value as '2D' | '3D'})}
                    className="w-full bg-white/10 border border-white/20 rounded px-2 py-1 text-xs"
                  >
                    <option value="2D">2D Geometric</option>
                    <option value="3D">3D Geometric</option>
                  </select>
                </div>
                <div className="flex items-center gap-2">
                  <input
                    type="checkbox"
                    checked={config.grokVisualizationEnabled}
                    onChange={(e) => setConfig({...config, grokVisualizationEnabled: e.target.checked})}
                    className="w-3 h-3"
                  />
                  <label className="text-xs text-gray-500">Grok Transitions</label>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Phase 5 Master Start/Stop */}
        <div className="mt-4 pt-4 border-t border-white/10">
          <button
            onClick={() => setIsTraining(!isTraining)}
            className={`
              w-full px-4 py-3 rounded-lg font-semibold transition-all flex items-center justify-center gap-2
              ${isTraining
                ? 'bg-red-600 hover:bg-red-700 text-white'
                : 'bg-orange-500 hover:bg-orange-600 text-white'
              }
            `}
          >
            <Flame className="w-5 h-5" />
            {isTraining ? 'Stop Phase 5 Training' : 'Start Sophisticated Phase 5'}
          </button>
        </div>
      </div>
    );
  };

  // Render current sub-phase interface with sphere at bottom
  const renderCurrentSubPhase = () => {
    const renderInterface = () => {
      switch (config.currentSubPhase) {
        case 'assessment':
          return <AssessmentInterface config={config} setConfig={setConfig} metrics={metrics} setMetrics={setMetrics} />;
        case 'training':
          return <TrainingLoopInterface config={config} setConfig={setConfig} metrics={metrics} setMetrics={setMetrics} />;
        case 'self-modeling':
          return <SelfModelingInterface config={config} setConfig={setConfig} metrics={metrics} setMetrics={setMetrics} />;
        case 'sleep-dream':
          return <SleepDreamInterface config={config} setConfig={setConfig} metrics={metrics} setMetrics={setMetrics} />;
        default:
          return <div className="text-center text-gray-400">Select a sub-phase to begin</div>;
      }
    };

    return (
      <div className="space-y-6">
        {/* Sub-phase Interface */}
        {renderInterface()}

        {/* BitLinear Weight Space Sphere - Always visible at bottom of each tab */}
        <div className="mt-8">
          <h3 className="text-lg font-bold text-cyan-400 mb-4 flex items-center gap-2">
            <Eye className="w-5 h-5" />
            BitLinear Weight Space Visualization (-1, 0, 1)
          </h3>
          <WeightSpaceSphere config={config} currentLevel={config.currentLevel} />
        </div>
      </div>
    );
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-orange-950 to-slate-950 text-white p-8">
      <Link href="/" className="flex items-center gap-2 text-orange-400 hover:text-orange-300 mb-8">
        <ArrowLeft className="w-5 h-5" />
        Back to Dashboard
      </Link>

      <div className="mb-8">
        <h1 className="text-5xl font-bold mb-4 bg-gradient-to-r from-orange-400 to-red-400 bg-clip-text text-transparent flex items-center gap-4">
          <Flame className="w-12 h-12 text-orange-400" />
          Phase 5: Sophisticated Training Loop
        </h1>
        <p className="text-xl text-gray-400">
          Multi-phase training with OpenRouter integration, BitLinear weight space proprioception, and 10-level progression
        </p>
      </div>

      {/* Conditional Rendering: Homepage vs Training Interface */}
      {!hasStarted ? (
        renderHomepage()
      ) : (
        <div className="grid grid-cols-1 xl:grid-cols-4 gap-6">
          {/* Left Column: Master Controls & Progress */}
          <div className="xl:col-span-1 space-y-4">
            {/* Back to Homepage Button */}
            <div className="bg-white/5 backdrop-blur-lg rounded-xl p-4 border border-white/10">
              <button
                onClick={() => {
                  setHasStarted(false);
                  setIsTraining(false);
                }}
                className="w-full px-3 py-2 bg-gray-600 hover:bg-gray-700 text-white rounded-lg text-sm transition-all"
              >
                ← Back to Homepage
              </button>
            </div>

            {renderLevelProgression()}
            {renderMasterControls()}
          </div>

          {/* Right Column: Sub-Phase Interfaces with Sphere */}
          <div className="xl:col-span-3">
            {renderSubPhaseTabs()}

            <div className="bg-white/5 backdrop-blur-lg rounded-2xl p-6 border border-white/10">
              <div className="mb-4">
                <h2 className="text-2xl font-bold text-orange-400">
                  {subPhases.find(p => p.id === config.currentSubPhase)?.name}
                </h2>
                <p className="text-gray-400">
                  {subPhases.find(p => p.id === config.currentSubPhase)?.description}
                </p>
              </div>

              {renderCurrentSubPhase()}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}