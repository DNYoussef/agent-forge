'use client';

import React, { useState, useEffect, useRef } from 'react';
import { Canvas } from '@react-three/fiber';
import { SVDWeightSpaceVisualization } from './components/SVDWeightSpaceVisualization';
import { ExpertConfigurationDiscovery } from './components/ExpertConfigurationDiscovery';
import { TransformersSquaredVisualizer } from './components/TransformersSquaredVisualizer';
import { Phase2IntegrationPanel } from './components/Phase2IntegrationPanel';
import { AdasMetaAgentMonitor } from './components/AdasMetaAgentMonitor';
import { SelfGuidedCompositionVisualizer } from './components/SelfGuidedCompositionVisualizer';
import { WebSocketProvider } from './components/WebSocketProvider';
import { AgenticDataProvider } from './contexts/AgenticDataContext';
import {
  Brain,
  Network,
  Zap,
  Settings,
  Activity,
  Search,
  Layers,
  GitBranch,
  Cpu,
  Target
} from 'lucide-react';

interface AgenticDashboardState {
  activeView: 'overview' | 'discovery' | 'transformers2' | 'svd' | 'phase2' | 'adas' | 'self_guided';
  isConnected: boolean;
  realTimeData: any;
  discoverySession: any;
  activeConfiguration: any;
  selfGuidedCompositions: any;
  systemStatus: {
    adasActive: boolean;
    t2Active: boolean;
    phase2Connected: boolean;
    selfGuidedActive: boolean;
    configurationsDiscovered: number;
    tasksProcessed: number;
    modelSelfExaminations: number;
  };
}

const AgenticDashboard: React.FC = () => {
  const [state, setState] = useState<AgenticDashboardState>({
    activeView: 'overview',
    isConnected: false,
    realTimeData: null,
    discoverySession: null,
    activeConfiguration: null,
    selfGuidedCompositions: null,
    systemStatus: {
      adasActive: false,
      t2Active: false,
      phase2Connected: false,
      selfGuidedActive: false,
      configurationsDiscovered: 0,
      tasksProcessed: 0,
      modelSelfExaminations: 0
    }
  });

  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    // Initialize WebSocket connection to backend
    const initializeWebSocket = () => {
      const ws = new WebSocket('ws://localhost:8080/agentic');
      wsRef.current = ws;

      ws.onopen = () => {
        setState(prev => ({ ...prev, isConnected: true }));
        console.log('Agentic dashboard connected to backend');
        // Request initial status
        ws.send(JSON.stringify({ type: 'GET_STATUS' }));
      };

      ws.onmessage = (event) => {
        const data = JSON.parse(event.data);

        switch (data.type) {
          case 'DISCOVERY_UPDATE':
            setState(prev => ({
              ...prev,
              discoverySession: data.session,
              systemStatus: {
                ...prev.systemStatus,
                configurationsDiscovered: data.configurationsFound
              }
            }));
            break;

          case 'SELF_GUIDED_UPDATE':
            setState(prev => ({
              ...prev,
              selfGuidedCompositions: data.compositions,
              systemStatus: {
                ...prev.systemStatus,
                selfGuidedActive: true,
                modelSelfExaminations: data.examinations || 0
              }
            }));
            break;

          case 'CONFIGURATION_APPLIED':
            setState(prev => ({
              ...prev,
              activeConfiguration: data.configuration
            }));
            break;

          case 'STATUS_UPDATE':
            setState(prev => ({
              ...prev,
              systemStatus: data.status
            }));
            break;

          default:
            setState(prev => ({ ...prev, realTimeData: data }));
        }
      };

      ws.onclose = () => {
        setState(prev => ({ ...prev, isConnected: false }));
        setTimeout(initializeWebSocket, 3000); // Reconnect after 3s
      };
    };

    initializeWebSocket();

    return () => {
      wsRef.current?.close();
    };
  }, []);

  const startDiscovery = (taskDescription: string) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({
        type: 'START_DISCOVERY',
        task: taskDescription
      }));
    }
  };

  const applyConfiguration = (configId: string) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({
        type: 'APPLY_CONFIGURATION',
        configurationId: configId
      }));
    }
  };

  const renderMainContent = () => {
    switch (state.activeView) {
      case 'discovery':
        return (
          <ExpertConfigurationDiscovery
            session={state.discoverySession}
            onStartDiscovery={startDiscovery}
            onApplyConfiguration={applyConfiguration}
            activeConfiguration={state.activeConfiguration}
          />
        );

      case 'transformers2':
        return (
          <TransformersSquaredVisualizer
            data={state.realTimeData?.transformers2}
            activeConfiguration={state.activeConfiguration}
          />
        );

      case 'svd':
        return (
          <div className="h-full">
            <Canvas camera={{ position: [0, 0, 5], fov: 75 }}>
              <SVDWeightSpaceVisualization
                data={state.realTimeData?.svd}
                expertVectors={state.activeConfiguration?.expertVectors}
              />
            </Canvas>
          </div>
        );

      case 'phase2':
        return (
          <Phase2IntegrationPanel
            weightData={state.realTimeData?.phase2}
            insights={state.discoverySession?.task_analysis?.weight_insights}
            connected={state.systemStatus.phase2Connected}
          />
        );

      case 'adas':
        return (
          <AdasMetaAgentMonitor
            searchData={state.realTimeData?.adas}
            discoverySession={state.discoverySession}
            configurationsArchive={state.realTimeData?.archive}
          />
        );

      case 'self_guided':
        return (
          <SelfGuidedCompositionVisualizer
            compositions={state.selfGuidedCompositions}
            discoverySession={state.discoverySession}
            modelExaminations={state.systemStatus.modelSelfExaminations}
          />
        );

      default:
        return (
          <div className="space-y-6">
            {/* System Title */}
            <div className="bg-gradient-to-r from-purple-900 to-blue-900 rounded-lg p-6 border border-purple-500">
              <h2 className="text-3xl font-bold mb-2">
                Self-Guided Automatic Discovery of Agentic Expert Vector Configurations
              </h2>
              <p className="text-gray-300">
                ADAS meta-agent search with Transformers² self-guided expert composition - Model directs its own configuration
              </p>
              <div className="mt-4 grid grid-cols-4 gap-4">
                <div className="bg-black/30 rounded-lg p-3">
                  <div className="text-sm text-gray-400">Innovation</div>
                  <div className="text-xl font-bold">Self-Guided</div>
                </div>
                <div className="bg-black/30 rounded-lg p-3">
                  <div className="text-sm text-gray-400">Configurations Found</div>
                  <div className="text-xl font-bold">{state.systemStatus.configurationsDiscovered}</div>
                </div>
                <div className="bg-black/30 rounded-lg p-3">
                  <div className="text-sm text-gray-400">Model Self-Examinations</div>
                  <div className="text-xl font-bold">{state.systemStatus.modelSelfExaminations}</div>
                </div>
                <div className="bg-black/30 rounded-lg p-3">
                  <div className="text-sm text-gray-400">Tasks Processed</div>
                  <div className="text-xl font-bold">{state.systemStatus.tasksProcessed}</div>
                </div>
              </div>
            </div>

            {/* Overview Grid */}
            <div className="grid grid-cols-2 gap-6">
              {/* Left Column */}
              <div className="space-y-4">
                {/* Discovery Status */}
                <div className="bg-gray-900 border border-gray-700 rounded-lg p-4">
                  <h3 className="text-lg font-semibold mb-3 flex items-center gap-2">
                    <Search className="w-5 h-5 text-purple-400" />
                    Expert Configuration Discovery
                  </h3>
                  {state.discoverySession ? (
                    <div className="space-y-2">
                      <div className="text-sm">
                        <span className="text-gray-400">Current Task:</span>
                        <span className="ml-2">{state.discoverySession.task_description}</span>
                      </div>
                      <div className="text-sm">
                        <span className="text-gray-400">Configurations:</span>
                        <span className="ml-2">{state.discoverySession.validated_configurations?.length || 0}</span>
                      </div>
                      <div className="text-sm">
                        <span className="text-gray-400">Best Score:</span>
                        <span className="ml-2">
                          {state.discoverySession.discovery_results?.best_performance?.toFixed(3) || 'N/A'}
                        </span>
                      </div>
                    </div>
                  ) : (
                    <div className="text-gray-500 text-sm">No active discovery session</div>
                  )}
                </div>

                {/* SVD Visualization Mini */}
                <div className="bg-gray-900 border border-gray-700 rounded-lg p-4 h-80">
                  <h3 className="text-lg font-semibold mb-3 flex items-center gap-2">
                    <Layers className="w-5 h-5 text-blue-400" />
                    SVD Weight Introspection
                  </h3>
                  <Canvas camera={{ position: [0, 0, 5] }}>
                    <SVDWeightSpaceVisualization
                      data={state.realTimeData?.svd}
                      compact
                    />
                  </Canvas>
                </div>
              </div>

              {/* Right Column */}
              <div className="space-y-4">
                {/* System Status */}
                <div className="bg-gray-900 border border-gray-700 rounded-lg p-4">
                  <h3 className="text-lg font-semibold mb-3 flex items-center gap-2">
                    <Cpu className="w-5 h-5 text-green-400" />
                    System Components
                  </h3>
                  <div className="grid grid-cols-2 gap-3">
                    <StatusIndicator
                      label="ADAS Meta-Agent"
                      active={state.systemStatus.adasActive}
                      icon={<Brain className="w-4 h-4" />}
                    />
                    <StatusIndicator
                      label="Transformers²"
                      active={state.systemStatus.t2Active}
                      icon={<GitBranch className="w-4 h-4" />}
                    />
                    <StatusIndicator
                      label="Phase 2 Integration"
                      active={state.systemStatus.phase2Connected}
                      icon={<Network className="w-4 h-4" />}
                    />
                    <StatusIndicator
                      label="Self-Guided Mode"
                      active={state.systemStatus.selfGuidedActive}
                      icon={<Target className="w-4 h-4" />}
                    />
                  </div>
                </div>

                {/* Active Configuration */}
                <div className="bg-gray-900 border border-gray-700 rounded-lg p-4">
                  <h3 className="text-lg font-semibold mb-3 flex items-center gap-2">
                    <Settings className="w-5 h-5 text-yellow-400" />
                    Active Configuration
                  </h3>
                  {state.activeConfiguration ? (
                    <div className="space-y-2">
                      <div className="text-sm">
                        <span className="text-gray-400">Task:</span>
                        <span className="ml-2">{state.activeConfiguration.task_description}</span>
                      </div>
                      <div className="text-sm">
                        <span className="text-gray-400">Config ID:</span>
                        <span className="ml-2 font-mono text-xs">
                          {state.activeConfiguration.config_data?.original_config?.config_id}
                        </span>
                      </div>
                      <div className="text-sm">
                        <span className="text-gray-400">Compatibility:</span>
                        <span className="ml-2">
                          {(state.activeConfiguration.config_data?.compatibility_score * 100).toFixed(1)}%
                        </span>
                      </div>
                      <div className="text-sm">
                        <span className="text-gray-400">Applied At:</span>
                        <span className="ml-2">
                          {new Date(state.activeConfiguration.applied_at * 1000).toLocaleTimeString()}
                        </span>
                      </div>
                    </div>
                  ) : (
                    <div className="text-gray-500 text-sm">No configuration applied</div>
                  )}
                </div>

                {/* Quick Actions */}
                <div className="bg-gray-900 border border-gray-700 rounded-lg p-4">
                  <h3 className="text-lg font-semibold mb-3">Quick Actions</h3>
                  <div className="space-y-2">
                    <button
                      onClick={() => startDiscovery("general task optimization")}
                      className="w-full px-4 py-2 bg-purple-600 hover:bg-purple-700 rounded-lg transition-colors text-sm"
                    >
                      Start General Discovery
                    </button>
                    <button
                      onClick={() => setState(prev => ({ ...prev, activeView: 'discovery' }))}
                      className="w-full px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded-lg transition-colors text-sm"
                    >
                      Open Discovery Panel
                    </button>
                  </div>
                </div>
              </div>
            </div>
          </div>
        );
    }
  };

  return (
    <WebSocketProvider>
      <AgenticDataProvider>
        <div className="min-h-screen bg-black text-white">
          {/* Header */}
          <div className="bg-gray-900 border-b border-gray-700 p-4">
            <div className="flex items-center justify-between">
              <h1 className="text-2xl font-bold flex items-center gap-2">
                <Brain className="w-8 h-8 text-purple-500" />
                Phase 7: Self-Guided Automatic Discovery of Agentic Expert Vector Configurations
              </h1>
              <div className="flex items-center gap-4">
                <div className="text-sm text-gray-400">
                  ADAS + Transformers² Self-Guided Integration
                </div>
                <div className="flex items-center gap-2">
                  <div className={`w-3 h-3 rounded-full ${
                    state.isConnected ? 'bg-green-500' : 'bg-red-500'
                  }`} />
                  <span className="text-sm text-gray-400">
                    {state.isConnected ? 'Connected' : 'Disconnected'}
                  </span>
                </div>
              </div>
            </div>

            {/* Navigation */}
            <div className="flex gap-2 mt-4">
              {[
                { id: 'overview', label: 'Overview', icon: Activity },
                { id: 'discovery', label: 'Expert Discovery', icon: Search },
                { id: 'self_guided', label: 'Self-Guided Composition', icon: Target },
                { id: 'transformers2', label: 'Transformers²', icon: GitBranch },
                { id: 'svd', label: 'SVD Analysis', icon: Layers },
                { id: 'phase2', label: 'Phase 2 Integration', icon: Network },
                { id: 'adas', label: 'ADAS Monitor', icon: Brain }
              ].map(({ id, label, icon: Icon }) => (
                <button
                  key={id}
                  onClick={() => setState(prev => ({ ...prev, activeView: id as any }))}
                  className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-colors ${
                    state.activeView === id
                      ? 'bg-purple-600 text-white'
                      : 'bg-gray-800 text-gray-300 hover:bg-gray-700'
                  }`}
                >
                  <Icon className="w-4 h-4" />
                  {label}
                </button>
              ))}
            </div>
          </div>

          {/* Main Content */}
          <div className="p-6 min-h-[calc(100vh-140px)]">
            {renderMainContent()}
          </div>
        </div>
      </AgenticDataProvider>
    </WebSocketProvider>
  );
};

// Status Indicator Component
const StatusIndicator: React.FC<{
  label: string;
  active: boolean;
  icon: React.ReactNode;
}> = ({ label, active, icon }) => (
  <div className={`flex items-center gap-2 p-2 rounded-lg ${
    active ? 'bg-green-900/30 border border-green-700' : 'bg-gray-800 border border-gray-700'
  }`}>
    <div className={active ? 'text-green-400' : 'text-gray-500'}>
      {icon}
    </div>
    <div>
      <div className="text-xs text-gray-400">{label}</div>
      <div className={`text-sm font-semibold ${active ? 'text-green-400' : 'text-gray-500'}`}>
        {active ? 'Active' : 'Inactive'}
      </div>
    </div>
  </div>
);

export default AgenticDashboard;