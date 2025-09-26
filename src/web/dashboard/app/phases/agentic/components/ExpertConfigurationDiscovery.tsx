'use client';

import React, { useState, useCallback } from 'react';
import {
  Search,
  Play,
  Pause,
  CheckCircle,
  XCircle,
  AlertCircle,
  Zap,
  Brain,
  Target,
  TrendingUp
} from 'lucide-react';

interface ExpertConfigurationDiscoveryProps {
  session: any;
  onStartDiscovery: (task: string) => void;
  onApplyConfiguration: (configId: string) => void;
  activeConfiguration: any;
}

export const ExpertConfigurationDiscovery: React.FC<ExpertConfigurationDiscoveryProps> = ({
  session,
  onStartDiscovery,
  onApplyConfiguration,
  activeConfiguration
}) => {
  const [taskInput, setTaskInput] = useState('');
  const [isDiscovering, setIsDiscovering] = useState(false);
  const [selectedConfig, setSelectedConfig] = useState<string | null>(null);

  const handleStartDiscovery = useCallback(() => {
    if (taskInput.trim()) {
      setIsDiscovering(true);
      onStartDiscovery(taskInput);
      // Simulation - in real app, this would be handled by WebSocket
      setTimeout(() => setIsDiscovering(false), 3000);
    }
  }, [taskInput, onStartDiscovery]);

  const handleApplyConfiguration = useCallback(() => {
    if (selectedConfig) {
      onApplyConfiguration(selectedConfig);
    }
  }, [selectedConfig, onApplyConfiguration]);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-gradient-to-r from-purple-900 to-blue-900 rounded-lg p-6 border border-purple-500">
        <h2 className="text-2xl font-bold mb-2 flex items-center gap-2">
          <Search className="w-6 h-6" />
          Expert Configuration Discovery
        </h2>
        <p className="text-gray-300">
          ADAS meta-agent search automatically discovers optimal Transformers² expert vector configurations
        </p>
      </div>

      {/* Discovery Input */}
      <div className="bg-gray-900 border border-gray-700 rounded-lg p-6">
        <h3 className="text-lg font-semibold mb-4">Start New Discovery</h3>
        <div className="flex gap-3">
          <input
            type="text"
            value={taskInput}
            onChange={(e) => setTaskInput(e.target.value)}
            placeholder="Enter task description (e.g., 'mathematical reasoning', 'creative writing')..."
            className="flex-1 px-4 py-2 bg-gray-800 border border-gray-600 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:border-purple-500"
          />
          <button
            onClick={handleStartDiscovery}
            disabled={isDiscovering || !taskInput.trim()}
            className="px-6 py-2 bg-purple-600 hover:bg-purple-700 disabled:bg-gray-700 disabled:cursor-not-allowed rounded-lg transition-colors flex items-center gap-2"
          >
            {isDiscovering ? (
              <><Pause className="w-4 h-4" /> Discovering...</>
            ) : (
              <><Play className="w-4 h-4" /> Start Discovery</>
            )}
          </button>
        </div>
      </div>

      {/* Current Session */}
      {session && (
        <div className="grid grid-cols-2 gap-6">
          {/* Session Info */}
          <div className="bg-gray-900 border border-gray-700 rounded-lg p-6">
            <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
              <Brain className="w-5 h-5 text-purple-400" />
              Discovery Session
            </h3>
            <div className="space-y-3">
              <div>
                <span className="text-gray-400 text-sm">Task:</span>
                <p className="text-white">{session.task_description}</p>
              </div>
              <div>
                <span className="text-gray-400 text-sm">Session ID:</span>
                <p className="text-white font-mono text-xs">{session.session_id}</p>
              </div>
              <div>
                <span className="text-gray-400 text-sm">Task Analysis:</span>
                <div className="mt-1 space-y-1">
                  <div className="text-sm">
                    Type: <span className="text-purple-400">{session.task_analysis?.task_type}</span>
                  </div>
                  <div className="text-sm">
                    Complexity: <span className="text-blue-400">
                      {(session.task_analysis?.complexity_estimate * 100).toFixed(0)}%
                    </span>
                  </div>
                  <div className="text-sm">
                    Capabilities: <span className="text-green-400">
                      {session.task_analysis?.required_capabilities?.join(', ')}
                    </span>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Discovery Progress */}
          <div className="bg-gray-900 border border-gray-700 rounded-lg p-6">
            <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
              <TrendingUp className="w-5 h-5 text-green-400" />
              Discovery Progress
            </h3>
            <div className="space-y-3">
              <ProgressStep
                label="Task Analysis"
                status={session.task_analysis ? 'complete' : 'pending'}
                icon={<Brain className="w-4 h-4" />}
              />
              <ProgressStep
                label="Phase 2 Weight Insights"
                status={session.task_analysis?.weight_insights ? 'complete' : 'pending'}
                icon={<Zap className="w-4 h-4" />}
              />
              <ProgressStep
                label="ADAS Meta-Agent Search"
                status={session.discovery_results ? 'complete' : 'pending'}
                icon={<Search className="w-4 h-4" />}
              />
              <ProgressStep
                label="Transformers² Validation"
                status={session.validated_configurations ? 'complete' : 'pending'}
                icon={<CheckCircle className="w-4 h-4" />}
              />
              <ProgressStep
                label="Expert System Creation"
                status={session.optimized_system ? 'complete' : 'pending'}
                icon={<Target className="w-4 h-4" />}
              />
            </div>
          </div>
        </div>
      )}

      {/* Discovered Configurations */}
      {session?.validated_configurations && (
        <div className="bg-gray-900 border border-gray-700 rounded-lg p-6">
          <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
            <Target className="w-5 h-5 text-yellow-400" />
            Discovered Configurations ({session.validated_configurations.length})
          </h3>
          <div className="space-y-3">
            {session.validated_configurations.map((config: any, idx: number) => (
              <div
                key={config.original_config.config_id}
                className={`border rounded-lg p-4 cursor-pointer transition-all ${
                  selectedConfig === config.original_config.config_id
                    ? 'border-purple-500 bg-purple-900/20'
                    : 'border-gray-700 bg-gray-800/50 hover:border-gray-600'
                }`}
                onClick={() => setSelectedConfig(config.original_config.config_id)}
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <div className="text-2xl font-bold text-gray-500">#{idx + 1}</div>
                    <div>
                      <div className="font-semibold">
                        {config.original_config.task_specialization} Specialist
                      </div>
                      <div className="text-sm text-gray-400">
                        ID: {config.original_config.config_id}
                      </div>
                    </div>
                  </div>
                  <div className="text-right">
                    <div className="text-lg font-bold text-green-400">
                      {(config.compatibility_score * 100).toFixed(1)}%
                    </div>
                    <div className="text-xs text-gray-400">Compatibility</div>
                  </div>
                </div>
                <div className="mt-3 grid grid-cols-3 gap-2 text-sm">
                  <div>
                    <span className="text-gray-400">Performance:</span>
                    <span className="ml-1 text-white">
                      {config.original_config.performance_score.toFixed(3)}
                    </span>
                  </div>
                  <div>
                    <span className="text-gray-400">SVD Layers:</span>
                    <span className="ml-1 text-white">
                      {Object.keys(config.original_config.svd_components).length}
                    </span>
                  </div>
                  <div>
                    <span className="text-gray-400">Validation:</span>
                    <span className="ml-1 text-green-400">
                      {config.validation_result.test_successful ? 'Passed' : 'Failed'}
                    </span>
                  </div>
                </div>
              </div>
            ))}
          </div>

          {/* Apply Configuration */}
          <div className="mt-6 flex items-center justify-between">
            <div className="text-sm text-gray-400">
              {selectedConfig
                ? `Selected: ${selectedConfig}`
                : 'Select a configuration to apply'}
            </div>
            <button
              onClick={handleApplyConfiguration}
              disabled={!selectedConfig}
              className="px-6 py-2 bg-green-600 hover:bg-green-700 disabled:bg-gray-700 disabled:cursor-not-allowed rounded-lg transition-colors"
            >
              Apply Configuration
            </button>
          </div>
        </div>
      )}

      {/* Active Configuration Status */}
      {activeConfiguration && (
        <div className="bg-green-900/20 border border-green-700 rounded-lg p-6">
          <h3 className="text-lg font-semibold mb-3 flex items-center gap-2">
            <CheckCircle className="w-5 h-5 text-green-400" />
            Active Configuration
          </h3>
          <div className="grid grid-cols-2 gap-4 text-sm">
            <div>
              <span className="text-gray-400">Task:</span>
              <span className="ml-2">{activeConfiguration.task_description}</span>
            </div>
            <div>
              <span className="text-gray-400">Applied:</span>
              <span className="ml-2">
                {new Date(activeConfiguration.applied_at * 1000).toLocaleString()}
              </span>
            </div>
            <div>
              <span className="text-gray-400">Config ID:</span>
              <span className="ml-2 font-mono text-xs">
                {activeConfiguration.config_data?.original_config?.config_id}
              </span>
            </div>
            <div>
              <span className="text-gray-400">Status:</span>
              <span className="ml-2 text-green-400">
                {activeConfiguration.application_result?.status || 'Active'}
              </span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

// Progress Step Component
const ProgressStep: React.FC<{
  label: string;
  status: 'complete' | 'pending' | 'error';
  icon: React.ReactNode;
}> = ({ label, status, icon }) => (
  <div className="flex items-center gap-3">
    <div className={`p-2 rounded-lg ${
      status === 'complete'
        ? 'bg-green-900/30 text-green-400'
        : status === 'error'
        ? 'bg-red-900/30 text-red-400'
        : 'bg-gray-800 text-gray-500'
    }`}>
      {icon}
    </div>
    <div className="flex-1">
      <div className="text-sm">{label}</div>
      <div className="text-xs text-gray-500">
        {status === 'complete' ? 'Completed' : status === 'error' ? 'Failed' : 'Pending'}
      </div>
    </div>
    <div>
      {status === 'complete' && <CheckCircle className="w-5 h-5 text-green-400" />}
      {status === 'error' && <XCircle className="w-5 h-5 text-red-400" />}
      {status === 'pending' && <AlertCircle className="w-5 h-5 text-gray-500" />}
    </div>
  </div>
);