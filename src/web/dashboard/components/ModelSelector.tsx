'use client';

import React, { useState, useEffect } from 'react';
import { ChevronDown, Folder, Check, AlertCircle, Cpu, Database } from 'lucide-react';

interface ModelInfo {
  path: string;
  name: string;
  source: 'cognate' | 'custom' | 'huggingface';
  parameters: number;
  architecture: string;
  fitness_score?: number;
}

interface ModelSelectorProps {
  onModelsSelected: (models: string[]) => void;
  apiUrl?: string;
}

export function ModelSelector({ onModelsSelected, apiUrl = 'http://localhost:8001' }: ModelSelectorProps) {
  const [availableModels, setAvailableModels] = useState<ModelInfo[]>([]);
  const [selectedModels, setSelectedModels] = useState<string[]>(['', '', '']);
  const [modelSource, setModelSource] = useState<'cognate' | 'custom' | 'all'>('cognate');
  const [customDir, setCustomDir] = useState<string>('');
  const [loading, setLoading] = useState(false);
  const [compatibility, setCompatibility] = useState<boolean | null>(null);
  const [dropdownOpen, setDropdownOpen] = useState<number | null>(null);

  // Fetch available models
  useEffect(() => {
    fetchAvailableModels();
  }, [modelSource]);

  const fetchAvailableModels = async () => {
    setLoading(true);
    try {
      const sourceParam = modelSource === 'all' ? '' : `?source=${modelSource}`;
      const response = await fetch(`${apiUrl}/api/models/available${sourceParam}`);
      const data = await response.json();
      setAvailableModels(data.models || []);
    } catch (error) {
      console.error('Failed to fetch models:', error);
      setAvailableModels([]);
    } finally {
      setLoading(false);
    }
  };

  // Validate model compatibility
  useEffect(() => {
    if (selectedModels.every(m => m !== '')) {
      validateCompatibility();
    } else {
      setCompatibility(null);
    }
  }, [selectedModels]);

  const validateCompatibility = async () => {
    try {
      const response = await fetch(`${apiUrl}/api/models/validate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model_paths: selectedModels })
      });
      const data = await response.json();
      setCompatibility(data.compatible);
    } catch (error) {
      console.error('Failed to validate compatibility:', error);
      setCompatibility(false);
    }
  };

  const handleModelSelect = (index: number, modelPath: string) => {
    const newSelection = [...selectedModels];
    newSelection[index] = modelPath;
    setSelectedModels(newSelection);
    setDropdownOpen(null);

    // Notify parent if all models selected
    if (newSelection.every(m => m !== '')) {
      onModelsSelected(newSelection);
    }
  };

  const handleBrowseFolder = async () => {
    // In a real implementation, this would open a file dialog
    // For now, we'll use a prompt
    const dir = prompt('Enter custom model directory path:');
    if (dir) {
      setCustomDir(dir);
      setModelSource('custom');
      // Trigger re-fetch with custom directory
      try {
        const response = await fetch(`${apiUrl}/api/models/scan`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ directory: dir })
        });
        const data = await response.json();
        if (data.success) {
          fetchAvailableModels();
        }
      } catch (error) {
        console.error('Failed to scan custom directory:', error);
      }
    }
  };

  const formatParams = (params: number) => {
    if (params >= 1e9) return `${(params / 1e9).toFixed(1)}B`;
    if (params >= 1e6) return `${(params / 1e6).toFixed(1)}M`;
    return params.toString();
  };

  const getSourceIcon = (source: string) => {
    switch (source) {
      case 'cognate': return <Cpu className="w-4 h-4 text-purple-400" />;
      case 'custom': return <Folder className="w-4 h-4 text-blue-400" />;
      case 'huggingface': return <Database className="w-4 h-4 text-green-400" />;
      default: return null;
    }
  };

  const getSourceColor = (source: string) => {
    switch (source) {
      case 'cognate': return 'text-purple-400';
      case 'custom': return 'text-blue-400';
      case 'huggingface': return 'text-green-400';
      default: return 'text-gray-400';
    }
  };

  return (
    <div className="space-y-6">
      {/* Source Selection */}
      <div className="bg-white/5 backdrop-blur rounded-xl p-6 border border-white/10">
        <h3 className="text-lg font-bold text-white mb-4">Model Source</h3>
        <div className="flex gap-4">
          <button
            onClick={() => setModelSource('cognate')}
            className={`px-4 py-2 rounded-lg border transition-all ${
              modelSource === 'cognate'
                ? 'bg-purple-600 border-purple-500 text-white'
                : 'bg-white/5 border-white/10 text-gray-400 hover:border-purple-500'
            }`}
          >
            <Cpu className="w-4 h-4 inline mr-2" />
            Cognate Models
          </button>
          <button
            onClick={() => setModelSource('custom')}
            className={`px-4 py-2 rounded-lg border transition-all ${
              modelSource === 'custom'
                ? 'bg-blue-600 border-blue-500 text-white'
                : 'bg-white/5 border-white/10 text-gray-400 hover:border-blue-500'
            }`}
          >
            <Folder className="w-4 h-4 inline mr-2" />
            Custom Folder
          </button>
          <button
            onClick={() => setModelSource('all')}
            className={`px-4 py-2 rounded-lg border transition-all ${
              modelSource === 'all'
                ? 'bg-gray-600 border-gray-500 text-white'
                : 'bg-white/5 border-white/10 text-gray-400 hover:border-gray-500'
            }`}
          >
            All Available
          </button>
          {modelSource === 'custom' && (
            <button
              onClick={handleBrowseFolder}
              className="px-4 py-2 rounded-lg bg-white/5 border border-white/10 text-white hover:bg-white/10 transition-all"
            >
              Browse...
            </button>
          )}
        </div>
        {customDir && (
          <p className="mt-2 text-sm text-gray-400">Custom directory: {customDir}</p>
        )}
      </div>

      {/* Model Selection Dropdowns */}
      <div className="bg-white/5 backdrop-blur rounded-xl p-6 border border-white/10">
        <h3 className="text-lg font-bold text-white mb-4">Select 3 Models for Evolution</h3>

        <div className="space-y-4">
          {[0, 1, 2].map(index => (
            <div key={index} className="relative">
              <label className="text-sm text-gray-400 mb-1 block">
                Model {index + 1} {index === 0 && '(Primary)'}
              </label>

              <button
                onClick={() => setDropdownOpen(dropdownOpen === index ? null : index)}
                className="w-full px-4 py-3 bg-black/20 border border-white/10 rounded-lg text-left text-white hover:bg-black/30 transition-all flex items-center justify-between"
              >
                <div className="flex items-center gap-2">
                  {selectedModels[index] ? (
                    <>
                      {getSourceIcon(
                        availableModels.find(m => m.path === selectedModels[index])?.source || ''
                      )}
                      <span className="font-medium">
                        {availableModels.find(m => m.path === selectedModels[index])?.name || 'Unknown'}
                      </span>
                      <span className="text-sm text-gray-400">
                        ({formatParams(
                          availableModels.find(m => m.path === selectedModels[index])?.parameters || 0
                        )})
                      </span>
                    </>
                  ) : (
                    <span className="text-gray-400">Select a model...</span>
                  )}
                </div>
                <ChevronDown className={`w-5 h-5 transition-transform ${
                  dropdownOpen === index ? 'rotate-180' : ''
                }`} />
              </button>

              {/* Dropdown Menu */}
              {dropdownOpen === index && (
                <div className="absolute z-10 w-full mt-2 bg-gray-900 border border-white/20 rounded-lg shadow-2xl max-h-64 overflow-y-auto">
                  {loading ? (
                    <div className="p-4 text-center text-gray-400">Loading models...</div>
                  ) : availableModels.length === 0 ? (
                    <div className="p-4 text-center text-gray-400">No models available</div>
                  ) : (
                    availableModels.map(model => (
                      <button
                        key={model.path}
                        onClick={() => handleModelSelect(index, model.path)}
                        className="w-full px-4 py-3 hover:bg-white/10 transition-all text-left flex items-center justify-between group"
                        disabled={selectedModels.includes(model.path) && selectedModels[index] !== model.path}
                      >
                        <div className="flex items-center gap-3">
                          {getSourceIcon(model.source)}
                          <div>
                            <div className="text-white font-medium">{model.name}</div>
                            <div className="text-xs text-gray-400">
                              {model.architecture} • {formatParams(model.parameters)}
                              {model.fitness_score && ` • Fitness: ${model.fitness_score.toFixed(3)}`}
                            </div>
                          </div>
                        </div>
                        {selectedModels[index] === model.path && (
                          <Check className="w-5 h-5 text-green-400" />
                        )}
                        {selectedModels.includes(model.path) && selectedModels[index] !== model.path && (
                          <span className="text-xs text-gray-500">Already selected</span>
                        )}
                      </button>
                    ))
                  )}
                </div>
              )}
            </div>
          ))}
        </div>

        {/* Compatibility Status */}
        {compatibility !== null && (
          <div className={`mt-4 p-3 rounded-lg flex items-center gap-2 ${
            compatibility
              ? 'bg-green-500/10 border border-green-500/30 text-green-400'
              : 'bg-red-500/10 border border-red-500/30 text-red-400'
          }`}>
            {compatibility ? (
              <>
                <Check className="w-5 h-5" />
                <span>Models are compatible for merging</span>
              </>
            ) : (
              <>
                <AlertCircle className="w-5 h-5" />
                <span>Models may not be compatible - different architectures or sizes</span>
              </>
            )}
          </div>
        )}
      </div>

      {/* Model Statistics */}
      {selectedModels.every(m => m !== '') && (
        <div className="bg-white/5 backdrop-blur rounded-xl p-6 border border-white/10">
          <h3 className="text-lg font-bold text-white mb-4">Selection Summary</h3>
          <div className="grid grid-cols-3 gap-4">
            {selectedModels.map((modelPath, index) => {
              const model = availableModels.find(m => m.path === modelPath);
              if (!model) return null;

              return (
                <div key={index} className="bg-black/20 rounded-lg p-3">
                  <div className="flex items-center gap-2 mb-2">
                    {getSourceIcon(model.source)}
                    <span className={`text-sm font-bold ${getSourceColor(model.source)}`}>
                      Model {index + 1}
                    </span>
                  </div>
                  <div className="text-xs text-gray-400 space-y-1">
                    <div>Params: {formatParams(model.parameters)}</div>
                    <div>Arch: {model.architecture}</div>
                    {model.fitness_score && (
                      <div>Fitness: {model.fitness_score.toFixed(3)}</div>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}