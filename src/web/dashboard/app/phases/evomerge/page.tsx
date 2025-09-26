'use client';

import { useState, useEffect } from 'react';
import { ArrowLeft, Dna, Trophy, Shuffle, GitBranch, Activity, Play, Square } from 'lucide-react';
import Link from 'next/link';
import { ModelSelector } from '../../../components/ModelSelector';
import { EvolutionTree3D } from '../../../components/EvolutionTree3D';

export default function EvoMergeTournamentPage() {
  const [currentGeneration, setCurrentGeneration] = useState(0);
  const [trainingStatus, setTrainingStatus] = useState('idle');
  const [evolutionData, setEvolutionData] = useState<any>(null);
  const [selectedModels, setSelectedModels] = useState<string[]>([]);
  const [evolutionTree, setEvolutionTree] = useState<any[]>([]);
  const [selectedNode, setSelectedNode] = useState<any>(null);
  const [showModelSelector, setShowModelSelector] = useState(true);
  const [storageStats, setStorageStats] = useState<any>(null);

  // Tournament rules as implemented in backend
  const tournamentRules = {
    initial: "3 Cognate models → 8 merged combinations",
    winners: "Top 2 models → 6 children (3 mutations each)",
    losers: "Bottom 6 models → 2 children (2 groups of 3 merged)",
    termination: "50 generations OR 3 consecutive tests with no improvement"
  };

  const startEvolution = async () => {
    if (selectedModels.length !== 3) {
      alert('Please select 3 models before starting evolution');
      return;
    }

    setTrainingStatus('evolving');
    setCurrentGeneration(0);
    setEvolutionTree([]);
    setShowModelSelector(false);

    try {
      const response = await fetch('http://localhost:8001/api/evomerge/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model_paths: selectedModels,
          model_source: 'custom',
          generations: 50,
          storage_dir: './models/evomerge',
          keep_generations: 2,
          cleanup_final: true,
          track_lineage: true,
          validate_compatibility: true
        })
      });

      const data = await response.json();
      console.log('Evolution started:', data);
      setEvolutionData(data);

      // Set up WebSocket for real-time updates
      setupWebSocket(data.session_id || 'default');

    } catch (error) {
      console.error('Failed to start evolution:', error);
      setTrainingStatus('error');
    }
  };

  // WebSocket setup for real-time updates
  const setupWebSocket = (sessionId: string) => {
    const ws = new WebSocket(`ws://localhost:8085/evomerge/${sessionId}`);

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);

      switch (data.type) {
        case 'generation_complete':
          setCurrentGeneration(data.generation);
          // Fetch updated evolution tree
          fetchEvolutionTree();
          break;
        case 'evolution_complete':
          setTrainingStatus('complete');
          setStorageStats(data.storage_stats);
          break;
        case 'error':
          setTrainingStatus('error');
          console.error('Evolution error:', data.message);
          break;
        default:
          console.log('WebSocket message:', data);
      }
    };

    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
    };

    ws.onclose = () => {
      console.log('WebSocket connection closed');
    };
  };

  // Fetch evolution tree data
  const fetchEvolutionTree = async () => {
    try {
      const response = await fetch('http://localhost:8001/api/evomerge/evolution-tree');
      const data = await response.json();
      setEvolutionTree(data.tree || []);
    } catch (error) {
      console.error('Failed to fetch evolution tree:', error);
    }
  };

  const stopEvolution = async () => {
    try {
      const response = await fetch('http://localhost:8001/api/evomerge/stop', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
      });
      const data = await response.json();
      console.log('Evolution stopped:', data);
      setTrainingStatus('idle');
    } catch (error) {
      console.error('Failed to stop evolution:', error);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-950 via-pink-950 to-purple-950 text-white p-8">
      <Link href="/" className="flex items-center gap-2 text-purple-400 hover:text-purple-300 mb-8">
        <ArrowLeft className="w-5 h-5" />
        Back to Dashboard
      </Link>

      <div className="mb-8">
        <h1 className="text-5xl font-bold mb-4 bg-gradient-to-r from-purple-400 to-pink-400 bg-clip-text text-transparent flex items-center gap-4">
          <Dna className="w-12 h-12 text-purple-400" />
          Phase 2: EvoMerge Tournament Evolution
        </h1>
        <p className="text-xl text-gray-400">
          Model-agnostic evolutionary optimization with 3D visualization
        </p>
      </div>

      {/* Model Selection */}
      {showModelSelector && (
        <ModelSelector
          onModelsSelected={setSelectedModels}
          apiUrl="http://localhost:8001"
        />
      )}

      {/* Reset Model Selection Button */}
      {!showModelSelector && trainingStatus === 'idle' && (
        <div className="mb-6">
          <button
            onClick={() => setShowModelSelector(true)}
            className="px-4 py-2 bg-purple-600 hover:bg-purple-700 rounded-lg transition-colors"
          >
            Change Model Selection
          </button>
        </div>
      )}

      {/* Tournament Rules Banner */}
      <div className="bg-gradient-to-r from-yellow-600/20 to-orange-600/20 rounded-2xl p-6 border border-yellow-500/30 mb-8">
        <h3 className="text-2xl font-bold mb-4 flex items-center gap-2">
          <Trophy className="w-6 h-6 text-yellow-400" />
          Tournament Selection Algorithm (Backend Implemented)
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <p className="text-sm text-gray-400">Initial Population</p>
            <p className="text-lg">{tournamentRules.initial}</p>
          </div>
          <div>
            <p className="text-sm text-gray-400">Winner Strategy</p>
            <p className="text-lg text-green-400">{tournamentRules.winners}</p>
          </div>
          <div>
            <p className="text-sm text-gray-400">Loser Strategy (Chaos)</p>
            <p className="text-lg text-orange-400">{tournamentRules.losers}</p>
          </div>
          <div>
            <p className="text-sm text-gray-400">Termination</p>
            <p className="text-lg">{tournamentRules.termination}</p>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Current Generation Status */}
        <div className="bg-white/5 backdrop-blur-lg rounded-xl p-6 border border-white/10">
          <h3 className="text-xl font-bold mb-4 flex items-center gap-2">
            <Activity className="w-6 h-6 text-blue-400" />
            Generation {currentGeneration}/50 Tournament
          </h3>

          {/* Winners Section */}
          <div className="bg-green-500/10 rounded-lg p-4 border border-green-500/30 mb-4">
            <div className="flex items-center justify-between mb-3">
              <span className="text-green-400 font-bold">🏆 Top 2 Winners</span>
              <span className="text-sm text-gray-400">→ 6 Children</span>
            </div>
            <div className="grid grid-cols-2 gap-2">
              <div className="bg-green-900/30 rounded p-2">
                <div className="text-xs text-gray-400">Winner 1 (Fitness: 0.924)</div>
                <div className="text-white">Creates 3 mutated children</div>
              </div>
              <div className="bg-green-900/30 rounded p-2">
                <div className="text-xs text-gray-400">Winner 2 (Fitness: 0.887)</div>
                <div className="text-white">Creates 3 mutated children</div>
              </div>
            </div>
          </div>

          {/* Losers Section */}
          <div className="bg-orange-500/10 rounded-lg p-4 border border-orange-500/30">
            <div className="flex items-center justify-between mb-3">
              <span className="text-orange-400 font-bold">
                <Shuffle className="w-4 h-4 inline mr-1" />
                Bottom 6 (Chaos Pool)
              </span>
              <span className="text-sm text-gray-400">→ 2 Children</span>
            </div>
            <div className="grid grid-cols-2 gap-2">
              <div className="bg-orange-900/30 rounded p-2">
                <div className="text-xs text-gray-400">Group 1: Models 3, 4, 5</div>
                <div className="text-white">Merge → 1 chaos child</div>
              </div>
              <div className="bg-orange-900/30 rounded p-2">
                <div className="text-xs text-gray-400">Group 2: Models 6, 7, 8</div>
                <div className="text-white">Merge → 1 chaos child</div>
              </div>
            </div>
          </div>
        </div>

        {/* 3D Evolution Tree Visualization */}
        <div className="bg-white/5 backdrop-blur-lg rounded-xl p-6 border border-white/10">
          <h3 className="text-xl font-bold mb-4 flex items-center gap-2">
            <GitBranch className="w-6 h-6 text-purple-400" />
            3D Evolution Tree Visualization
          </h3>

          {evolutionTree.length > 0 ? (
            <EvolutionTree3D
              treeData={evolutionTree}
              currentGeneration={currentGeneration}
              animateBreeding={trainingStatus === 'evolving'}
              onNodeClick={setSelectedNode}
            />
          ) : (
            <div className="h-[600px] bg-gradient-to-b from-gray-900 to-black rounded-xl flex items-center justify-center">
              <div className="text-center text-gray-400">
                <GitBranch className="w-16 h-16 mx-auto mb-4 opacity-50" />
                <p className="text-lg">Evolution tree will appear here</p>
                <p className="text-sm">Start evolution to see 3D visualization</p>
              </div>
            </div>
          )}

          {/* Progress Bar */}
          <div className="mt-4">
            <div className="flex justify-between text-sm mb-2">
              <span>Evolution Progress</span>
              <span>{Math.round((currentGeneration / 50) * 100)}%</span>
            </div>
            <div className="w-full bg-gray-700 rounded-full h-3">
              <div
                className="bg-gradient-to-r from-purple-500 to-pink-500 h-3 rounded-full transition-all duration-300"
                style={{ width: `${(currentGeneration / 50) * 100}%` }}
              />
            </div>
          </div>
        </div>
      </div>

      {/* Training Control */}
      <div className="mt-8 bg-white/5 backdrop-blur-lg rounded-xl p-6 border border-white/10">
        <div className="flex items-center justify-between">
          <div>
            <p className="text-lg">Status: <span className={
              trainingStatus === 'evolving' ? 'text-yellow-400' :
              trainingStatus === 'complete' ? 'text-green-400' :
              trainingStatus === 'error' ? 'text-red-400' : 'text-gray-400'
            }>{trainingStatus.toUpperCase()}</span></p>
            <p className="text-sm text-gray-400">Backend: Python (port 8001) | Frontend: Next.js (port 3000)</p>
            {evolutionData && (
              <p className="text-sm text-gray-400">Session: {evolutionData.session_id}</p>
            )}
          </div>
          {trainingStatus === 'evolving' ? (
            <button
              onClick={stopEvolution}
              className="bg-red-600 hover:bg-red-700 px-6 py-3 rounded-xl font-bold transition-colors flex items-center gap-2"
            >
              <Square className="w-5 h-5" />
              Stop Evolution
            </button>
          ) : (
            <button
              onClick={startEvolution}
              disabled={trainingStatus === 'evolving' || selectedModels.length !== 3}
              className="bg-purple-600 hover:bg-purple-700 disabled:bg-gray-600 px-6 py-3 rounded-xl font-bold transition-colors flex items-center gap-2"
            >
              <Play className="w-5 h-5" />
              {selectedModels.length === 3 ? 'Start Evolution' : `Select ${3 - selectedModels.length} more model(s)`}
            </button>
          )}
        </div>
      </div>

      {/* Metrics */}
      <div className="mt-8 grid grid-cols-1 md:grid-cols-4 gap-6">
        <div className="bg-gradient-to-r from-green-600/20 to-emerald-600/20 rounded-2xl p-6 border border-green-500/30">
          <h3 className="text-xl font-bold mb-2">Current Generation</h3>
          <p className="text-3xl font-bold text-green-400">{currentGeneration}/50</p>
        </div>
        <div className="bg-gradient-to-r from-purple-600/20 to-pink-600/20 rounded-2xl p-6 border border-purple-500/30">
          <h3 className="text-xl font-bold mb-2">Population Size</h3>
          <p className="text-3xl font-bold text-purple-400">8</p>
          <p className="text-sm text-gray-400">Fixed (Tournament)</p>
        </div>
        <div className="bg-gradient-to-r from-blue-600/20 to-cyan-600/20 rounded-2xl p-6 border border-blue-500/30">
          <h3 className="text-xl font-bold mb-2">Best Fitness</h3>
          <p className="text-3xl font-bold text-blue-400">0.924</p>
        </div>
        <div className="bg-gradient-to-r from-orange-600/20 to-yellow-600/20 rounded-2xl p-6 border border-orange-500/30">
          <h3 className="text-xl font-bold mb-2">Diversity</h3>
          <p className="text-3xl font-bold text-orange-400">0.67</p>
        </div>
        {storageStats && (
          <div className="bg-gradient-to-r from-cyan-600/20 to-blue-600/20 rounded-2xl p-6 border border-cyan-500/30">
            <h3 className="text-xl font-bold mb-2">Storage</h3>
            <p className="text-3xl font-bold text-cyan-400">{storageStats.total_size_mb?.toFixed(1) || '0'}MB</p>
            <p className="text-sm text-gray-400">
              {storageStats.total_models_deleted || 0} models deleted
            </p>
          </div>
        )}
      </div>

      {/* Selected Node Details */}
      {selectedNode && (
        <div className="mt-8 bg-white/5 backdrop-blur-lg rounded-xl p-6 border border-white/10">
          <h3 className="text-xl font-bold mb-4">Selected Model Details</h3>
          <div className="grid grid-cols-2 gap-4">
            <div>
              <p className="text-sm text-gray-400">Model ID</p>
              <p className="text-white font-bold">{selectedNode.id}</p>
            </div>
            <div>
              <p className="text-sm text-gray-400">Generation</p>
              <p className="text-white font-bold">{selectedNode.generation}</p>
            </div>
            <div>
              <p className="text-sm text-gray-400">Fitness Score</p>
              <p className="text-white font-bold">{selectedNode.fitness.toFixed(4)}</p>
            </div>
            <div>
              <p className="text-sm text-gray-400">Type</p>
              <p className="text-white font-bold capitalize">{selectedNode.type}</p>
            </div>
            {selectedNode.breeding_type && (
              <div>
                <p className="text-sm text-gray-400">Breeding Type</p>
                <p className="text-white font-bold capitalize">{selectedNode.breeding_type}</p>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}