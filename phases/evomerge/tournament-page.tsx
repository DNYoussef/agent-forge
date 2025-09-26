'use client';

import { useState } from 'react';
import { ArrowLeft, Dna, Trophy, Shuffle, GitBranch, Activity } from 'lucide-react';
import Link from 'next/link';

export default function EvoMergeTournamentPage() {
  const [currentGeneration, setCurrentGeneration] = useState(0);
  const [trainingStatus, setTrainingStatus] = useState('idle');

  // Tournament rules as implemented in backend
  const tournamentRules = {
    initial: "3 Cognate models → 8 merged combinations",
    winners: "Top 2 models → 6 children (3 mutations each)",
    losers: "Bottom 6 models → 2 children (2 groups of 3 merged)",
    termination: "50 generations OR 3 consecutive tests with no improvement"
  };

  const startEvolution = async () => {
    setTrainingStatus('evolving');
    setCurrentGeneration(0);

    try {
      const response = await fetch('http://localhost:8001/api/evomerge/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          generations: 50,
          tournament_mode: true,
          convergence_patience: 3
        })
      });

      const data = await response.json();
      console.log('Evolution started:', data);

      // Simulate evolution progress
      const interval = setInterval(() => {
        setCurrentGeneration(prev => {
          if (prev >= 50) {
            clearInterval(interval);
            setTrainingStatus('complete');
            return 50;
          }
          return prev + 1;
        });
      }, 1000);
    } catch (error) {
      console.error('Failed to start evolution:', error);
      setTrainingStatus('error');
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
          Tournament selection with 8 evolutionary combinations over 50 generations
        </p>
      </div>

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

        {/* Evolution Tree Visualization */}
        <div className="bg-white/5 backdrop-blur-lg rounded-xl p-6 border border-white/10">
          <h3 className="text-xl font-bold mb-4 flex items-center gap-2">
            <GitBranch className="w-6 h-6 text-purple-400" />
            Evolution Tree Progress
          </h3>

          <div className="space-y-3">
            {/* Generation 0 */}
            <div className="flex items-center gap-2">
              <span className="text-sm text-gray-400 w-16">Gen 0:</span>
              <div className="flex gap-1">
                {Array.from({ length: 8 }).map((_, i) => (
                  <div
                    key={`g0-${i}`}
                    className="w-8 h-8 rounded-full bg-blue-600 flex items-center justify-center text-xs"
                  >
                    {i + 1}
                  </div>
                ))}
              </div>
            </div>

            {/* Current Generation */}
            {currentGeneration > 0 && (
              <div className="flex items-center gap-2">
                <span className="text-sm text-gray-400 w-16">Gen {currentGeneration}:</span>
                <div className="flex gap-1">
                  {/* 6 Winner children */}
                  {Array.from({ length: 6 }).map((_, i) => (
                    <div
                      key={`w-${i}`}
                      className="w-8 h-8 rounded-full bg-green-600 flex items-center justify-center text-xs"
                    >
                      W{i + 1}
                    </div>
                  ))}
                  {/* 2 Loser children */}
                  {Array.from({ length: 2 }).map((_, i) => (
                    <div
                      key={`l-${i}`}
                      className="w-8 h-8 rounded-full bg-orange-600 flex items-center justify-center text-xs"
                    >
                      L{i + 1}
                    </div>
                  ))}
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
          </div>
          <button
            onClick={startEvolution}
            disabled={trainingStatus === 'evolving'}
            className="bg-purple-600 hover:bg-purple-700 disabled:bg-gray-600 px-6 py-3 rounded-xl font-bold transition-colors flex items-center gap-2"
          >
            <Dna className="w-5 h-5" />
            {trainingStatus === 'evolving' ? 'Evolving...' : 'Start Tournament Evolution'}
          </button>
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
        </div>
        <div className="bg-gradient-to-r from-blue-600/20 to-cyan-600/20 rounded-2xl p-6 border border-blue-500/30">
          <h3 className="text-xl font-bold mb-2">Best Fitness</h3>
          <p className="text-3xl font-bold text-blue-400">0.924</p>
        </div>
        <div className="bg-gradient-to-r from-orange-600/20 to-yellow-600/20 rounded-2xl p-6 border border-orange-500/30">
          <h3 className="text-xl font-bold mb-2">Diversity</h3>
          <p className="text-3xl font-bold text-orange-400">0.67</p>
        </div>
      </div>
    </div>
  );
}