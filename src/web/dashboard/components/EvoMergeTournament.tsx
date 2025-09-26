'use client';

import React, { useState, useEffect } from 'react';
import { Trophy, Shuffle, GitBranch, Activity } from 'lucide-react';

interface TournamentGeneration {
  generation: number;
  winners: Array<{
    id: string;
    fitness: number;
    children: string[];
  }>;
  losers: Array<{
    id: string;
    fitness: number;
    group: number;
  }>;
  loserChildren: Array<{
    id: string;
    parentGroups: number[];
  }>;
}

interface EvolutionTreeNode {
  id: string;
  generation: number;
  fitness: number;
  type: 'winner' | 'loser' | 'winner_child' | 'loser_child';
  parent?: string;
  position: { x: number; y: number };
}

export function EvoMergeTournament() {
  const [currentGeneration, setCurrentGeneration] = useState(0);
  const [generations, setGenerations] = useState<TournamentGeneration[]>([]);
  const [treeNodes, setTreeNodes] = useState<EvolutionTreeNode[]>([]);

  // Tournament rules as implemented in backend
  const tournamentRules = {
    initial: "3 Cognate models → 8 merged combinations",
    winners: "Top 2 models → 6 children (3 mutations each)",
    losers: "Bottom 6 models → 2 children (2 groups of 3 merged)",
    termination: "50 generations OR 3 consecutive tests with no improvement"
  };

  useEffect(() => {
    // Generate sample evolution tree data
    const generateTreeData = () => {
      const nodes: EvolutionTreeNode[] = [];
      const genWidth = 800;
      const genHeight = 60;

      // Initial 8 models
      for (let i = 0; i < 8; i++) {
        nodes.push({
          id: `G0-M${i}`,
          generation: 0,
          fitness: 0.5 + Math.random() * 0.3,
          type: 'winner',
          position: { x: (i * genWidth / 8) + 50, y: 50 }
        });
      }

      // Subsequent generations
      for (let gen = 1; gen <= currentGeneration && gen <= 50; gen++) {
        const y = 50 + gen * genHeight;

        // Winner children (6)
        for (let i = 0; i < 6; i++) {
          nodes.push({
            id: `G${gen}-W${i}`,
            generation: gen,
            fitness: 0.6 + Math.random() * 0.35,
            type: 'winner_child',
            parent: `G${gen-1}-M${Math.floor(i/3)}`,
            position: { x: (i * genWidth / 8) + 50, y }
          });
        }

        // Loser children (2)
        for (let i = 0; i < 2; i++) {
          nodes.push({
            id: `G${gen}-L${i}`,
            generation: gen,
            fitness: 0.4 + Math.random() * 0.3,
            type: 'loser_child',
            position: { x: ((6 + i) * genWidth / 8) + 50, y }
          });
        }
      }

      setTreeNodes(nodes);
    };

    generateTreeData();
  }, [currentGeneration]);

  return (
    <div className="space-y-6">
      {/* Tournament Rules Display */}
      <div className="bg-gradient-to-r from-purple-900/20 to-pink-900/20 rounded-xl p-6 border border-purple-500/30">
        <h3 className="text-xl font-bold text-white mb-4 flex items-center gap-2">
          <Trophy className="w-6 h-6 text-yellow-400" />
          Tournament Selection Algorithm
        </h3>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="bg-black/20 rounded-lg p-4">
            <div className="text-sm text-gray-400">Initial Population</div>
            <div className="text-white font-semibold">{tournamentRules.initial}</div>
          </div>

          <div className="bg-green-900/20 rounded-lg p-4 border border-green-500/30">
            <div className="text-sm text-green-400">Winners Strategy</div>
            <div className="text-white font-semibold">{tournamentRules.winners}</div>
          </div>

          <div className="bg-orange-900/20 rounded-lg p-4 border border-orange-500/30">
            <div className="text-sm text-orange-400">Chaos Preservation</div>
            <div className="text-white font-semibold">{tournamentRules.losers}</div>
          </div>

          <div className="bg-black/20 rounded-lg p-4">
            <div className="text-sm text-gray-400">Termination</div>
            <div className="text-white font-semibold">{tournamentRules.termination}</div>
          </div>
        </div>
      </div>

      {/* Current Generation Breakdown */}
      <div className="bg-white/5 backdrop-blur rounded-xl p-6 border border-white/10">
        <h3 className="text-xl font-bold text-white mb-4 flex items-center gap-2">
          <Activity className="w-6 h-6 text-blue-400" />
          Generation {currentGeneration} Tournament
        </h3>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {/* Winners Section */}
          <div className="bg-green-500/10 rounded-lg p-4 border border-green-500/30">
            <div className="flex items-center justify-between mb-3">
              <span className="text-green-400 font-bold">🏆 Top 2 Winners</span>
              <span className="text-sm text-gray-400">→ 6 Children</span>
            </div>

            <div className="space-y-2">
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

            <div className="space-y-2">
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

        {/* Population Summary */}
        <div className="mt-4 p-3 bg-purple-900/20 rounded-lg border border-purple-500/30">
          <div className="flex items-center justify-between">
            <span className="text-purple-400 font-semibold">Next Generation</span>
            <span className="text-white">
              6 winner children + 2 loser children = <span className="font-bold text-purple-400">8 models</span>
            </span>
          </div>
        </div>
      </div>

      {/* Evolution Tree Visualization */}
      <div className="bg-white/5 backdrop-blur rounded-xl p-6 border border-white/10">
        <h3 className="text-xl font-bold text-white mb-4 flex items-center gap-2">
          <GitBranch className="w-6 h-6 text-purple-400" />
          Evolution Tree (Generation {currentGeneration}/50)
        </h3>

        <div className="relative h-96 overflow-auto bg-black/20 rounded-lg p-4">
          <svg width="900" height={Math.max(400, (currentGeneration + 1) * 60 + 100)} className="w-full">
            {/* Draw connections */}
            {treeNodes.filter(n => n.parent).map(node => {
              const parent = treeNodes.find(p => p.id === node.parent);
              if (!parent) return null;

              return (
                <line
                  key={`${parent.id}-${node.id}`}
                  x1={parent.position.x}
                  y1={parent.position.y}
                  x2={node.position.x}
                  y2={node.position.y}
                  stroke={node.type === 'winner_child' ? '#10b981' : '#f97316'}
                  strokeWidth="1"
                  opacity="0.3"
                />
              );
            })}

            {/* Draw nodes */}
            {treeNodes.map(node => (
              <g key={node.id}>
                <circle
                  cx={node.position.x}
                  cy={node.position.y}
                  r={node.type.includes('winner') ? 8 : 6}
                  fill={
                    node.type === 'winner' ? '#10b981' :
                    node.type === 'winner_child' ? '#86efac' :
                    node.type === 'loser' ? '#f97316' : '#fb923c'
                  }
                  stroke="white"
                  strokeWidth="1"
                />
                <text
                  x={node.position.x}
                  y={node.position.y - 12}
                  textAnchor="middle"
                  className="text-xs fill-gray-400"
                >
                  {node.fitness.toFixed(3)}
                </text>
              </g>
            ))}

            {/* Generation labels */}
            {Array.from({ length: currentGeneration + 1 }, (_, i) => (
              <text
                key={`gen-${i}`}
                x={10}
                y={50 + i * 60}
                className="text-sm fill-gray-500"
              >
                Gen {i}
              </text>
            ))}
          </svg>
        </div>

        <div className="mt-4 flex items-center justify-between text-sm">
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-1">
              <div className="w-3 h-3 rounded-full bg-green-500"></div>
              <span className="text-gray-400">Winner/Winner Child</span>
            </div>
            <div className="flex items-center gap-1">
              <div className="w-3 h-3 rounded-full bg-orange-500"></div>
              <span className="text-gray-400">Loser/Chaos Child</span>
            </div>
          </div>
          <button
            onClick={() => setCurrentGeneration(prev => Math.min(prev + 1, 50))}
            className="px-4 py-2 bg-purple-600 hover:bg-purple-700 rounded-lg text-white transition-colors"
            disabled={currentGeneration >= 50}
          >
            Next Generation
          </button>
        </div>
      </div>
    </div>
  );
}