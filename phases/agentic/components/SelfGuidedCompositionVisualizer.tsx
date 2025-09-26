'use client';

import React, { useState, useEffect } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, Text, Box, Sphere, Line } from '@react-three/drei';
import {
  Brain,
  Target,
  Zap,
  Eye,
  Settings,
  TrendingUp,
  CheckCircle,
  AlertCircle,
  RefreshCw,
  Network
} from 'lucide-react';

interface SelfGuidedComposition {
  composition_id: string;
  composition_strategy: string;
  target_layers?: string[];
  target_capabilities?: string[];
  composition_rationale: string;
  expert_vector_design: {
    num_experts: number;
    expert_specifications: Array<{
      expert_id: string;
      target_layer?: string;
      target_capability?: string;
      adaptation_strength: number;
      specialization: string;
    }>;
  };
  confidence: number;
  self_evaluation?: {
    alignment_score: number;
    feasibility_score: number;
    innovation_score: number;
    should_refine: boolean;
  };
  refinement_applied?: boolean;
}

interface SelfGuidedCompositionVisualizerProps {
  compositions: any;
  discoverySession: any;
  modelExaminations: number;
}

export const SelfGuidedCompositionVisualizer: React.FC<SelfGuidedCompositionVisualizerProps> = ({
  compositions,
  discoverySession,
  modelExaminations
}) => {
  const [selectedComposition, setSelectedComposition] = useState<SelfGuidedComposition | null>(null);
  const [viewMode, setViewMode] = useState<'overview' | 'examination' | 'proposals' | 'refinement'>('overview');
  const [animationSpeed, setAnimationSpeed] = useState(1.0);

  // Extract compositions data
  const selfExamination = compositions?.self_examination;
  const modelProposals = compositions?.model_proposals || [];
  const refinedCompositions = compositions?.refined_compositions || [];
  const compositionLearning = compositions?.composition_learning;

  useEffect(() => {
    if (refinedCompositions.length > 0 && !selectedComposition) {
      setSelectedComposition(refinedCompositions[0]);
    }
  }, [refinedCompositions, selectedComposition]);

  const renderSelfExaminationView = () => (
    <div className="space-y-6">
      {/* Model Self-Examination Header */}
      <div className="bg-gradient-to-r from-blue-900 to-purple-900 rounded-lg p-6">
        <h2 className="text-2xl font-bold mb-4 flex items-center gap-3">
          <Eye className="w-8 h-8 text-blue-400" />
          Model Self-Examination
          <span className="text-sm bg-blue-800 px-3 py-1 rounded-full">
            Transformers² Self-Guided
          </span>
        </h2>
        <p className="text-gray-300">
          The model examines its own weight patterns and identifies capabilities for expert vector composition.
        </p>
      </div>

      <div className="grid grid-cols-2 gap-6">
        {/* Weight Pattern Analysis */}
        <div className="bg-gray-900 border border-gray-700 rounded-lg p-4">
          <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
            <Brain className="w-5 h-5 text-purple-400" />
            Weight Pattern Insights
          </h3>
          {selfExamination?.weight_patterns_identified ? (
            <div className="space-y-3">
              <div className="text-sm text-gray-400">
                Analyzed Layers: {selfExamination.weight_patterns_identified.length}
              </div>
              <div className="max-h-48 overflow-y-auto space-y-2">
                {selfExamination.weight_patterns_identified.slice(0, 5).map((pattern: any, idx: number) => (
                  <div key={idx} className="bg-gray-800 rounded-lg p-3">
                    <div className="text-sm font-mono text-blue-300">{pattern.layer_name}</div>
                    <div className="text-xs text-gray-400 mt-1">
                      Magnitude: {pattern.weight_magnitude?.toFixed(4)} |
                      Sparsity: {(pattern.sparsity * 100)?.toFixed(1)}% |
                      Adaptation: {pattern.adaptation_potential?.toFixed(4)}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          ) : (
            <div className="text-gray-500 text-sm">No weight patterns analyzed yet</div>
          )}
        </div>

        {/* Self-Identified Capabilities */}
        <div className="bg-gray-900 border border-gray-700 rounded-lg p-4">
          <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
            <Target className="w-5 h-5 text-green-400" />
            Self-Identified Capabilities
          </h3>
          {selfExamination?.self_identified_capabilities ? (
            <div className="space-y-2">
              {selfExamination.self_identified_capabilities.map((capability: string, idx: number) => (
                <div key={idx} className="bg-green-900/20 border border-green-700 rounded-lg p-2">
                  <div className="text-sm text-green-300 font-medium">{capability}</div>
                </div>
              ))}
              <div className="mt-4 text-xs text-gray-400">
                Task Type: {selfExamination.task_type}
              </div>
            </div>
          ) : (
            <div className="text-gray-500 text-sm">No capabilities identified yet</div>
          )}
        </div>
      </div>

      {/* Adaptation Opportunities */}
      <div className="bg-gray-900 border border-gray-700 rounded-lg p-4">
        <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
          <Zap className="w-5 h-5 text-yellow-400" />
          High-Potential Adaptation Layers
        </h3>
        {selfExamination?.adaptation_opportunities ? (
          <div className="grid grid-cols-3 gap-3">
            {selfExamination.adaptation_opportunities.map((layer: string, idx: number) => (
              <div key={idx} className="bg-yellow-900/20 border border-yellow-700 rounded-lg p-3 text-center">
                <div className="text-sm font-mono text-yellow-300">{layer}</div>
                <div className="text-xs text-yellow-400 mt-1">High Variance</div>
              </div>
            ))}
          </div>
        ) : (
          <div className="text-gray-500 text-sm">No adaptation opportunities identified</div>
        )}
      </div>
    </div>
  );

  const renderProposalsView = () => (
    <div className="space-y-6">
      {/* Model Proposals Header */}
      <div className="bg-gradient-to-r from-green-900 to-blue-900 rounded-lg p-6">
        <h2 className="text-2xl font-bold mb-4 flex items-center gap-3">
          <Settings className="w-8 h-8 text-green-400" />
          Model-Generated Proposals
          <span className="text-sm bg-green-800 px-3 py-1 rounded-full">
            Self-Guided Composition
          </span>
        </h2>
        <p className="text-gray-300">
          The model proposes expert vector compositions based on its self-knowledge and ADAS discoveries.
        </p>
      </div>

      <div className="grid grid-cols-1 gap-4">
        {modelProposals.map((proposal: SelfGuidedComposition, idx: number) => (
          <div
            key={idx}
            className={`bg-gray-900 border rounded-lg p-4 cursor-pointer transition-all ${
              selectedComposition?.composition_id === proposal.composition_id
                ? 'border-blue-500 bg-blue-900/20'
                : 'border-gray-700 hover:border-gray-600'
            }`}
            onClick={() => setSelectedComposition(proposal)}
          >
            <div className="flex items-center justify-between mb-3">
              <h3 className="text-lg font-semibold flex items-center gap-2">
                <div className={`w-3 h-3 rounded-full ${
                  proposal.composition_strategy === 'adas_self_hybrid' ? 'bg-purple-400' :
                  proposal.composition_strategy === 'weight_pattern_guided' ? 'bg-blue-400' :
                  'bg-green-400'
                }`} />
                {proposal.composition_strategy.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
              </h3>
              <div className="flex items-center gap-2">
                <div className="text-sm text-gray-400">
                  Confidence: {(proposal.confidence * 100).toFixed(0)}%
                </div>
                {proposal.refinement_applied && (
                  <RefreshCw className="w-4 h-4 text-yellow-400" />
                )}
              </div>
            </div>

            <p className="text-gray-300 text-sm mb-3">{proposal.composition_rationale}</p>

            <div className="grid grid-cols-2 gap-4">
              <div>
                <div className="text-xs text-gray-400 mb-1">Expert Specifications</div>
                <div className="text-sm">
                  {proposal.expert_vector_design.num_experts} experts configured
                </div>
              </div>
              <div>
                <div className="text-xs text-gray-400 mb-1">Target Layers/Capabilities</div>
                <div className="text-sm">
                  {proposal.target_layers?.length || proposal.target_capabilities?.length || 0} targets
                </div>
              </div>
            </div>

            {proposal.self_evaluation && (
              <div className="mt-3 grid grid-cols-3 gap-2">
                <div className="bg-gray-800 rounded p-2 text-center">
                  <div className="text-xs text-gray-400">Alignment</div>
                  <div className="text-sm font-bold">
                    {(proposal.self_evaluation.alignment_score * 100).toFixed(0)}%
                  </div>
                </div>
                <div className="bg-gray-800 rounded p-2 text-center">
                  <div className="text-xs text-gray-400">Feasibility</div>
                  <div className="text-sm font-bold">
                    {(proposal.self_evaluation.feasibility_score * 100).toFixed(0)}%
                  </div>
                </div>
                <div className="bg-gray-800 rounded p-2 text-center">
                  <div className="text-xs text-gray-400">Innovation</div>
                  <div className="text-sm font-bold">
                    {(proposal.self_evaluation.innovation_score * 100).toFixed(0)}%
                  </div>
                </div>
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );

  const render3DCompositionView = () => {
    if (!selectedComposition) return null;

    return (
      <div className="h-96 bg-black rounded-lg border border-gray-700">
        <Canvas camera={{ position: [0, 0, 10], fov: 75 }}>
          <OrbitControls enablePan={true} enableZoom={true} enableRotate={true} />
          <ambientLight intensity={0.5} />
          <pointLight position={[10, 10, 10]} />

          {/* Central model representation */}
          <Sphere position={[0, 0, 0]} args={[1]} material-color="#6366f1" material-transparent material-opacity={0.7} />

          {/* Expert vectors as connections */}
          {selectedComposition.expert_vector_design.expert_specifications.map((expert, idx) => {
            const angle = (idx / selectedComposition.expert_vector_design.num_experts) * Math.PI * 2;
            const radius = 3;
            const x = Math.cos(angle) * radius;
            const y = Math.sin(angle) * radius;

            return (
              <group key={expert.expert_id}>
                {/* Expert node */}
                <Box position={[x, y, 0]} args={[0.5, 0.5, 0.5]} material-color="#10b981" />

                {/* Connection line */}
                <Line
                  points={[[0, 0, 0], [x, y, 0]]}
                  color="#6366f1"
                  lineWidth={2}
                />

                {/* Expert label */}
                <Text
                  position={[x, y - 1, 0]}
                  fontSize={0.3}
                  color="white"
                  anchorX="center"
                  anchorY="middle"
                >
                  {expert.specialization.split('_')[0]}
                </Text>
              </group>
            );
          })}

          {/* Model self-examination indicator */}
          <group position={[0, 3, 0]}>
            <Sphere args={[0.3]} material-color="#f59e0b" />
            <Text
              position={[0, -0.7, 0]}
              fontSize={0.4}
              color="#f59e0b"
              anchorX="center"
              anchorY="middle"
            >
              Self-Examination
            </Text>
          </group>
        </Canvas>
      </div>
    );
  };

  const renderCompositionDetails = () => {
    if (!selectedComposition) return null;

    return (
      <div className="bg-gray-900 border border-gray-700 rounded-lg p-4">
        <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
          <Target className="w-5 h-5 text-blue-400" />
          Composition Details
        </h3>

        <div className="space-y-4">
          <div>
            <div className="text-sm text-gray-400 mb-1">Strategy</div>
            <div className="text-lg font-bold text-blue-300">
              {selectedComposition.composition_strategy.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
            </div>
          </div>

          <div>
            <div className="text-sm text-gray-400 mb-2">Expert Vector Specifications</div>
            <div className="space-y-2">
              {selectedComposition.expert_vector_design.expert_specifications.map((expert, idx) => (
                <div key={idx} className="bg-gray-800 rounded-lg p-3">
                  <div className="flex justify-between items-start">
                    <div>
                      <div className="font-medium text-green-300">{expert.expert_id}</div>
                      <div className="text-sm text-gray-400">
                        {expert.target_layer && `Layer: ${expert.target_layer}`}
                        {expert.target_capability && `Capability: ${expert.target_capability}`}
                      </div>
                      <div className="text-xs text-gray-500 mt-1">
                        Specialization: {expert.specialization}
                      </div>
                    </div>
                    <div className="text-right">
                      <div className="text-sm font-bold">
                        {(expert.adaptation_strength * 100).toFixed(0)}%
                      </div>
                      <div className="text-xs text-gray-400">Strength</div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {selectedComposition.self_evaluation && (
            <div>
              <div className="text-sm text-gray-400 mb-2">Model Self-Evaluation</div>
              <div className="bg-gray-800 rounded-lg p-3">
                <div className="grid grid-cols-3 gap-4 text-center">
                  <div>
                    <div className="text-lg font-bold text-blue-300">
                      {(selectedComposition.self_evaluation.alignment_score * 100).toFixed(0)}%
                    </div>
                    <div className="text-xs text-gray-400">Task Alignment</div>
                  </div>
                  <div>
                    <div className="text-lg font-bold text-green-300">
                      {(selectedComposition.self_evaluation.feasibility_score * 100).toFixed(0)}%
                    </div>
                    <div className="text-xs text-gray-400">Feasibility</div>
                  </div>
                  <div>
                    <div className="text-lg font-bold text-purple-300">
                      {(selectedComposition.self_evaluation.innovation_score * 100).toFixed(0)}%
                    </div>
                    <div className="text-xs text-gray-400">Innovation</div>
                  </div>
                </div>
                {selectedComposition.self_evaluation.should_refine && (
                  <div className="mt-3 text-center">
                    <div className="inline-flex items-center gap-2 text-yellow-400 text-sm">
                      <RefreshCw className="w-4 h-4" />
                      Model recommended refinement
                    </div>
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      </div>
    );
  };

  return (
    <div className="min-h-screen bg-black text-white p-6">
      {/* Header */}
      <div className="mb-6">
        <div className="flex items-center justify-between mb-4">
          <h1 className="text-3xl font-bold flex items-center gap-3">
            <Brain className="w-8 h-8 text-purple-500" />
            Self-Guided Expert Vector Composition
          </h1>
          <div className="flex items-center gap-4">
            <div className="text-sm text-gray-400">
              Model Self-Examinations: {modelExaminations}
            </div>
            <div className="text-sm text-gray-400">
              Compositions: {refinedCompositions.length}
            </div>
          </div>
        </div>

        {/* View Mode Toggle */}
        <div className="flex gap-2">
          {[
            { id: 'overview', label: 'Overview', icon: Network },
            { id: 'examination', label: 'Self-Examination', icon: Eye },
            { id: 'proposals', label: 'Model Proposals', icon: Settings },
            { id: 'refinement', label: '3D Visualization', icon: Target }
          ].map(({ id, label, icon: Icon }) => (
            <button
              key={id}
              onClick={() => setViewMode(id as any)}
              className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-colors ${
                viewMode === id
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
      {viewMode === 'examination' && renderSelfExaminationView()}
      {viewMode === 'proposals' && renderProposalsView()}
      {viewMode === 'refinement' && (
        <div className="space-y-6">
          {render3DCompositionView()}
          {renderCompositionDetails()}
        </div>
      )}
      {viewMode === 'overview' && (
        <div className="grid grid-cols-2 gap-6">
          <div>{renderSelfExaminationView()}</div>
          <div>{renderProposalsView()}</div>
        </div>
      )}
    </div>
  );
};