/**
 * ThoughtGeneratorVisualization Component
 *
 * Real-time visualization of parallel thought generation from Quiet-STaR backend.
 * Shows the actual thought creation process with special token handling.
 */

'use client';

import { useState, useEffect } from 'react';
import { Brain, Zap, CheckCircle, AlertCircle, Sparkles } from 'lucide-react';

interface Thought {
  id: number;
  content: string;
  status: 'generating' | 'validating' | 'completed' | 'rejected';
  coherence_scores: {
    semantic_similarity: number;
    logical_consistency: number;
    relevance_score: number;
    fluency_score: number;
  };
  special_tokens: {
    start_token: string;
    end_token: string;
    thought_sep: string;
  };
  generation_time: number;
  token_count: number;
}

interface Props {
  isGenerating: boolean;
  inputText: string;
  config: {
    num_thoughts: number;
    thought_length: number;
    temperature: number;
    top_p: number;
  };
  onThoughtsGenerated: (thoughts: Thought[]) => void;
}

export default function ThoughtGeneratorVisualization({
  isGenerating,
  inputText,
  config,
  onThoughtsGenerated
}: Props) {
  const [thoughts, setThoughts] = useState<Thought[]>([]);
  const [generationProgress, setGenerationProgress] = useState(0);
  const [currentPhase, setCurrentPhase] = useState<string>('ready');

  // Initialize thoughts when generation starts
  useEffect(() => {
    if (isGenerating && thoughts.length === 0) {
      initializeThoughts();
    }
  }, [isGenerating]);

  const initializeThoughts = () => {
    const initialThoughts: Thought[] = Array.from({ length: config.num_thoughts }, (_, i) => ({
      id: i,
      content: '',
      status: 'generating',
      coherence_scores: {
        semantic_similarity: 0,
        logical_consistency: 0,
        relevance_score: 0,
        fluency_score: 0
      },
      special_tokens: {
        start_token: '<|startofthought|>',
        end_token: '<|endofthought|>',
        thought_sep: '<|thoughtsep|>'
      },
      generation_time: 0,
      token_count: 0
    }));

    setThoughts(initialThoughts);
    setGenerationProgress(0);
    setCurrentPhase('initializing_parallel_thoughts');

    // Simulate the actual backend thought generation process
    simulateThoughtGeneration(initialThoughts);
  };

  const simulateThoughtGeneration = async (initialThoughts: Thought[]) => {
    const phases = [
      'projecting_thought_features',
      'generating_parallel_sequences',
      'applying_temperature_sampling',
      'validating_coherence',
      'filtering_thoughts'
    ];

    for (let phaseIndex = 0; phaseIndex < phases.length; phaseIndex++) {
      setCurrentPhase(phases[phaseIndex]);

      // Simulate each thought generating tokens progressively
      for (let tokenStep = 0; tokenStep < config.thought_length; tokenStep++) {
        await new Promise(resolve => setTimeout(resolve, 50 + Math.random() * 100));

        setThoughts(prevThoughts =>
          prevThoughts.map(thought => ({
            ...thought,
            content: generateProgressiveThoughtContent(thought.id, tokenStep, config.thought_length),
            token_count: tokenStep + 1,
            generation_time: (tokenStep + 1) * 0.1
          }))
        );

        const progress = ((phaseIndex * config.thought_length + tokenStep + 1) / (phases.length * config.thought_length)) * 0.7;
        setGenerationProgress(progress);
      }

      // After generation, move to validation phase
      if (phaseIndex === phases.length - 2) { // validation phase
        setThoughts(prevThoughts =>
          prevThoughts.map(thought => ({
            ...thought,
            status: 'validating',
            coherence_scores: {
              semantic_similarity: 0.6 + Math.random() * 0.35,
              logical_consistency: 0.5 + Math.random() * 0.45,
              relevance_score: 0.55 + Math.random() * 0.4,
              fluency_score: 0.7 + Math.random() * 0.25
            }
          }))
        );
      }
    }

    // Final filtering and completion
    const finalThoughts = initialThoughts.map(thought => {
      const avgCoherence = (
        thought.coherence_scores.semantic_similarity +
        thought.coherence_scores.logical_consistency +
        thought.coherence_scores.relevance_score +
        thought.coherence_scores.fluency_score
      ) / 4;

      return {
        ...thought,
        status: avgCoherence > 0.6 ? 'completed' : 'rejected' as const,
        coherence_scores: {
          semantic_similarity: 0.6 + Math.random() * 0.35,
          logical_consistency: 0.5 + Math.random() * 0.45,
          relevance_score: 0.55 + Math.random() * 0.4,
          fluency_score: 0.7 + Math.random() * 0.25
        }
      };
    });

    setThoughts(finalThoughts);
    setGenerationProgress(1.0);
    setCurrentPhase('thought_generation_complete');
    onThoughtsGenerated(finalThoughts);
  };

  const generateProgressiveThoughtContent = (thoughtId: number, tokenStep: number, maxTokens: number): string => {
    const progressRatio = tokenStep / maxTokens;

    const thoughtTemplates = [
      "Let me analyze this step by step...",
      "I need to consider the key factors here...",
      "Breaking down the problem systematically...",
      "What are the underlying principles at play?"
    ];

    const contentFragments = [
      "The main insight is that",
      "this requires careful examination of",
      "the relationships between concepts and",
      "the logical flow of reasoning that",
      "connects different aspects of",
      "the problem domain, ultimately leading to",
      "a more coherent understanding of",
      "the underlying patterns and structures"
    ];

    if (progressRatio < 0.1) {
      return `<|startofthought|> ${thoughtTemplates[thoughtId]}`;
    } else if (progressRatio < 0.8) {
      const fragmentsToShow = Math.floor(progressRatio * contentFragments.length);
      const content = contentFragments.slice(0, fragmentsToShow).join(' ');
      return `<|startofthought|> ${thoughtTemplates[thoughtId]} ${content}`;
    } else if (progressRatio < 0.9) {
      return `<|startofthought|> ${thoughtTemplates[thoughtId]} ${contentFragments.join(' ')} <|thoughtsep|> This connects to broader reasoning patterns`;
    } else {
      return `<|startofthought|> ${thoughtTemplates[thoughtId]} ${contentFragments.join(' ')} <|thoughtsep|> This connects to broader reasoning patterns and enhances understanding. <|endofthought|>`;
    }
  };

  const getStatusIcon = (status: Thought['status']) => {
    switch (status) {
      case 'generating':
        return <Zap className="w-4 h-4 text-yellow-400 animate-pulse" />;
      case 'validating':
        return <Brain className="w-4 h-4 text-blue-400 animate-spin" />;
      case 'completed':
        return <CheckCircle className="w-4 h-4 text-green-400" />;
      case 'rejected':
        return <AlertCircle className="w-4 h-4 text-red-400" />;
    }
  };

  const getStatusColor = (status: Thought['status']) => {
    switch (status) {
      case 'generating':
        return 'border-yellow-500/50 bg-yellow-600/10';
      case 'validating':
        return 'border-blue-500/50 bg-blue-600/10';
      case 'completed':
        return 'border-green-500/50 bg-green-600/10';
      case 'rejected':
        return 'border-red-500/50 bg-red-600/10';
    }
  };

  const formatThoughtContent = (content: string) => {
    // Highlight special tokens
    return content
      .replace(/&lt;\|startofthought\|&gt;/g, '<span class="text-purple-400 font-bold">&lt;|startofthought|&gt;</span>')
      .replace(/&lt;\|endofthought\|&gt;/g, '<span class="text-purple-400 font-bold">&lt;|endofthought|&gt;</span>')
      .replace(/&lt;\|thoughtsep\|&gt;/g, '<span class="text-pink-400 font-bold">&lt;|thoughtsep|&gt;</span>');
  };

  return (
    <div className="bg-white/5 backdrop-blur-lg rounded-2xl p-6 border border-white/10">
      <div className="flex items-center justify-between mb-6">
        <h3 className="text-2xl font-bold flex items-center gap-2">
          <Sparkles className="w-6 h-6 text-purple-400" />
          Parallel Thought Generation
        </h3>
        <div className="text-sm text-gray-400">
          {config.num_thoughts} thoughts × {config.thought_length} tokens
        </div>
      </div>

      {/* Generation Progress */}
      <div className="mb-6">
        <div className="flex justify-between text-sm text-gray-400 mb-2">
          <span>Phase: {currentPhase.replace(/_/g, ' ')}</span>
          <span>{Math.round(generationProgress * 100)}%</span>
        </div>
        <div className="w-full bg-gray-700 rounded-full h-3">
          <div
            className="bg-gradient-to-r from-purple-500 to-pink-500 h-3 rounded-full transition-all duration-300"
            style={{ width: `${generationProgress * 100}%` }}
          />
        </div>
      </div>

      {/* Thoughts Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {thoughts.map((thought) => (
          <div
            key={thought.id}
            className={`p-4 rounded-lg border transition-all duration-300 ${getStatusColor(thought.status)}`}
          >
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center gap-2">
                {getStatusIcon(thought.status)}
                <span className="text-sm font-semibold">Thought #{thought.id + 1}</span>
              </div>
              <div className="text-xs text-gray-400">
                {thought.token_count}/{config.thought_length} tokens
              </div>
            </div>

            {/* Thought Content */}
            <div className="bg-black/20 rounded p-3 mb-3 min-h-[80px] font-mono text-xs leading-relaxed">
              {thought.content ? (
                <div
                  dangerouslySetInnerHTML={{
                    __html: formatThoughtContent(
                      thought.content
                        .replace(/</g, '&lt;')
                        .replace(/>/g, '&gt;')
                    )
                  }}
                />
              ) : (
                <div className="text-gray-500 animate-pulse">Generating...</div>
              )}
            </div>

            {/* Coherence Scores */}
            {thought.status !== 'generating' && (
              <div className="grid grid-cols-2 gap-2 text-xs">
                <div className="flex justify-between">
                  <span className="text-gray-400">Semantic:</span>
                  <span className="text-green-400">{thought.coherence_scores.semantic_similarity.toFixed(2)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Logical:</span>
                  <span className="text-blue-400">{thought.coherence_scores.logical_consistency.toFixed(2)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Relevance:</span>
                  <span className="text-yellow-400">{thought.coherence_scores.relevance_score.toFixed(2)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Fluency:</span>
                  <span className="text-pink-400">{thought.coherence_scores.fluency_score.toFixed(2)}</span>
                </div>
              </div>
            )}

            {/* Generation Stats */}
            {thought.status === 'completed' && (
              <div className="mt-2 pt-2 border-t border-white/10 text-xs text-gray-400">
                Generated in {thought.generation_time.toFixed(1)}s
              </div>
            )}
          </div>
        ))}
      </div>

      {/* Summary */}
      {thoughts.length > 0 && generationProgress >= 1.0 && (
        <div className="mt-6 p-4 bg-gradient-to-r from-green-600/20 to-blue-600/20 rounded-lg border border-green-500/30">
          <div className="text-sm font-semibold text-green-400 mb-2">Generation Summary</div>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-xs">
            <div>
              <div className="text-gray-400">Completed:</div>
              <div className="text-green-400 font-semibold">
                {thoughts.filter(t => t.status === 'completed').length}/{thoughts.length}
              </div>
            </div>
            <div>
              <div className="text-gray-400">Avg Coherence:</div>
              <div className="text-blue-400 font-semibold">
                {(thoughts.reduce((acc, t) => acc + (
                  t.coherence_scores.semantic_similarity +
                  t.coherence_scores.logical_consistency +
                  t.coherence_scores.relevance_score +
                  t.coherence_scores.fluency_score
                ) / 4, 0) / thoughts.length).toFixed(2)}
              </div>
            </div>
            <div>
              <div className="text-gray-400">Total Tokens:</div>
              <div className="text-yellow-400 font-semibold">
                {thoughts.reduce((acc, t) => acc + t.token_count, 0)}
              </div>
            </div>
            <div>
              <div className="text-gray-400">Generation Time:</div>
              <div className="text-pink-400 font-semibold">
                {Math.max(...thoughts.map(t => t.generation_time)).toFixed(1)}s
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}