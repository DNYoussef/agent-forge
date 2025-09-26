/**
 * CoherenceScoreBreakdown Component
 *
 * Detailed breakdown of the 4-metric coherence validation system from Quiet-STaR backend.
 * Shows semantic similarity, logical consistency, relevance, and fluency scores.
 */

'use client';

import { useState } from 'react';
import { TrendingUp, Brain, Target, BookOpen, ChevronDown, ChevronUp } from 'lucide-react';

interface CoherenceScores {
  semantic_similarity: number;
  logical_consistency: number;
  relevance_score: number;
  fluency_score: number;
}

interface Props {
  scores: CoherenceScores;
  threshold: number;
  thoughtId?: number;
  isExpanded?: boolean;
  onToggleExpanded?: () => void;
}

const METRIC_DEFINITIONS = {
  semantic_similarity: {
    name: 'Semantic Similarity',
    icon: Target,
    color: 'green',
    description: 'Cosine similarity between input and thought embeddings',
    interpretation: {
      high: 'Strong semantic alignment with input context',
      medium: 'Moderate semantic relevance to input',
      low: 'Weak semantic connection to input'
    }
  },
  logical_consistency: {
    name: 'Logical Consistency',
    icon: Brain,
    color: 'blue',
    description: 'Entropy-based measure of prediction confidence',
    interpretation: {
      high: 'Highly confident and consistent predictions',
      medium: 'Moderately consistent logical flow',
      low: 'Inconsistent or uncertain reasoning'
    }
  },
  relevance_score: {
    name: 'Relevance Score',
    icon: TrendingUp,
    color: 'yellow',
    description: 'Token overlap between input and thought content',
    interpretation: {
      high: 'High contextual relevance to input',
      medium: 'Some relevant connections to input',
      low: 'Limited relevance to input context'
    }
  },
  fluency_score: {
    name: 'Fluency Score',
    icon: BookOpen,
    color: 'pink',
    description: 'Perplexity-based language quality measure',
    interpretation: {
      high: 'Natural, fluent language generation',
      medium: 'Acceptable language quality',
      low: 'Disfluent or unnatural language'
    }
  }
};

export default function CoherenceScoreBreakdown({
  scores,
  threshold,
  thoughtId,
  isExpanded = false,
  onToggleExpanded
}: Props) {
  const [expandedMetric, setExpandedMetric] = useState<string | null>(null);

  const getOverallScore = (): number => {
    // Weighted average as used in backend (quietstar.py line 456-464)
    const weights = {
      semantic_similarity: 0.3,
      logical_consistency: 0.3,
      relevance_score: 0.25,
      fluency_score: 0.15
    };

    return (
      scores.semantic_similarity * weights.semantic_similarity +
      scores.logical_consistency * weights.logical_consistency +
      scores.relevance_score * weights.relevance_score +
      scores.fluency_score * weights.fluency_score
    );
  };

  const getScoreLevel = (score: number): 'high' | 'medium' | 'low' => {
    if (score >= 0.75) return 'high';
    if (score >= 0.5) return 'medium';
    return 'low';
  };

  const getScoreColor = (score: number, baseColor: string): string => {
    const level = getScoreLevel(score);
    const colors = {
      green: {
        high: 'text-green-400 bg-green-600/20 border-green-500/50',
        medium: 'text-green-300 bg-green-600/15 border-green-500/30',
        low: 'text-green-200 bg-green-600/10 border-green-500/20'
      },
      blue: {
        high: 'text-blue-400 bg-blue-600/20 border-blue-500/50',
        medium: 'text-blue-300 bg-blue-600/15 border-blue-500/30',
        low: 'text-blue-200 bg-blue-600/10 border-blue-500/20'
      },
      yellow: {
        high: 'text-yellow-400 bg-yellow-600/20 border-yellow-500/50',
        medium: 'text-yellow-300 bg-yellow-600/15 border-yellow-500/30',
        low: 'text-yellow-200 bg-yellow-600/10 border-yellow-500/20'
      },
      pink: {
        high: 'text-pink-400 bg-pink-600/20 border-pink-500/50',
        medium: 'text-pink-300 bg-pink-600/15 border-pink-500/30',
        low: 'text-pink-200 bg-pink-600/10 border-pink-500/20'
      }
    };

    return colors[baseColor as keyof typeof colors][level];
  };

  const overallScore = getOverallScore();
  const passesThreshold = overallScore >= threshold;

  return (
    <div className="bg-white/5 backdrop-blur-lg rounded-xl p-4 border border-white/10">
      {/* Header */}
      <div
        className="flex items-center justify-between cursor-pointer"
        onClick={onToggleExpanded}
      >
        <div className="flex items-center gap-3">
          <div className={`p-2 rounded-lg ${passesThreshold ? 'bg-green-600/20 border border-green-500/50' : 'bg-red-600/20 border border-red-500/50'}`}>
            <Brain className={`w-5 h-5 ${passesThreshold ? 'text-green-400' : 'text-red-400'}`} />
          </div>
          <div>
            <h4 className="text-lg font-semibold">
              Coherence Validation {thoughtId !== undefined && `#${thoughtId + 1}`}
            </h4>
            <div className="text-sm text-gray-400">
              Overall Score: <span className={passesThreshold ? 'text-green-400' : 'text-red-400'}>
                {overallScore.toFixed(3)}
              </span> / {threshold.toFixed(2)} threshold
            </div>
          </div>
        </div>

        {onToggleExpanded && (
          <div className="flex items-center gap-2">
            <div className={`px-3 py-1 rounded-full text-xs font-semibold ${
              passesThreshold ? 'bg-green-600/20 text-green-400' : 'bg-red-600/20 text-red-400'
            }`}>
              {passesThreshold ? 'VALID' : 'REJECTED'}
            </div>
            {isExpanded ? <ChevronUp className="w-5 h-5" /> : <ChevronDown className="w-5 h-5" />}
          </div>
        )}
      </div>

      {/* Metrics Grid */}
      {(isExpanded || !onToggleExpanded) && (
        <div className="mt-4 space-y-3">
          {Object.entries(METRIC_DEFINITIONS).map(([key, metric]) => {
            const score = scores[key as keyof CoherenceScores];
            const level = getScoreLevel(score);
            const Icon = metric.icon;

            return (
              <div key={key} className="space-y-2">
                {/* Metric Header */}
                <div
                  className="flex items-center justify-between cursor-pointer p-2 rounded-lg hover:bg-white/5 transition-colors"
                  onClick={() => setExpandedMetric(expandedMetric === key ? null : key)}
                >
                  <div className="flex items-center gap-3">
                    <div className={`p-1.5 rounded ${getScoreColor(score, metric.color)}`}>
                      <Icon className="w-4 h-4" />
                    </div>
                    <div>
                      <div className="text-sm font-semibold">{metric.name}</div>
                      <div className="text-xs text-gray-400">{score.toFixed(3)}</div>
                    </div>
                  </div>

                  <div className="flex items-center gap-3">
                    {/* Score Bar */}
                    <div className="w-20 bg-gray-700 rounded-full h-2">
                      <div
                        className={`h-2 rounded-full transition-all duration-300 ${
                          metric.color === 'green' ? 'bg-green-400' :
                          metric.color === 'blue' ? 'bg-blue-400' :
                          metric.color === 'yellow' ? 'bg-yellow-400' :
                          'bg-pink-400'
                        }`}
                        style={{ width: `${score * 100}%` }}
                      />
                    </div>

                    <ChevronDown className={`w-4 h-4 transition-transform ${
                      expandedMetric === key ? 'rotate-180' : ''
                    }`} />
                  </div>
                </div>

                {/* Expanded Metric Details */}
                {expandedMetric === key && (
                  <div className="ml-8 p-3 bg-black/20 rounded-lg border border-white/10">
                    <div className="text-xs text-gray-300 mb-2">
                      {metric.description}
                    </div>
                    <div className="text-xs">
                      <span className="text-gray-400">Interpretation: </span>
                      <span className={
                        level === 'high' ? 'text-green-300' :
                        level === 'medium' ? 'text-yellow-300' :
                        'text-red-300'
                      }>
                        {metric.interpretation[level]}
                      </span>
                    </div>

                    {/* Algorithm Details */}
                    <div className="mt-2 pt-2 border-t border-white/10">
                      <div className="text-xs text-gray-500">
                        {key === 'semantic_similarity' && 'F.cosine_similarity(input_repr, thought_repr)'}
                        {key === 'logical_consistency' && '1.0 - (entropy.mean() / log(vocab_size))'}
                        {key === 'relevance_score' && 'len(input_tokens ∩ thought_tokens) / len(input_tokens ∪ thought_tokens)'}
                        {key === 'fluency_score' && '1.0 / (1.0 + perplexity / 10.0)'}
                      </div>
                    </div>
                  </div>
                )}
              </div>
            );
          })}

          {/* Weighted Scoring Formula */}
          <div className="mt-4 p-3 bg-gradient-to-r from-purple-600/20 to-blue-600/20 rounded-lg border border-purple-500/30">
            <div className="text-sm font-semibold text-purple-400 mb-2">
              Weighted Scoring Formula
            </div>
            <div className="text-xs font-mono text-gray-300 space-y-1">
              <div>Overall = Semantic(30%) + Logical(30%) + Relevance(25%) + Fluency(15%)</div>
              <div className="text-gray-400">
                = {scores.semantic_similarity.toFixed(2)}×0.3 + {scores.logical_consistency.toFixed(2)}×0.3 +
                  {scores.relevance_score.toFixed(2)}×0.25 + {scores.fluency_score.toFixed(2)}×0.15
              </div>
              <div className={`font-semibold ${passesThreshold ? 'text-green-400' : 'text-red-400'}`}>
                = {overallScore.toFixed(3)} {passesThreshold ? '≥' : '<'} {threshold.toFixed(2)} threshold
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}