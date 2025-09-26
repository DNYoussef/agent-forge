/**
 * ThoughtInspector Component
 *
 * Interactive inspector for examining generated thoughts in detail.
 * Shows thought content, coherence breakdowns, and special token analysis.
 */

'use client';

import { useState } from 'react';
import { Eye, EyeOff, Search, Copy, Check, Expand, Minimize, Code2 } from 'lucide-react';
import CoherenceScoreBreakdown from './CoherenceScoreBreakdown';

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
  thoughts: Thought[];
  coherenceThreshold: number;
  searchTerm: string;
  onSearchChange: (term: string) => void;
  className?: string;
}

export default function ThoughtInspector({
  thoughts,
  coherenceThreshold,
  searchTerm,
  onSearchChange,
  className = ''
}: Props) {
  const [expandedThought, setExpandedThought] = useState<number | null>(null);
  const [showRejected, setShowRejected] = useState(false);
  const [viewMode, setViewMode] = useState<'formatted' | 'raw'>('formatted');
  const [copiedId, setCopiedId] = useState<number | null>(null);

  const filteredThoughts = thoughts.filter(thought => {
    const matchesSearch = !searchTerm ||
      thought.content.toLowerCase().includes(searchTerm.toLowerCase()) ||
      thought.id.toString().includes(searchTerm);

    const matchesFilter = showRejected || thought.status !== 'rejected';

    return matchesSearch && matchesFilter;
  });

  const copyToClipboard = async (content: string, thoughtId: number) => {
    try {
      await navigator.clipboard.writeText(content);
      setCopiedId(thoughtId);
      setTimeout(() => setCopiedId(null), 2000);
    } catch (error) {
      console.error('Failed to copy:', error);
    }
  };

  const parseThoughtContent = (content: string) => {
    const parts = content.split(/(<\|[^|]+\|>)/g);
    return parts.map((part, index) => {
      if (part.startsWith('<|') && part.endsWith('|>')) {
        return {
          type: 'token',
          content: part,
          index
        };
      }
      return {
        type: 'text',
        content: part.trim(),
        index
      };
    }).filter(part => part.content.length > 0);
  };

  const formatThoughtForDisplay = (content: string) => {
    if (viewMode === 'raw') {
      return content;
    }

    const parts = parseThoughtContent(content);
    return parts.map(part => {
      if (part.type === 'token') {
        let colorClass = 'text-purple-400';
        if (part.content.includes('sep')) colorClass = 'text-pink-400';
        if (part.content.includes('end')) colorClass = 'text-green-400';

        return (
          <span key={part.index} className={`${colorClass} font-bold bg-white/10 px-1 rounded`}>
            {part.content}
          </span>
        );
      }
      return <span key={part.index} className="text-gray-100">{part.content}</span>;
    });
  };

  const getOverallScore = (scores: Thought['coherence_scores']): number => {
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

  const getStatusBadge = (thought: Thought) => {
    const overallScore = getOverallScore(thought.coherence_scores);
    const passesThreshold = overallScore >= coherenceThreshold;

    switch (thought.status) {
      case 'generating':
        return <span className="px-2 py-1 text-xs bg-yellow-600/20 text-yellow-400 rounded-full">GENERATING</span>;
      case 'validating':
        return <span className="px-2 py-1 text-xs bg-blue-600/20 text-blue-400 rounded-full">VALIDATING</span>;
      case 'completed':
        return <span className="px-2 py-1 text-xs bg-green-600/20 text-green-400 rounded-full">
          ✓ VALID ({overallScore.toFixed(2)})
        </span>;
      case 'rejected':
        return <span className="px-2 py-1 text-xs bg-red-600/20 text-red-400 rounded-full">
          ✗ REJECTED ({overallScore.toFixed(2)})
        </span>;
    }
  };

  return (
    <div className={`bg-white/5 backdrop-blur-lg rounded-2xl p-6 border border-white/10 ${className}`}>
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <h3 className="text-2xl font-bold flex items-center gap-2">
          <Eye className="w-6 h-6 text-cyan-400" />
          Thought Inspector
        </h3>
        <div className="text-sm text-gray-400">
          {filteredThoughts.length} of {thoughts.length} thoughts
        </div>
      </div>

      {/* Controls */}
      <div className="flex flex-col sm:flex-row gap-4 mb-6">
        {/* Search */}
        <div className="flex-1 relative">
          <Search className="w-5 h-5 absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400" />
          <input
            type="text"
            placeholder="Search thoughts by content or ID..."
            value={searchTerm}
            onChange={(e) => onSearchChange(e.target.value)}
            className="w-full pl-10 pr-4 py-2 bg-white/10 border border-white/20 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:border-cyan-400"
          />
        </div>

        {/* View Mode Toggle */}
        <div className="flex items-center gap-2 bg-white/10 rounded-lg p-1">
          <button
            onClick={() => setViewMode('formatted')}
            className={`px-3 py-1 text-xs rounded ${viewMode === 'formatted' ? 'bg-cyan-600/20 text-cyan-400' : 'text-gray-400'}`}
          >
            Formatted
          </button>
          <button
            onClick={() => setViewMode('raw')}
            className={`px-3 py-1 text-xs rounded ${viewMode === 'raw' ? 'bg-cyan-600/20 text-cyan-400' : 'text-gray-400'}`}
          >
            <Code2 className="w-3 h-3 inline mr-1" />
            Raw
          </button>
        </div>

        {/* Show Rejected Toggle */}
        <button
          onClick={() => setShowRejected(!showRejected)}
          className={`flex items-center gap-2 px-3 py-2 rounded-lg text-sm ${
            showRejected ? 'bg-red-600/20 text-red-400' : 'bg-white/10 text-gray-400'
          }`}
        >
          {showRejected ? <Eye className="w-4 h-4" /> : <EyeOff className="w-4 h-4" />}
          Show Rejected
        </button>
      </div>

      {/* Thoughts List */}
      <div className="space-y-4">
        {filteredThoughts.length === 0 ? (
          <div className="text-center py-8 text-gray-400">
            {searchTerm ? 'No thoughts match your search.' : 'No thoughts to display.'}
          </div>
        ) : (
          filteredThoughts.map((thought) => {
            const isExpanded = expandedThought === thought.id;
            const overallScore = getOverallScore(thought.coherence_scores);

            return (
              <div
                key={thought.id}
                className={`border rounded-lg transition-all duration-300 ${
                  thought.status === 'rejected' ? 'border-red-500/30 bg-red-600/5' :
                  thought.status === 'completed' ? 'border-green-500/30 bg-green-600/5' :
                  'border-white/20 bg-white/5'
                }`}
              >
                {/* Thought Header */}
                <div
                  className="flex items-center justify-between p-4 cursor-pointer hover:bg-white/5"
                  onClick={() => setExpandedThought(isExpanded ? null : thought.id)}
                >
                  <div className="flex items-center gap-3">
                    <div className="text-lg font-semibold text-cyan-400">
                      #{thought.id + 1}
                    </div>
                    {getStatusBadge(thought)}
                    <div className="text-sm text-gray-400">
                      {thought.token_count} tokens • {thought.generation_time.toFixed(1)}s
                    </div>
                  </div>

                  <div className="flex items-center gap-2">
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        copyToClipboard(thought.content, thought.id);
                      }}
                      className="p-1 hover:bg-white/10 rounded"
                    >
                      {copiedId === thought.id ? (
                        <Check className="w-4 h-4 text-green-400" />
                      ) : (
                        <Copy className="w-4 h-4 text-gray-400" />
                      )}
                    </button>
                    {isExpanded ? (
                      <Minimize className="w-4 h-4 text-gray-400" />
                    ) : (
                      <Expand className="w-4 h-4 text-gray-400" />
                    )}
                  </div>
                </div>

                {/* Expanded Content */}
                {isExpanded && (
                  <div className="border-t border-white/10">
                    {/* Thought Content */}
                    <div className="p-4 bg-black/20">
                      <div className="mb-3">
                        <div className="text-sm font-semibold text-gray-300 mb-2">Content:</div>
                        <div className="font-mono text-sm leading-relaxed bg-black/40 p-3 rounded border">
                          {formatThoughtForDisplay(thought.content)}
                        </div>
                      </div>

                      {/* Token Analysis */}
                      {viewMode === 'formatted' && (
                        <div className="mb-4">
                          <div className="text-sm font-semibold text-gray-300 mb-2">Token Analysis:</div>
                          <div className="grid grid-cols-1 md:grid-cols-3 gap-3 text-xs">
                            {Object.entries(thought.special_tokens).map(([key, token]) => (
                              <div key={key} className="bg-black/40 p-2 rounded">
                                <div className="text-gray-400">{key.replace(/_/g, ' ')}:</div>
                                <div className="font-mono text-purple-400">{token}</div>
                              </div>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>

                    {/* Coherence Breakdown */}
                    {thought.status !== 'generating' && (
                      <div className="p-4">
                        <CoherenceScoreBreakdown
                          scores={thought.coherence_scores}
                          threshold={coherenceThreshold}
                          thoughtId={thought.id}
                          isExpanded={true}
                        />
                      </div>
                    )}

                    {/* Additional Stats */}
                    <div className="p-4 bg-black/20 border-t border-white/10">
                      <div className="text-sm font-semibold text-gray-300 mb-2">Generation Stats:</div>
                      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-xs">
                        <div>
                          <span className="text-gray-400">Processing Time:</span>
                          <div className="text-white font-semibold">{thought.generation_time.toFixed(2)}s</div>
                        </div>
                        <div>
                          <span className="text-gray-400">Token Count:</span>
                          <div className="text-white font-semibold">{thought.token_count}</div>
                        </div>
                        <div>
                          <span className="text-gray-400">Avg Tokens/sec:</span>
                          <div className="text-white font-semibold">
                            {thought.generation_time > 0 ? (thought.token_count / thought.generation_time).toFixed(1) : '0'}
                          </div>
                        </div>
                        <div>
                          <span className="text-gray-400">Overall Score:</span>
                          <div className={`font-semibold ${overallScore >= coherenceThreshold ? 'text-green-400' : 'text-red-400'}`}>
                            {overallScore.toFixed(3)}
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            );
          })
        )}
      </div>

      {/* Summary Stats */}
      {thoughts.length > 0 && (
        <div className="mt-6 p-4 bg-gradient-to-r from-cyan-600/20 to-blue-600/20 rounded-lg border border-cyan-500/30">
          <div className="text-sm font-semibold text-cyan-400 mb-2">Inspector Summary</div>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-xs">
            <div>
              <div className="text-gray-400">Valid Thoughts:</div>
              <div className="text-green-400 font-semibold">
                {thoughts.filter(t => t.status === 'completed').length}/{thoughts.length}
              </div>
            </div>
            <div>
              <div className="text-gray-400">Avg Score:</div>
              <div className="text-cyan-400 font-semibold">
                {thoughts.length > 0 ?
                  (thoughts.reduce((acc, t) => acc + getOverallScore(t.coherence_scores), 0) / thoughts.length).toFixed(3) :
                  '0.000'
                }
              </div>
            </div>
            <div>
              <div className="text-gray-400">Total Tokens:</div>
              <div className="text-yellow-400 font-semibold">
                {thoughts.reduce((acc, t) => acc + t.token_count, 0).toLocaleString()}
              </div>
            </div>
            <div>
              <div className="text-gray-400">Search Results:</div>
              <div className="text-pink-400 font-semibold">
                {filteredThoughts.length} shown
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}