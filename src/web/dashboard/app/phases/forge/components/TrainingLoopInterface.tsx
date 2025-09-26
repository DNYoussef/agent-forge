'use client';

import { useState, useEffect } from 'react';
import { Zap, HelpCircle, X, CheckCircle, AlertCircle, RefreshCw, Target, Lightbulb } from 'lucide-react';

interface TrainingLoopInterfaceProps {
  config: any;
  setConfig: (config: any) => void;
  metrics: any;
  setMetrics: (metrics: any) => void;
}

interface Question {
  id: string;
  content: string;
  difficulty: number;
  hints: string[];
  variations: string[];
  strikes: number;
  status: 'pending' | 'attempting' | 'failed' | 'completed';
}

export default function TrainingLoopInterface({ config, setConfig, metrics, setMetrics }: TrainingLoopInterfaceProps) {
  const [currentQuestions, setCurrentQuestions] = useState<Question[]>([]);
  const [selectedQuestion, setSelectedQuestion] = useState<Question | null>(null);
  const [isTraining, setIsTraining] = useState(false);
  const [showHints, setShowHints] = useState(false);

  // Generate sample questions for demonstration
  useEffect(() => {
    if (currentQuestions.length === 0) {
      const sampleQuestions: Question[] = [
        {
          id: '1',
          content: 'Implement a self-attention mechanism with temperature scaling',
          difficulty: 7.2,
          hints: [
            'Start with the basic dot-product attention formula',
            'Apply temperature scaling before the softmax operation',
            'Remember to normalize by sqrt(d_k) for stability'
          ],
          variations: [
            'Multi-head attention variant',
            'Causal attention mask version',
            'Relative positional encoding addition'
          ],
          strikes: 0,
          status: 'pending'
        },
        {
          id: '2',
          content: 'Optimize gradient flow through residual connections',
          difficulty: 8.5,
          hints: [
            'Consider the gradient highways in ResNet architecture',
            'Apply layer normalization at strategic positions'
          ],
          variations: [
            'Pre-activation residuals',
            'Dense connectivity pattern'
          ],
          strikes: 1,
          status: 'attempting'
        }
      ];
      setCurrentQuestions(sampleQuestions);
      setSelectedQuestion(sampleQuestions[0]);
    }
  }, [currentQuestions.length]);

  // Simulate training progress
  useEffect(() => {
    if (isTraining) {
      const interval = setInterval(() => {
        setMetrics((prev: any) => ({
          ...prev,
          questionsAttempted: prev.questionsAttempted + Math.floor(Math.random() * 2),
          questionsCompleted: Math.min(prev.questionsCompleted + (Math.random() > 0.7 ? 1 : 0), prev.questionsAttempted),
          hintsAccumulated: prev.hintsAccumulated + (Math.random() > 0.8 ? 1 : 0),
          questionVariationsCreated: prev.questionVariationsCreated + (Math.random() > 0.9 ? 1 : 0)
        }));
      }, 3000);

      return () => clearInterval(interval);
    }
  }, [isTraining, setMetrics]);

  const handleQuestionAttempt = (questionId: string, success: boolean) => {
    setCurrentQuestions(prev => prev.map(q => {
      if (q.id === questionId) {
        if (success) {
          return { ...q, status: 'completed' as const };
        } else {
          const newStrikes = q.strikes + 1;
          return {
            ...q,
            strikes: newStrikes,
            status: newStrikes >= 3 ? 'failed' as const : 'attempting' as const
          };
        }
      }
      return q;
    }));

    if (!success && selectedQuestion?.id === questionId) {
      const question = currentQuestions.find(q => q.id === questionId);
      if (question && question.strikes + 1 < 3) {
        setShowHints(true);
      }
    }
  };

  const renderQuestionConfiguration = () => (
    <div className="bg-white/5 backdrop-blur-lg rounded-xl p-4 border border-white/10 mb-4">
      <h3 className="text-lg font-bold mb-3 flex items-center gap-2">
        <Zap className="w-5 h-5 text-yellow-400" />
        Training Loop Configuration
      </h3>

      <div className="grid grid-cols-2 gap-4">
        <div>
          <label className="text-sm text-gray-400 block mb-1">Questions per Level</label>
          <input
            type="number"
            min="10"
            max="500"
            value={config.questionsPerLevel}
            onChange={(e) => setConfig({ ...config, questionsPerLevel: parseInt(e.target.value) })}
            className="w-full bg-white/10 border border-white/20 rounded px-3 py-2 text-white"
          />
        </div>

        <div>
          <label className="text-sm text-gray-400 block mb-1">Max Hints per Question</label>
          <input
            type="number"
            min="1"
            max="10"
            value={config.maxHintsPerQuestion}
            onChange={(e) => setConfig({ ...config, maxHintsPerQuestion: parseInt(e.target.value) })}
            className="w-full bg-white/10 border border-white/20 rounded px-3 py-2 text-white"
          />
        </div>

        <div>
          <label className="text-sm text-gray-400 block mb-1">Question Variations</label>
          <input
            type="number"
            min="1"
            max="10"
            value={config.questionVariations}
            onChange={(e) => setConfig({ ...config, questionVariations: parseInt(e.target.value) })}
            className="w-full bg-white/10 border border-white/20 rounded px-3 py-2 text-white"
          />
        </div>

        <div className="flex items-center gap-2">
          <input
            type="checkbox"
            checked={config.strikeSystemEnabled}
            onChange={(e) => setConfig({ ...config, strikeSystemEnabled: e.target.checked })}
            className="w-4 h-4"
          />
          <label className="text-sm text-gray-400">3-Strike System Enabled</label>
        </div>
      </div>
    </div>
  );

  const renderQuestionQueue = () => (
    <div className="bg-white/5 backdrop-blur-lg rounded-xl p-4 border border-white/10 mb-4">
      <h3 className="text-lg font-bold mb-3 flex items-center gap-2">
        <Target className="w-5 h-5 text-green-400" />
        Active Question Queue
      </h3>

      <div className="space-y-2">
        {currentQuestions.map((question) => (
          <div
            key={question.id}
            onClick={() => setSelectedQuestion(question)}
            className={`
              p-3 rounded-lg border cursor-pointer transition-all
              ${selectedQuestion?.id === question.id
                ? 'border-orange-400 bg-orange-400/10'
                : 'border-white/20 bg-white/5 hover:bg-white/10'
              }
            `}
          >
            <div className="flex justify-between items-start mb-2">
              <div className="flex items-center gap-2">
                <div className={`w-3 h-3 rounded-full ${
                  question.status === 'completed' ? 'bg-green-400' :
                  question.status === 'failed' ? 'bg-red-400' :
                  question.status === 'attempting' ? 'bg-yellow-400' :
                  'bg-gray-400'
                }`} />
                <span className="text-sm font-medium">Question {question.id}</span>
                <span className="text-xs text-gray-400">Difficulty: {question.difficulty.toFixed(1)}</span>
              </div>

              {config.strikeSystemEnabled && (
                <div className="flex gap-1">
                  {[1, 2, 3].map(strike => (
                    <div key={strike} className={`w-2 h-2 rounded-full ${
                      strike <= question.strikes ? 'bg-red-400' : 'bg-gray-600'
                    }`} />
                  ))}
                </div>
              )}
            </div>

            <div className="text-xs text-gray-300 truncate">
              {question.content}
            </div>

            <div className="flex justify-between items-center mt-2 text-xs text-gray-500">
              <span>{question.hints.length} hints available</span>
              <span>{question.variations.length} variations</span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );

  const renderQuestionDetails = () => {
    if (!selectedQuestion) return null;

    return (
      <div className="bg-white/5 backdrop-blur-lg rounded-xl p-4 border border-white/10 mb-4">
        <h3 className="text-lg font-bold mb-3 flex items-center gap-2">
          <HelpCircle className="w-5 h-5 text-blue-400" />
          Current Question
        </h3>

        <div className="bg-black/20 rounded-lg p-4 mb-4">
          <div className="text-lg text-white mb-2">{selectedQuestion.content}</div>
          <div className="flex gap-4 text-sm text-gray-400">
            <span>Difficulty: {selectedQuestion.difficulty.toFixed(1)}</span>
            <span>Strikes: {selectedQuestion.strikes}/3</span>
            <span>Status: {selectedQuestion.status}</span>
          </div>
        </div>

        {(showHints || selectedQuestion.strikes > 0) && (
          <div className="bg-yellow-400/10 border border-yellow-400/30 rounded-lg p-3 mb-4">
            <div className="flex items-center gap-2 mb-2">
              <Lightbulb className="w-4 h-4 text-yellow-400" />
              <span className="text-sm font-medium text-yellow-400">Available Hints</span>
            </div>
            {selectedQuestion.hints.slice(0, selectedQuestion.strikes + (showHints ? selectedQuestion.hints.length : 0)).map((hint, index) => (
              <div key={index} className="text-sm text-gray-300 mb-1">
                {index + 1}. {hint}
              </div>
            ))}
          </div>
        )}

        <div className="flex gap-2">
          <button
            onClick={() => handleQuestionAttempt(selectedQuestion.id, true)}
            className="px-4 py-2 bg-green-600 hover:bg-green-700 text-white rounded-lg flex items-center gap-2 transition-all"
          >
            <CheckCircle className="w-4 h-4" />
            Mark Completed
          </button>
          <button
            onClick={() => handleQuestionAttempt(selectedQuestion.id, false)}
            className="px-4 py-2 bg-red-600 hover:bg-red-700 text-white rounded-lg flex items-center gap-2 transition-all"
          >
            <X className="w-4 h-4" />
            Mark Failed
          </button>
          <button
            onClick={() => setShowHints(!showHints)}
            className="px-4 py-2 bg-yellow-600 hover:bg-yellow-700 text-white rounded-lg flex items-center gap-2 transition-all"
          >
            <HelpCircle className="w-4 h-4" />
            {showHints ? 'Hide' : 'Show'} Hints
          </button>
        </div>
      </div>
    );
  };

  const renderTrainingMetrics = () => (
    <div className="bg-white/5 backdrop-blur-lg rounded-xl p-4 border border-white/10">
      <h3 className="text-lg font-bold mb-3 flex items-center gap-2">
        <RefreshCw className="w-5 h-5 text-purple-400" />
        Training Loop Metrics
      </h3>

      <div className="grid grid-cols-2 gap-4">
        <div className="bg-black/20 rounded-lg p-3">
          <div className="text-xs text-gray-400 mb-1">Questions Attempted</div>
          <div className="text-2xl font-bold text-blue-400">{metrics.questionsAttempted}</div>
          <div className="text-xs text-gray-500">This session</div>
        </div>

        <div className="bg-black/20 rounded-lg p-3">
          <div className="text-xs text-gray-400 mb-1">Success Rate</div>
          <div className="text-2xl font-bold text-green-400">
            {metrics.questionsAttempted > 0 ? ((metrics.questionsCompleted / metrics.questionsAttempted) * 100).toFixed(1) : '0.0'}%
          </div>
          <div className="text-xs text-gray-500">Completed/Attempted</div>
        </div>

        <div className="bg-black/20 rounded-lg p-3">
          <div className="text-xs text-gray-400 mb-1">Hints Used</div>
          <div className="text-2xl font-bold text-yellow-400">{metrics.hintsAccumulated}</div>
          <div className="text-xs text-gray-500">Cumulative learning aids</div>
        </div>

        <div className="bg-black/20 rounded-lg p-3">
          <div className="text-xs text-gray-400 mb-1">Variations Created</div>
          <div className="text-2xl font-bold text-purple-400">{metrics.questionVariationsCreated}</div>
          <div className="text-xs text-gray-500">Adaptive complexity</div>
        </div>
      </div>

      <button
        onClick={() => setIsTraining(!isTraining)}
        className={`
          w-full mt-4 px-4 py-3 rounded-lg font-semibold transition-all flex items-center justify-center gap-2
          ${isTraining
            ? 'bg-red-600 hover:bg-red-700 text-white'
            : 'bg-green-600 hover:bg-green-700 text-white'
          }
        `}
      >
        <Zap className="w-5 h-5" />
        {isTraining ? 'Stop Training Loop' : 'Start Training Loop'}
      </button>
    </div>
  );

  return (
    <div className="space-y-4">
      {renderQuestionConfiguration()}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <div>
          {renderQuestionQueue()}
        </div>
        <div>
          {renderQuestionDetails()}
        </div>
      </div>

      {renderTrainingMetrics()}
    </div>
  );
}