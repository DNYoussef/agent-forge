'use client';

import React from 'react';
import { Network, Eye, Database, TrendingUp, Layers, AlertCircle } from 'lucide-react';

interface Phase2IntegrationPanelProps {
  weightData?: any;
  insights?: any;
  connected: boolean;
}

export const Phase2IntegrationPanel: React.FC<Phase2IntegrationPanelProps> = ({
  weightData,
  insights,
  connected
}) => {
  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-gradient-to-r from-green-900 to-blue-900 rounded-lg p-6 border border-green-500">
        <h2 className="text-2xl font-bold mb-2 flex items-center gap-2">
          <Network className="w-6 h-6" />
          Phase 2 Weight Observation Integration
        </h2>
        <p className="text-gray-300">
          Leveraging existing weight space observation to inform expert vector discovery
        </p>
      </div>

      {/* Connection Status */}
      <div className={`rounded-lg p-4 border ${
        connected
          ? 'bg-green-900/20 border-green-700'
          : 'bg-red-900/20 border-red-700'
      }`}>
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className={`w-3 h-3 rounded-full ${
              connected ? 'bg-green-500' : 'bg-red-500'
            }`} />
            <span className="font-semibold">
              Phase 2 Integration Status
            </span>
          </div>
          <span className={connected ? 'text-green-400' : 'text-red-400'}>
            {connected ? 'Connected & Active' : 'Disconnected'}
          </span>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-6">
        {/* Weight Insights */}
        <div className="bg-gray-900 border border-gray-700 rounded-lg p-6">
          <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
            <Eye className="w-5 h-5 text-blue-400" />
            Weight Space Insights
          </h3>

          {insights ? (
            <div className="space-y-4">
              {/* Weight Distribution */}
              <div>
                <div className="text-sm text-gray-400 mb-2">Weight Distribution</div>
                <div className="grid grid-cols-2 gap-2 text-sm">
                  <div className="bg-gray-800 rounded p-2">
                    <span className="text-gray-400">Mean:</span>
                    <span className="ml-2 font-mono">
                      {insights.weight_distribution?.mean_magnitude?.toFixed(4) || 'N/A'}
                    </span>
                  </div>
                  <div className="bg-gray-800 rounded p-2">
                    <span className="text-gray-400">Std:</span>
                    <span className="ml-2 font-mono">
                      {insights.weight_distribution?.std_magnitude?.toFixed(4) || 'N/A'}
                    </span>
                  </div>
                  <div className="bg-gray-800 rounded p-2">
                    <span className="text-gray-400">Max:</span>
                    <span className="ml-2 font-mono">
                      {insights.weight_distribution?.max_magnitude?.toFixed(4) || 'N/A'}
                    </span>
                  </div>
                  <div className="bg-gray-800 rounded p-2">
                    <span className="text-gray-400">Sparsity:</span>
                    <span className="ml-2 font-mono">
                      {(insights.weight_distribution?.sparsity_estimate * 100)?.toFixed(1) || '0'}%
                    </span>
                  </div>
                </div>
              </div>

              {/* Critical Layers */}
              <div>
                <div className="text-sm text-gray-400 mb-2">Critical Layers Identified</div>
                <div className="space-y-1">
                  {insights.critical_layers?.map((layer: string, idx: number) => (
                    <div key={idx} className="bg-gray-800 rounded px-3 py-1 text-sm font-mono">
                      {layer}
                    </div>
                  )) || <span className="text-gray-500 text-sm">No critical layers identified</span>}
                </div>
              </div>

              {/* SVD Focus Areas */}
              <div>
                <div className="text-sm text-gray-400 mb-2">Suggested SVD Focus</div>
                <div className="flex flex-wrap gap-2">
                  {insights.suggested_svd_focus?.map((area: string, idx: number) => (
                    <span key={idx} className="bg-purple-900/30 border border-purple-700 rounded px-2 py-1 text-xs">
                      {area}
                    </span>
                  )) || <span className="text-gray-500 text-sm">No focus areas suggested</span>}
                </div>
              </div>
            </div>
          ) : (
            <div className="flex items-center justify-center h-32 text-gray-500">
              <AlertCircle className="w-5 h-5 mr-2" />
              No insights available
            </div>
          )}
        </div>

        {/* Weight Data Visualization */}
        <div className="bg-gray-900 border border-gray-700 rounded-lg p-6">
          <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
            <Database className="w-5 h-5 text-green-400" />
            3D Weight Data
          </h3>

          {weightData ? (
            <div className="space-y-4">
              {/* Data Points */}
              <div className="bg-gray-800 rounded-lg p-4">
                <div className="flex justify-between items-center mb-2">
                  <span className="text-sm text-gray-400">Total Data Points</span>
                  <span className="text-xl font-bold text-green-400">
                    {weightData.length || 0}
                  </span>
                </div>
                <div className="w-full bg-gray-700 rounded-full h-2">
                  <div
                    className="bg-green-500 h-2 rounded-full transition-all"
                    style={{ width: `${Math.min(100, (weightData.length || 0) / 100)}%` }}
                  />
                </div>
              </div>

              {/* Layer Coverage */}
              <div>
                <div className="text-sm text-gray-400 mb-2">Layer Coverage</div>
                <div className="grid grid-cols-2 gap-2">
                  {Array.from({ length: 6 }, (_, i) => (
                    <div key={i} className="bg-gray-800 rounded p-2 flex items-center justify-between">
                      <span className="text-xs">Layer {i + 1}</span>
                      <div className="w-8 h-8 rounded-full bg-gradient-to-br from-blue-500 to-purple-500 opacity-50" />
                    </div>
                  ))}
                </div>
              </div>
            </div>
          ) : (
            <div className="flex items-center justify-center h-32 text-gray-500">
              <AlertCircle className="w-5 h-5 mr-2" />
              No weight data available
            </div>
          )}
        </div>
      </div>

      {/* Integration Metrics */}
      <div className="bg-gray-900 border border-gray-700 rounded-lg p-6">
        <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
          <TrendingUp className="w-5 h-5 text-yellow-400" />
          Integration Metrics
        </h3>

        <div className="grid grid-cols-4 gap-4">
          {[
            {
              label: 'Weight Patterns',
              value: insights?.weight_distribution ? 'Analyzed' : 'Pending',
              color: insights?.weight_distribution ? 'text-green-400' : 'text-gray-500'
            },
            {
              label: 'Critical Layers',
              value: insights?.critical_layers?.length || 0,
              color: 'text-blue-400'
            },
            {
              label: 'Magnitude Patterns',
              value: insights?.weight_distribution?.magnitude_range ? 'Found' : 'None',
              color: insights?.weight_distribution?.magnitude_range ? 'text-purple-400' : 'text-gray-500'
            },
            {
              label: 'Integration Score',
              value: connected ? '100%' : '0%',
              color: connected ? 'text-green-400' : 'text-red-400'
            }
          ].map((metric, idx) => (
            <div key={idx} className="text-center">
              <div className="text-sm text-gray-400">{metric.label}</div>
              <div className={`text-2xl font-bold ${metric.color}`}>{metric.value}</div>
            </div>
          ))}
        </div>
      </div>

      {/* How Phase 2 Informs Discovery */}
      <div className="bg-gray-900 border border-gray-700 rounded-lg p-6">
        <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
          <Layers className="w-5 h-5 text-purple-400" />
          How Phase 2 Informs Expert Discovery
        </h3>

        <div className="grid grid-cols-3 gap-4 text-sm">
          <div className="bg-gray-800 rounded-lg p-3">
            <div className="font-semibold text-blue-400 mb-2">1. Weight Analysis</div>
            <p className="text-gray-300">
              Phase 2 identifies weight magnitude patterns and distributions to guide where expert vectors should focus.
            </p>
          </div>
          <div className="bg-gray-800 rounded-lg p-3">
            <div className="font-semibold text-green-400 mb-2">2. Layer Priority</div>
            <p className="text-gray-300">
              Critical layers with high variance become priority targets for SVD decomposition and expert adaptation.
            </p>
          </div>
          <div className="bg-gray-800 rounded-lg p-3">
            <div className="font-semibold text-purple-400 mb-2">3. Search Space</div>
            <p className="text-gray-300">
              Weight insights narrow the configuration search space, making ADAS discovery more efficient and targeted.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};