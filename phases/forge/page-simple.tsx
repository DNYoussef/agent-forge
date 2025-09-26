'use client';

import { useState } from 'react';
import { ArrowLeft, Flame } from 'lucide-react';
import Link from 'next/link';
import WeightSpaceSphere from './components/WeightSpaceSphere';

export default function SimplifiedPhase5Page() {
  const [config] = useState({
    currentLevel: 5,
    grokfastEnabled: true,
    grokfastLambda: 0.05,
    sleepMode: false,
    metrics: {
      grokfast_active: true,
      grokfast_lambda: 0.05
    }
  });

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-orange-950 to-slate-950 text-white p-8">
      <Link href="/" className="flex items-center gap-2 text-orange-400 hover:text-orange-300 mb-8">
        <ArrowLeft className="w-5 h-5" />
        Back to Dashboard
      </Link>

      <div className="mb-8">
        <h1 className="text-5xl font-bold mb-4 bg-gradient-to-r from-orange-400 to-red-400 bg-clip-text text-transparent flex items-center gap-4">
          <Flame className="w-12 h-12 text-orange-400" />
          Phase 5: BitLinear Weight Space Sphere
        </h1>
        <p className="text-xl text-gray-400">
          3D visualization of ternary weights (-1, 0, 1) with rippling effects
        </p>
      </div>

      <div className="max-w-6xl mx-auto">
        <WeightSpaceSphere config={config} currentLevel={config.currentLevel} />
      </div>
    </div>
  );
}