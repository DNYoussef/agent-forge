'use client';

import { ArrowLeft, Wrench, Code, Sparkles } from 'lucide-react';
import Link from 'next/link';

export default function BakingPage() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-green-950 to-slate-950 text-white p-8">
      <Link href="/" className="flex items-center gap-2 text-green-400 hover:text-green-300 mb-8">
        <ArrowLeft className="w-5 h-5" />
        Back to Dashboard
      </Link>

      <div className="mb-8">
        <h1 className="text-5xl font-bold mb-4 bg-gradient-to-r from-green-400 to-emerald-400 bg-clip-text text-transparent flex items-center gap-4">
          <Wrench className="w-12 h-12 text-green-400" />
          Phase 6: Tool & Persona Baking
        </h1>
        <p className="text-xl text-gray-400">
          Integrate tools and define agent personas
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {[
          { icon: Code, title: 'Code Interpreter', desc: 'Execute Python code', color: '#10b981' },
          { icon: Sparkles, title: 'Web Browser', desc: 'Browse and scrape', color: '#059669' },
          { icon: Wrench, title: 'API Tools', desc: 'External integrations', color: '#047857' }
        ].map((tool, i) => (
          <div key={i} className="bg-white/5 backdrop-blur-lg rounded-2xl p-6 border border-white/10">
            <tool.icon className="w-12 h-12 mb-4" style={{ color: tool.color }} />
            <h3 className="text-2xl font-bold mb-2">{tool.title}</h3>
            <p className="text-gray-400">{tool.desc}</p>
          </div>
        ))}
      </div>
    </div>
  );
}
