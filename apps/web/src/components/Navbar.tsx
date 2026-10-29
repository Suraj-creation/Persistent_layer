import React from 'react';
import { BookOpen, FlaskConical, Activity, ArrowRight, Sparkles } from 'lucide-react';

export type AppView = 'chat' | 'lab' | 'researcher';

interface NavbarProps {
  currentView: AppView;
  onViewChange: (view: AppView) => void;
  activeLearner: string;
  onLearnerChange?: (id: string) => void;
}

export const Navbar: React.FC<NavbarProps> = ({
  currentView,
  onViewChange,
  activeLearner,
  onLearnerChange,
}) => {
  return (
    <header className="bg-white border-b border-slate-200 sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          {/* Logo & Platform Name */}
          <div
            className="flex items-center space-x-3 cursor-pointer"
            onClick={() => onViewChange('chat')}
          >
            <div className="bg-indigo-600 text-white p-2 rounded-xl shadow-md">
              <Activity className="w-5 h-5" />
            </div>
            <div>
              <span className="font-bold text-slate-900 tracking-tight text-lg">
                LearnerState<span className="text-indigo-600">.AI</span>
              </span>
              <span className="hidden md:inline-block ml-3 px-2 py-0.5 text-xs font-medium bg-indigo-50 text-indigo-700 rounded-full border border-indigo-200">
                v2.0 Azure gpt-4.1-mini
              </span>
            </div>
          </div>

          {/* Central Loop Pipeline Indicator (PRD core pipeline) */}
          <div className="hidden xl:flex items-center space-x-1.5 text-xs text-slate-500 font-mono bg-slate-50 px-3 py-1.5 rounded-full border border-slate-200">
            <span className="text-indigo-600 font-semibold">Language</span>
            <ArrowRight className="w-3 h-3 text-slate-400" />
            <span className="text-blue-600 font-semibold">Evidence</span>
            <ArrowRight className="w-3 h-3 text-slate-400" />
            <span className="text-amber-600 font-semibold">State</span>
            <ArrowRight className="w-3 h-3 text-slate-400" />
            <span className="text-emerald-600 font-semibold">Diagnosis</span>
            <ArrowRight className="w-3 h-3 text-slate-400" />
            <span className="text-purple-600 font-semibold">Update</span>
          </div>

          {/* Persona / Interface Switcher */}
          <div className="flex items-center space-x-3">
            <div className="bg-slate-100 p-1 rounded-xl flex items-center space-x-1 border border-slate-200">
              <button
                onClick={() => onViewChange('chat')}
                className={`flex items-center space-x-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold transition-all ${
                  currentView === 'chat'
                    ? 'bg-white text-indigo-600 shadow-sm'
                    : 'text-slate-600 hover:text-slate-900'
                }`}
                title="ChatGPT-inspired interactive diagnostic AI tutor"
              >
                <Sparkles className="w-3.5 h-3.5 text-indigo-600" />
                <span>AI Tutor</span>
              </button>
              <button
                onClick={() => onViewChange('lab')}
                className={`flex items-center space-x-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold transition-all ${
                  currentView === 'lab'
                    ? 'bg-white text-indigo-600 shadow-sm'
                    : 'text-slate-600 hover:text-slate-900'
                }`}
                title="Structured test assessment laboratory"
              >
                <BookOpen className="w-3.5 h-3.5" />
                <span>Lab</span>
              </button>
              <button
                onClick={() => onViewChange('researcher')}
                className={`flex items-center space-x-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold transition-all ${
                  currentView === 'researcher'
                    ? 'bg-white text-indigo-600 shadow-sm'
                    : 'text-slate-600 hover:text-slate-900'
                }`}
                title="Evidence audit, benchmarks & Bayesian trajectory explorer"
              >
                <FlaskConical className="w-3.5 h-3.5" />
                <span>Researcher</span>
              </button>
            </div>

            <div className="hidden sm:flex items-center space-x-1.5 pl-2 border-l border-slate-200 text-xs text-slate-500">
              <span>Learner:</span>
              <select
                value={activeLearner}
                onChange={(e) => onLearnerChange?.(e.target.value)}
                className="font-mono font-semibold text-xs bg-slate-100 hover:bg-slate-200 text-slate-800 border border-slate-200 rounded-lg px-2 py-1 outline-none cursor-pointer transition"
              >
                <option value="student_A">student_A</option>
                <option value="student_B">student_B</option>
                <option value="student_C">student_C</option>
              </select>
            </div>
          </div>
        </div>
      </div>
    </header>
  );
};
