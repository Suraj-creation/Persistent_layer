import { useState } from 'react';
import { Navbar, AppView } from './components/Navbar';
import { ChatInterface } from './components/ChatInterface';
import { StudentPortal } from './components/StudentPortal';
import { ResearcherStudio } from './components/ResearcherStudio';

export function App() {
  const [currentView, setCurrentView] = useState<AppView>('chat');
  const [activeLearner, setActiveLearner] = useState<string>('student_A');

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900 flex flex-col font-sans">
      <Navbar
        currentView={currentView}
        onViewChange={setCurrentView}
        activeLearner={activeLearner}
        onLearnerChange={setActiveLearner}
      />

      <main className="flex-1 flex flex-col">
        {currentView === 'chat' && (
          <ChatInterface
            activeLearner={activeLearner}
            onSelectLearner={setActiveLearner}
            onOpenResearcher={() => setCurrentView('researcher')}
          />
        )}
        {currentView === 'lab' && (
          <div className="pb-16 pt-6">
            <StudentPortal activeLearner={activeLearner} onStateUpdated={() => {}} />
          </div>
        )}
        {currentView === 'researcher' && (
          <div className="pb-16 pt-6">
            <ResearcherStudio />
          </div>
        )}
      </main>

      {currentView !== 'chat' && (
        <footer className="border-t border-slate-200 bg-white py-6">
          <div className="max-w-7xl mx-auto px-4 flex flex-col sm:flex-row items-center justify-between text-xs text-slate-400 gap-2">
            <span>
              Language-Grounded Learner State Modeling for Misconception-Aware Adaptive Learning &copy; 2026.
            </span>
            <div className="flex items-center space-x-2">
              <span>Simulate Learner:</span>
              <select
                value={activeLearner}
                onChange={(e) => setActiveLearner(e.target.value)}
                className="text-[11px] font-mono bg-slate-100 border border-slate-200 rounded px-2 py-0.5 text-slate-700 outline-none"
              >
                <option value="student_A">student_A (Misconception)</option>
                <option value="student_B">student_B (Slip)</option>
                <option value="student_C">student_C (Mastery)</option>
              </select>
            </div>
          </div>
        </footer>
      )}
    </div>
  );
}

export default App;
