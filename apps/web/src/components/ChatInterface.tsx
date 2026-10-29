import React, { useState, useRef, useEffect } from 'react';
import {
  Sparkles,
  Brain,
  PanelLeftClose,
  PanelLeft,
  Plus,
  ArrowUp,
  MessageSquare,
  Sliders,
  X,
  FileText,
  Copy,
  Check,
} from 'lucide-react';
import { sendChatMessage, getLearnerState, getLearnerAuditReport } from '../api/client';
import {
  ChatMessage,
  CognitiveEvidence,
  LearnerStateSnapshot,
  DiagnosticOption,
  LearnerAuditReport,
} from '../types';

interface ChatInterfaceProps {
  activeLearner: string;
  onSelectLearner: (id: string) => void;
  onOpenResearcher: () => void;
}

export const ChatInterface: React.FC<ChatInterfaceProps> = ({
  activeLearner,
  onSelectLearner,
  onOpenResearcher,
}) => {
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [inspectorOpen, setInspectorOpen] = useState(false);
  const [reportModalOpen, setReportModalOpen] = useState(false);
  const [auditReport, setAuditReport] = useState<LearnerAuditReport | null>(null);
  const [copiedReport, setCopiedReport] = useState(false);

  const [selectedEvidence, setSelectedEvidence] = useState<CognitiveEvidence | null>(null);
  const [activeState, setActiveState] = useState<LearnerStateSnapshot | null>(null);

  // Current math challenge
  const [currentProblem] = useState({
    id: 'ITEM_FRAC_01',
    text: 'Solve: 1/2 + 1/3',
    concept: 'fraction_addition_unlike_denominators',
    conceptName: 'Unlike Denominators Addition',
  });

  const [selectedOption, setSelectedOption] = useState<string | null>(null);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  // Chat message stream
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isLoading]);

  useEffect(() => {
    // Load persisted state for active learner from SQLite
    getLearnerState(activeLearner)
      .then((s) => setActiveState(s))
      .catch(() => {});
  }, [activeLearner]);

  const handleSend = async (customMessage?: string, customOption?: string) => {
    const textToSend = customMessage !== undefined ? customMessage : inputValue;
    if (!textToSend.trim() && !customOption) return;

    const chosenOption = customOption || selectedOption || undefined;

    const userMsg: ChatMessage = {
      id: `usr_${Date.now()}`,
      role: 'user',
      content: textToSend,
      selected_option: chosenOption,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
    };

    setMessages((prev) => [...prev, userMsg]);
    setInputValue('');
    setIsLoading(true);

    try {
      const resp = await sendChatMessage({
        learner_id: activeLearner,
        message: textToSend,
        item_id: currentProblem.id,
        item_text: currentProblem.text,
        selected_option: chosenOption,
        chat_history: messages.map((m) => ({ role: m.role, content: m.content })),
      });

      const assistantMsg: ChatMessage = {
        id: `asst_${Date.now()}`,
        role: 'assistant',
        content: resp.reply,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
        evidence: resp.evidence,
        events: resp.events,
        diagnostic_probe: resp.recommended_diagnostic_probe,
        diagnostic_rationale: resp.diagnostic_rationale,
      };

      setMessages((prev) => [...prev, assistantMsg]);
      setActiveState(resp.new_state);
      setSelectedEvidence(resp.evidence);
    } catch (err: any) {
      const errorMsg: ChatMessage = {
        id: `err_${Date.now()}`,
        role: 'assistant',
        content: `Error connecting to persistent backend: ${err.message}. Please verify the server is running on port 8001.`,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      };
      setMessages((prev) => [...prev, errorMsg]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleDiagnosticAnswer = (opt: DiagnosticOption) => {
    handleSend(
      `For the follow-up, I choose ${opt.option_key}: "${opt.option_text}".`,
      opt.option_key
    );
  };

  const openAuditReport = async () => {
    try {
      const report = await getLearnerAuditReport(activeLearner);
      setAuditReport(report);
      setReportModalOpen(true);
    } catch (err: any) {
      alert(`Could not load audit report: ${err.message}`);
    }
  };

  const copyReportToClipboard = () => {
    if (auditReport) {
      navigator.clipboard.writeText(auditReport.report_markdown);
      setCopiedReport(true);
      setTimeout(() => setCopiedReport(false), 2000);
    }
  };

  const cstate = activeState?.concept_states?.[currentProblem.concept];
  const alphaVal = cstate?.alpha ?? 1.0;
  const betaVal = cstate?.beta ?? 1.0;

  return (
    <div className="flex h-[calc(100vh-4rem)] bg-white text-slate-800 overflow-hidden font-sans">
      {/* 1. Collapsible Left Sidebar */}
      <aside
        className={`bg-slate-900 text-slate-300 flex flex-col transition-all duration-300 ease-in-out border-r border-slate-800 z-30 ${
          sidebarOpen ? 'w-64' : 'w-0 -translate-x-full lg:translate-x-0 lg:w-16'
        }`}
      >
        {/* Sidebar Header */}
        <div className="p-3.5 flex items-center justify-between border-b border-slate-800/80">
          {sidebarOpen ? (
            <div className="flex items-center space-x-2">
              <div className="w-7 h-7 rounded-lg bg-indigo-600 flex items-center justify-center text-white font-bold text-xs shadow-sm">
                LS
              </div>
              <span className="font-bold text-sm tracking-tight text-white">LearnerState</span>
            </div>
          ) : (
            <div className="mx-auto">
              <div className="w-7 h-7 rounded-lg bg-indigo-600 flex items-center justify-center text-white font-bold text-xs">
                LS
              </div>
            </div>
          )}

          <button
            onClick={() => setSidebarOpen(!sidebarOpen)}
            className="p-1.5 rounded-lg text-slate-400 hover:text-white hover:bg-slate-800 transition"
            title="Toggle Sidebar"
          >
            {sidebarOpen ? <PanelLeftClose className="w-4 h-4" /> : <PanelLeft className="w-4 h-4" />}
          </button>
        </div>

        {/* New Session Button */}
        <div className="p-3">
          <button
            onClick={() => {
              setMessages([]);
              setSelectedOption(null);
            }}
            className="w-full flex items-center space-x-2.5 px-3 py-2 bg-slate-800 hover:bg-slate-700 text-white rounded-xl text-xs font-semibold border border-slate-700/60 transition shadow-sm"
          >
            <Plus className="w-4 h-4 text-slate-400" />
            {sidebarOpen && <span>New Session</span>}
          </button>
        </div>

        {/* Persistent Learner Profiles in SQLite */}
        <div className="flex-1 overflow-y-auto px-3 py-2 space-y-1.5 text-xs">
          {sidebarOpen && (
            <div className="text-[10px] font-bold text-slate-500 uppercase tracking-wider px-2 py-1">
              Active Learners (SQLite)
            </div>
          )}

          {[
            { id: 'student_A', name: 'Student A', desc: 'Misconception Model' },
            { id: 'student_B', name: 'Student B', desc: 'Arithmetic Slip Model' },
            { id: 'student_C', name: 'Student C', desc: 'Mastery Model' },
          ].map((item) => (
            <button
              key={item.id}
              onClick={() => {
                onSelectLearner(item.id);
                setMessages([]);
              }}
              className={`w-full text-left px-2.5 py-2.5 rounded-xl flex items-center justify-between transition ${
                activeLearner === item.id
                  ? 'bg-indigo-600/25 text-indigo-200 border border-indigo-500/40 font-semibold'
                  : 'hover:bg-slate-800 text-slate-400 hover:text-slate-200'
              }`}
            >
              <div className="flex items-center space-x-2 truncate">
                <MessageSquare className="w-3.5 h-3.5 flex-shrink-0" />
                {sidebarOpen && (
                  <div className="truncate">
                    <div className="text-xs text-white">{item.name}</div>
                    <div className="text-[10px] text-slate-400">{item.desc}</div>
                  </div>
                )}
              </div>
            </button>
          ))}

          {sidebarOpen && (
            <div className="pt-4 border-t border-slate-800/60">
              <div className="text-[10px] font-bold text-slate-500 uppercase tracking-wider px-2 py-1">
                Curriculum Coordinates
              </div>
              <div className="px-2 py-1 text-xs text-slate-300 font-medium">
                &bull; Unlike Denominator Addition
              </div>
              <div className="px-2 py-1 text-xs text-slate-500 font-medium">
                &bull; Equivalent Fractions
              </div>
              <div className="px-2 py-1 text-xs text-slate-500 font-medium">
                &bull; Common Denominators (LCM)
              </div>
            </div>
          )}
        </div>

        {/* Sidebar Footer: Researcher Studio Switcher */}
        <div className="p-3 border-t border-slate-800/80 bg-slate-900/50">
          <button
            onClick={onOpenResearcher}
            className="w-full flex items-center space-x-2.5 px-3 py-2 bg-indigo-600 hover:bg-indigo-500 text-white rounded-xl text-xs font-bold transition shadow-md"
          >
            <Sliders className="w-4 h-4" />
            {sidebarOpen && <span>Researcher Studio</span>}
          </button>
        </div>
      </aside>

      {/* 2. Main Chat Canvas */}
      <main className="flex-1 flex flex-col h-full bg-white relative overflow-hidden">
        {/* Top Header Bar */}
        <header className="h-14 border-b border-slate-100 flex items-center justify-between px-4 sm:px-6 z-10 bg-white/80 backdrop-blur-md">
          <div className="flex items-center space-x-3">
            {!sidebarOpen && (
              <button
                onClick={() => setSidebarOpen(true)}
                className="p-1.5 rounded-lg text-slate-500 hover:bg-slate-100 transition"
              >
                <PanelLeft className="w-4 h-4" />
              </button>
            )}

            <div className="flex items-center space-x-2">
              <span className="font-semibold text-sm text-slate-900">
                Tutor Session: <span className="font-mono text-indigo-600">{activeLearner}</span>
              </span>
              <span className="hidden sm:inline-flex px-2.5 py-0.5 rounded-full text-[11px] font-medium bg-slate-100 text-slate-700 border border-slate-200">
                SQLite Persistent State
              </span>
            </div>
          </div>

          <div className="flex items-center space-x-2">
            {/* Deterministic Audit Report Modal Button */}
            <button
              onClick={openAuditReport}
              className="flex items-center space-x-1.5 px-3 py-1.5 rounded-xl text-xs font-semibold bg-white border border-slate-200 text-slate-700 hover:bg-slate-50 hover:border-slate-300 transition shadow-sm"
              title="View deterministic mathematical state computation report"
            >
              <FileText className="w-3.5 h-3.5 text-indigo-600" />
              <span className="hidden sm:inline">State Audit Report</span>
            </button>

            {/* Cognitive Inspector Drawer Toggle */}
            <button
              onClick={() => setInspectorOpen(!inspectorOpen)}
              className={`flex items-center space-x-1.5 px-3 py-1.5 rounded-xl text-xs font-semibold border transition ${
                inspectorOpen
                  ? 'bg-indigo-50 border-indigo-200 text-indigo-700'
                  : 'bg-white border-slate-200 text-slate-700 hover:bg-slate-50'
              }`}
            >
              <Brain className="w-3.5 h-3.5 text-indigo-600" />
              <span className="hidden md:inline">Cognitive Inspector</span>
            </button>
          </div>
        </header>

        {/* Conversation Stream */}
        <div className="flex-1 overflow-y-auto px-4 sm:px-6 py-6 space-y-6">
          <div className="max-w-3xl mx-auto space-y-6">
            {/* If no messages: Clean Active Challenge Card */}
            {messages.length === 0 && (
              <div className="py-8 text-center space-y-6">
                <div className="w-12 h-12 rounded-2xl bg-indigo-600 text-white flex items-center justify-center mx-auto shadow-md">
                  <Sparkles className="w-6 h-6" />
                </div>

                <div className="space-y-1">
                  <h2 className="text-xl font-bold text-slate-900 tracking-tight">
                    Mathematics Diagnostic Dialogue
                  </h2>
                  <p className="text-slate-500 text-xs max-w-md mx-auto">
                    Explain your reasoning naturally. Every response is grounded into the persistent cognitive state in SQLite.
                  </p>
                </div>

                {/* Embedded Math Problem Card */}
                <div className="bg-slate-50 border border-slate-200 rounded-2xl p-5 max-w-md mx-auto text-center shadow-sm">
                  <span className="text-[10px] font-bold uppercase tracking-wider text-indigo-600 bg-indigo-50 px-2 py-0.5 rounded">
                    Active Problem
                  </span>
                  <div className="text-2xl font-extrabold text-slate-900 mt-2 font-mono">
                    1/2 + 1/3 = ?
                  </div>

                  <div className="grid grid-cols-4 gap-2 mt-4">
                    {['2/5', '3/5', '4/6', '5/6'].map((opt) => (
                      <button
                        key={opt}
                        onClick={() => setSelectedOption(opt)}
                        className={`py-2 px-3 rounded-xl text-xs font-bold border transition ${
                          selectedOption === opt
                            ? 'bg-indigo-600 text-white border-indigo-600 shadow-sm'
                            : 'bg-white border-slate-200 text-slate-700 hover:border-slate-300'
                        }`}
                      >
                        {opt}
                      </button>
                    ))}
                  </div>
                </div>
              </div>
            )}

            {/* Message Stream */}
            {messages.map((msg) => (
              <div
                key={msg.id}
                className={`flex gap-3.5 ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
              >
                {msg.role === 'assistant' && (
                  <div className="w-8 h-8 rounded-full bg-slate-900 text-white flex items-center justify-center flex-shrink-0 shadow-sm mt-0.5">
                    <Sparkles className="w-4 h-4 text-indigo-400" />
                  </div>
                )}

                <div
                  className={`space-y-2.5 max-w-[85%] sm:max-w-[78%] ${
                    msg.role === 'user'
                      ? 'bg-slate-100 text-slate-900 rounded-2xl rounded-tr-sm px-4 py-3 border border-slate-200/60'
                      : 'text-slate-800'
                  }`}
                >
                  {/* Selected Answer Badge if present */}
                  {msg.selected_option && (
                    <div className="inline-block bg-white/80 border border-slate-200 px-2 py-0.5 rounded text-[10px] font-mono font-bold text-slate-600 mb-1">
                      Chosen: {msg.selected_option}
                    </div>
                  )}

                  {/* Main Message Text */}
                  <div className="text-sm leading-relaxed whitespace-pre-wrap">{msg.content}</div>

                  {/* Subtle Cognitive Evidence & Math Badge */}
                  {msg.evidence && (
                    <div className="pt-1 flex flex-wrap items-center gap-1.5">
                      <button
                        onClick={() => {
                          setSelectedEvidence(msg.evidence!);
                          setInspectorOpen(true);
                        }}
                        className="inline-flex items-center space-x-1 px-2.5 py-1 rounded-lg text-[11px] font-medium bg-slate-100 hover:bg-indigo-50 border border-slate-200 hover:border-indigo-200 transition text-slate-700 hover:text-indigo-700"
                        title="Click to inspect extracted evidence & mathematical state derivation"
                      >
                        <Brain className="w-3.5 h-3.5 text-indigo-600" />
                        <span>
                          Cognitive Evidence: <strong>{msg.evidence.candidate_interpretation.category}</strong> (
                          {(msg.evidence.candidate_interpretation.hypothesis_confidence * 100).toFixed(0)}%)
                        </span>
                      </button>
                    </div>
                  )}

                  {/* Clean contextual option pills if diagnostic probe exists */}
                  {msg.diagnostic_probe && (
                    <div className="pt-2 space-y-1.5">
                      <div className="text-xs text-slate-500 font-medium">Quick response options:</div>
                      <div className="grid grid-cols-1 sm:grid-cols-2 gap-1.5">
                        {msg.diagnostic_probe.options.map((opt) => (
                          <button
                            key={opt.option_key}
                            onClick={() => handleDiagnosticAnswer(opt)}
                            className="text-left px-3 py-2 rounded-xl border border-slate-200 hover:border-indigo-300 bg-white hover:bg-indigo-50/50 transition text-xs text-slate-700 font-medium"
                          >
                            <strong className="text-indigo-600 mr-1">{opt.option_key}:</strong> {opt.option_text}
                          </button>
                        ))}
                      </div>
                    </div>
                  )}
                </div>

                {msg.role === 'user' && (
                  <div className="w-8 h-8 rounded-full bg-indigo-600 text-white flex items-center justify-center flex-shrink-0 shadow-sm mt-0.5 font-bold text-xs">
                    U
                  </div>
                )}
              </div>
            ))}

            {/* Spinner Indicator */}
            {isLoading && (
              <div className="flex gap-3.5 items-center">
                <div className="w-8 h-8 rounded-full bg-slate-900 text-white flex items-center justify-center flex-shrink-0 shadow-sm">
                  <Sparkles className="w-4 h-4 text-indigo-400 animate-spin" />
                </div>
                <div className="text-xs text-slate-400 italic font-medium">
                  Extracting evidence, updating Beta-Bayesian parameters & persisting to SQLite...
                </div>
              </div>
            )}

            <div ref={messagesEndRef} />
          </div>
        </div>

        {/* 3. Floating ChatGPT Composer Bar (Clean, Minimal, No Clutter) */}
        <div className="p-4 sm:p-6 bg-gradient-to-t from-white via-white to-transparent">
          <div className="max-w-3xl mx-auto">
            <form
              onSubmit={(e) => {
                e.preventDefault();
                handleSend();
              }}
              className="bg-slate-50 border border-slate-300 rounded-3xl p-2.5 focus-within:border-slate-500 focus-within:ring-2 focus-within:ring-slate-100 shadow-sm transition flex items-end gap-2"
            >
              <textarea
                rows={1}
                value={inputValue}
                onChange={(e) => setInputValue(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    handleSend();
                  }
                }}
                placeholder="Explain your calculation steps, ask a question, or respond..."
                className="flex-1 bg-transparent border-0 outline-none resize-none text-sm p-1.5 max-h-36 placeholder:text-slate-400 text-slate-900"
              />

              <button
                type="submit"
                disabled={!inputValue.trim() || isLoading}
                className="w-8 h-8 rounded-full bg-slate-900 hover:bg-slate-800 disabled:opacity-30 text-white flex items-center justify-center flex-shrink-0 transition shadow-sm"
              >
                <ArrowUp className="w-4 h-4 stroke-[2.5]" />
              </button>
            </form>
          </div>
        </div>
      </main>

      {/* 4. Cognitive Inspector & Observability Drawer */}
      {inspectorOpen && (
        <aside className="w-80 lg:w-96 border-l border-slate-200 bg-slate-50/70 p-5 overflow-y-auto space-y-6 z-20 shadow-lg">
          <div className="flex items-center justify-between pb-3 border-b border-slate-200">
            <div className="flex items-center space-x-2">
              <Brain className="w-5 h-5 text-indigo-600" />
              <h3 className="font-bold text-sm text-slate-900">Cognitive State Inspector</h3>
            </div>
            <button
              onClick={() => setInspectorOpen(false)}
              className="p-1 rounded-lg text-slate-400 hover:text-slate-600 transition"
            >
              <X className="w-4 h-4" />
            </button>
          </div>

          {/* Mathematical Beta Distribution Belief State */}
          {cstate && (
            <div className="bg-white p-4 rounded-2xl border border-slate-200 shadow-sm space-y-3">
              <div className="flex items-center justify-between text-xs">
                <span className="font-bold text-slate-700">Calculated Mastery E[L]</span>
                <span className="font-mono font-bold text-indigo-600">
                  {(cstate.mastery_estimate * 100).toFixed(1)}%
                </span>
              </div>

              <div className="w-full bg-slate-100 h-2 rounded-full overflow-hidden">
                <div
                  className="bg-indigo-600 h-full rounded-full transition-all duration-500"
                  style={{ width: `${cstate.mastery_estimate * 100}%` }}
                />
              </div>

              {/* Exact Conjugate Beta Parameters */}
              <div className="p-2.5 rounded-xl bg-slate-50 border border-slate-100 font-mono text-[11px] space-y-1">
                <div className="text-[10px] font-bold text-slate-400 uppercase tracking-wider">
                  Conjugate Beta Parameters:
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-500">&alpha; (Positive Evidence):</span>
                  <span className="font-bold text-slate-800">{alphaVal.toFixed(2)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-500">&beta; (Misconception Mass):</span>
                  <span className="font-bold text-slate-800">{betaVal.toFixed(2)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-500">Epistemic Uncertainty (&sigma;):</span>
                  <span className="font-bold text-indigo-700">{(cstate.uncertainty * 100).toFixed(1)}%</span>
                </div>
              </div>

              {/* Candidate Misconception Posteriors */}
              {Object.keys(cstate.candidate_misconceptions).length > 0 && (
                <div className="pt-2 border-t border-slate-100 space-y-2">
                  <div className="text-[10px] font-bold text-rose-700 uppercase tracking-wider">
                    Candidate Misconceptions (Bayesian Odds):
                  </div>
                  {Object.values(cstate.candidate_misconceptions).map((m) => (
                    <div
                      key={m.misconception_id}
                      className="p-2 rounded-xl bg-rose-50 border border-rose-200 text-xs text-rose-950 flex items-center justify-between"
                    >
                      <div>
                        <div className="font-bold text-[11px] truncate max-w-[160px]">{m.misconception_name}</div>
                        <div className="text-[9px] text-slate-500 capitalize">Status: {m.status}</div>
                      </div>
                      <span className="font-mono font-bold text-rose-700">
                        {(m.posterior_probability * 100).toFixed(0)}%
                      </span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}

          {/* Selected Message Evidence Card */}
          {selectedEvidence ? (
            <div className="bg-white p-4 rounded-2xl border border-slate-200 shadow-sm space-y-3 text-xs">
              <div className="text-[10px] font-bold text-slate-400 uppercase tracking-wider">
                Verbatim Evidence: {selectedEvidence.evidence_id}
              </div>

              <div className="bg-slate-50 p-2.5 rounded-xl border border-slate-200 italic text-slate-800 text-[11px]">
                "{selectedEvidence.source_quote}"
              </div>

              <div>
                <span className="text-[10px] text-slate-400 font-semibold block">Extracted Claim:</span>
                <span className="font-medium text-slate-900">{selectedEvidence.extracted_claim}</span>
              </div>

              <div className="grid grid-cols-2 gap-2 text-[10px]">
                <div>
                  <span className="text-slate-400 font-semibold block">Category:</span>
                  <span className="font-bold uppercase text-indigo-700">
                    {selectedEvidence.candidate_interpretation.category}
                  </span>
                </div>
                <div>
                  <span className="text-slate-400 font-semibold block">Confidence:</span>
                  <span className="font-bold text-slate-800">
                    {(selectedEvidence.candidate_interpretation.hypothesis_confidence * 100).toFixed(0)}%
                  </span>
                </div>
              </div>

              <div className="text-[10px] text-slate-400 font-mono pt-1">
                Grounded KC: {selectedEvidence.concept_links[0]?.concept_id || 'unknown'}
              </div>
            </div>
          ) : (
            <div className="text-center p-6 text-xs text-slate-400 italic">
              Click any "Cognitive Evidence" badge in the conversation to inspect its exact verbatim quote span and claim.
            </div>
          )}

          {/* Quick Action: View Full Deterministic Audit Report */}
          <button
            onClick={openAuditReport}
            className="w-full py-2.5 px-3 bg-slate-900 hover:bg-slate-800 text-white rounded-xl text-xs font-semibold flex items-center justify-center space-x-2 transition shadow-sm"
          >
            <FileText className="w-3.5 h-3.5" />
            <span>Generate Mathematical Audit Report</span>
          </button>
        </aside>
      )}

      {/* 5. Deterministic State Audit Report Modal */}
      {reportModalOpen && auditReport && (
        <div className="fixed inset-0 bg-slate-900/60 backdrop-blur-sm z-50 flex items-center justify-center p-4">
          <div className="bg-white rounded-2xl max-w-3xl w-full max-h-[85vh] flex flex-col shadow-2xl border border-slate-200 overflow-hidden">
            <div className="p-4 border-b border-slate-200 flex items-center justify-between bg-slate-50">
              <div className="flex items-center space-x-2">
                <FileText className="w-5 h-5 text-indigo-600" />
                <h3 className="font-bold text-sm text-slate-900">
                  Deterministic Cognitive Audit Report: {auditReport.learner_id}
                </h3>
              </div>
              <div className="flex items-center space-x-2">
                <button
                  onClick={copyReportToClipboard}
                  className="px-2.5 py-1 text-xs font-semibold rounded-lg border border-slate-200 bg-white hover:bg-slate-100 transition flex items-center space-x-1"
                >
                  {copiedReport ? <Check className="w-3.5 h-3.5 text-emerald-600" /> : <Copy className="w-3.5 h-3.5" />}
                  <span>{copiedReport ? 'Copied' : 'Copy Markdown'}</span>
                </button>
                <button
                  onClick={() => setReportModalOpen(false)}
                  className="p-1 rounded-lg text-slate-400 hover:text-slate-600 transition"
                >
                  <X className="w-4 h-4" />
                </button>
              </div>
            </div>

            <div className="p-6 overflow-y-auto font-mono text-xs text-slate-800 space-y-4 whitespace-pre-wrap leading-relaxed select-text bg-white">
              {auditReport.report_markdown}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
