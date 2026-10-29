import React, { useState } from 'react';
import {
  Brain,
  HelpCircle,
  Sparkles,
  Lightbulb,
  Send,
  RefreshCw,
} from 'lucide-react';
import { processInteraction } from '../api/client';
import { ProcessInteractionResponse, DiagnosticOption } from '../types';

interface StudentPortalProps {
  activeLearner: string;
  onStateUpdated?: () => void;
}

export const StudentPortal: React.FC<StudentPortalProps> = ({ activeLearner, onStateUpdated }) => {
  const [selectedOption, setSelectedOption] = useState<string>('2/5');
  const [explanation, setExplanation] = useState<string>(
    'I added the top numbers and the bottom numbers because we add both parts of a fraction.'
  );
  const [isSubmitting, setIsSubmitting] = useState<boolean>(false);
  const [result, setResult] = useState<ProcessInteractionResponse | null>(null);
  const [diagnosticAnswer, setDiagnosticAnswer] = useState<string | null>(null);

  // Quick Preset Scenarios directly from the PRD
  const applyPreset = (type: 'student_a' | 'student_b' | 'mastery') => {
    if (type === 'student_a') {
      setSelectedOption('2/5');
      setExplanation('I added the top numbers and the bottom numbers because we add both parts of a fraction.');
    } else if (type === 'student_b') {
      setSelectedOption('4/6');
      setExplanation('I know the denominators must match, so I tried to find a common denominator, but I made a calculation mistake while multiplying.');
    } else {
      setSelectedOption('5/6');
      setExplanation('I found a common denominator of 6, converted fractions to 3/6 and 2/6, and added numerators to get 5/6.');
    }
    setResult(null);
    setDiagnosticAnswer(null);
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsSubmitting(true);
    setDiagnosticAnswer(null);

    const isCorrect = selectedOption === '5/6' ? 1 : 0;

    try {
      const resp = await processInteraction({
        learner_id: activeLearner,
        item_id: 'ITEM_FRAC_01',
        item_text: 'Solve: 1/2 + 1/3',
        item_concepts: ['fraction_addition_unlike_denominators'],
        response_text: explanation,
        selected_option: selectedOption,
        correctness: isCorrect,
        latency_seconds: 28.5,
      });
      setResult(resp);
      if (onStateUpdated) onStateUpdated();
    } catch (err: any) {
      alert(`Submission error: ${err.message}`);
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="max-w-6xl mx-auto px-4 py-8 space-y-8">
      {/* Motivating Case Header & Scenario Presets */}
      <div className="bg-gradient-to-r from-indigo-900 to-slate-900 rounded-3xl p-6 md:p-8 text-white shadow-xl">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div>
            <div className="inline-flex items-center space-x-2 text-indigo-300 text-xs font-semibold tracking-wider uppercase mb-2">
              <Sparkles className="w-4 h-4" />
              <span>Interactive Learning Laboratory</span>
            </div>
            <h1 className="text-2xl md:text-3xl font-extrabold tracking-tight">
              Fractions Mastery & Diagnostic Workspace
            </h1>
            <p className="text-slate-300 text-sm mt-1 max-w-2xl">
              Experience the core PRD case study: Student A (systematic misconception) vs. Student B (calculation slip).
            </p>
          </div>

          {/* Quick Presets */}
          <div className="flex flex-wrap items-center gap-2">
            <button
              onClick={() => applyPreset('student_a')}
              className="px-3.5 py-2 text-xs font-semibold bg-rose-500/20 text-rose-200 border border-rose-500/30 rounded-xl hover:bg-rose-500/30 transition-all"
            >
              Preset: Student A (Misconception)
            </button>
            <button
              onClick={() => applyPreset('student_b')}
              className="px-3.5 py-2 text-xs font-semibold bg-amber-500/20 text-amber-200 border border-amber-500/30 rounded-xl hover:bg-amber-500/30 transition-all"
            >
              Preset: Student B (Slip)
            </button>
            <button
              onClick={() => applyPreset('mastery')}
              className="px-3.5 py-2 text-xs font-semibold bg-emerald-500/20 text-emerald-200 border border-emerald-500/30 rounded-xl hover:bg-emerald-500/30 transition-all"
            >
              Preset: Mastery (Common Denom)
            </button>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
        {/* Left Column: Problem Workspace & Explanation Box */}
        <div className="lg:col-span-7 space-y-6">
          <div className="bg-white rounded-2xl border border-slate-200 shadow-sm p-6 md:p-8">
            <div className="flex items-center justify-between pb-4 border-b border-slate-100">
              <span className="text-xs font-bold uppercase tracking-wider text-indigo-600 bg-indigo-50 px-2.5 py-1 rounded-md">
                Problem #01 — Fractions
              </span>
              <span className="text-xs text-slate-400 font-mono">Concept: Unlike Denominators</span>
            </div>

            <div className="my-6">
              <h2 className="text-lg font-semibold text-slate-800 mb-2">Evaluate the following addition:</h2>
              <div className="bg-slate-50 border border-slate-200 rounded-2xl p-6 text-center">
                <span className="text-3xl font-extrabold text-slate-900 tracking-wider">
                  <span className="inline-block px-2">1/2</span>
                  <span className="text-indigo-600 font-bold px-2">+</span>
                  <span className="inline-block px-2">1/3</span>
                  <span className="text-slate-400 font-bold px-2">=</span>
                  <span className="text-indigo-600 font-bold">?</span>
                </span>
              </div>
            </div>

            <form onSubmit={handleSubmit} className="space-y-6">
              {/* Answer Options */}
              <div>
                <label className="block text-xs font-bold uppercase tracking-wider text-slate-700 mb-3">
                  Select your answer:
                </label>
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                  {[
                    { val: '2/5', label: 'A: 2/5', tag: 'Independent add' },
                    { val: '3/5', label: 'B: 3/5', tag: 'Numerator slip' },
                    { val: '4/6', label: 'C: 4/6', tag: 'Calculation slip' },
                    { val: '5/6', label: 'D: 5/6', tag: 'Correct' },
                  ].map((opt) => (
                    <button
                      type="button"
                      key={opt.val}
                      onClick={() => setSelectedOption(opt.val)}
                      className={`p-3 rounded-xl border text-center transition-all ${
                        selectedOption === opt.val
                          ? 'border-indigo-600 bg-indigo-50/50 text-indigo-900 ring-2 ring-indigo-500/20 shadow-sm'
                          : 'border-slate-200 hover:border-slate-300 bg-white text-slate-700'
                      }`}
                    >
                      <div className="font-bold text-base">{opt.val}</div>
                      <div className="text-[10px] text-slate-400 font-medium truncate">{opt.tag}</div>
                    </button>
                  ))}
                </div>
              </div>

              {/* Free-Text Explanation Area (Crux of Language Grounding) */}
              <div>
                <div className="flex items-center justify-between mb-2">
                  <label className="block text-xs font-bold uppercase tracking-wider text-slate-700">
                    Explain Your Reasoning (Language Grounding):
                  </label>
                  <span className="text-[11px] text-indigo-600 font-medium flex items-center">
                    <HelpCircle className="w-3.5 h-3.5 mr-1" />
                    Crucial for Cognitive Evidence
                  </span>
                </div>
                <textarea
                  rows={4}
                  value={explanation}
                  onChange={(e) => setExplanation(e.target.value)}
                  placeholder="Explain how you calculated this or why you chose this answer..."
                  className="w-full text-sm rounded-xl border border-slate-300 p-3.5 focus:border-indigo-600 focus:ring-2 focus:ring-indigo-100 outline-none transition-all placeholder:text-slate-400 bg-slate-50/30"
                />
                <p className="text-xs text-slate-500 mt-1.5">
                  Notice how Student A's component explanation triggers a misconception while Student B's calculation explanation triggers a slip attenuation!
                </p>
              </div>

              <button
                type="submit"
                disabled={isSubmitting}
                className="w-full py-3.5 px-6 bg-indigo-600 hover:bg-indigo-700 text-white font-bold rounded-xl shadow-md shadow-indigo-600/20 transition-all flex items-center justify-center space-x-2 text-sm disabled:opacity-50"
              >
                {isSubmitting ? (
                  <>
                    <RefreshCw className="w-4 h-4 animate-spin" />
                    <span>Extracting Evidence & Updating Bayesian State...</span>
                  </>
                ) : (
                  <>
                    <Send className="w-4 h-4" />
                    <span>Submit & Run Cognitive Intelligence Pipeline</span>
                  </>
                )}
              </button>
            </form>
          </div>
        </div>

        {/* Right Column: Extracted Cognitive Evidence & Active Diagnostic Probing */}
        <div className="lg:col-span-5 space-y-6">
          {!result && (
            <div className="bg-slate-100/60 border border-dashed border-slate-300 rounded-2xl p-8 text-center text-slate-500">
              <Brain className="w-12 h-12 text-slate-300 mx-auto mb-3" />
              <h3 className="text-sm font-bold text-slate-700 mb-1">Cognitive Pipeline Standby</h3>
              <p className="text-xs text-slate-500 max-w-xs mx-auto">
                Submit an explanation on the left to watch the NLP engine extract structured Cognitive Evidence and update the persistent belief state in real-time.
              </p>
            </div>
          )}

          {result && (
            <>
              {/* Extracted Cognitive Evidence Card */}
              <div className="bg-white rounded-2xl border border-slate-200 shadow-sm p-6 space-y-4">
                <div className="flex items-center justify-between pb-3 border-b border-slate-100">
                  <div className="flex items-center space-x-2">
                    <Brain className="w-5 h-5 text-indigo-600" />
                    <h3 className="font-bold text-slate-900 text-sm">Extracted Cognitive Evidence</h3>
                  </div>
                  <span className="text-[11px] font-mono bg-indigo-50 text-indigo-700 px-2 py-0.5 rounded font-semibold">
                    {result.evidence.evidence_id}
                  </span>
                </div>

                {/* Source Quote Span */}
                <div className="bg-slate-50 rounded-xl p-3 border border-slate-200">
                  <div className="text-[10px] font-bold text-slate-400 uppercase tracking-wider mb-1">
                    Verbatim Source Span:
                  </div>
                  <blockquote className="text-xs text-slate-800 font-medium italic border-l-2 border-indigo-500 pl-2.5">
                    "{result.evidence.source_quote}"
                  </blockquote>
                </div>

                {/* Extracted Claim & Polarity */}
                <div>
                  <div className="text-[10px] font-bold text-slate-400 uppercase tracking-wider mb-1">
                    Normalized Claim:
                  </div>
                  <p className="text-xs text-slate-800 font-semibold">{result.evidence.extracted_claim}</p>
                </div>

                {/* Category Badge & Confidence Decomposition */}
                <div className="grid grid-cols-2 gap-3 pt-2">
                  <div className="bg-slate-50 p-2.5 rounded-xl border border-slate-100">
                    <div className="text-[10px] text-slate-400 font-semibold">Cognitive Category:</div>
                    <span
                      className={`inline-block mt-1 px-2.5 py-0.5 rounded-full text-xs font-bold ${
                        result.evidence.candidate_interpretation.category === 'misconception'
                          ? 'bg-rose-100 text-rose-800'
                          : result.evidence.candidate_interpretation.category === 'calculation_slip'
                          ? 'bg-amber-100 text-amber-800'
                          : 'bg-emerald-100 text-emerald-800'
                      }`}
                    >
                      {result.evidence.candidate_interpretation.category.toUpperCase()}
                    </span>
                  </div>

                  <div className="bg-slate-50 p-2.5 rounded-xl border border-slate-100">
                    <div className="text-[10px] text-slate-400 font-semibold">Hypothesis Conf:</div>
                    <div className="mt-1 font-bold text-slate-800 text-sm">
                      {(result.evidence.candidate_interpretation.hypothesis_confidence * 100).toFixed(0)}%
                    </div>
                  </div>
                </div>

                {/* Applied State Rule */}
                {result.events.length > 0 && (
                  <div className="bg-indigo-50/50 rounded-xl p-3 border border-indigo-100 text-xs text-indigo-900">
                    <div className="font-bold flex items-center space-x-1.5 mb-1">
                      <Sparkles className="w-3.5 h-3.5 text-indigo-600" />
                      <span>{result.events[0].applied_rule}</span>
                    </div>
                    <p className="text-[11px] text-slate-600">{result.events[0].change_summary}</p>
                  </div>
                )}
              </div>

              {/* Active Diagnostic Probe (Phase 2 - Diagnose) */}
              {result.recommended_diagnostic_probe && (
                <div className="bg-gradient-to-br from-amber-50 to-orange-50 rounded-2xl border border-amber-200 p-6 space-y-4 shadow-sm">
                  <div className="flex items-center justify-between pb-3 border-b border-amber-200/60">
                    <div className="flex items-center space-x-2 text-amber-900">
                      <Lightbulb className="w-5 h-5 text-amber-600" />
                      <h4 className="font-bold text-sm">Active Diagnostic Probe (EIG Selection)</h4>
                    </div>
                    <span className="text-[10px] bg-amber-200/60 text-amber-900 px-2 py-0.5 rounded font-mono font-semibold">
                      Max Info Gain
                    </span>
                  </div>

                  <p className="text-xs text-amber-900 font-medium">
                    The system detected uncertainty regarding your mental model. Please answer this targeted diagnostic probe to help clarify your reasoning:
                  </p>

                  <div className="bg-white rounded-xl p-4 border border-amber-200 shadow-sm">
                    <p className="font-bold text-sm text-slate-900 mb-3">
                      {result.recommended_diagnostic_probe.question_text}
                    </p>

                    <div className="space-y-2">
                      {result.recommended_diagnostic_probe.options.map((opt: DiagnosticOption) => (
                        <button
                          key={opt.option_key}
                          type="button"
                          onClick={() => setDiagnosticAnswer(opt.option_key)}
                          className={`w-full text-left p-3 rounded-lg text-xs font-medium border transition-all flex items-center justify-between ${
                            diagnosticAnswer === opt.option_key
                              ? 'bg-amber-100 border-amber-500 text-amber-950 font-bold'
                              : 'bg-slate-50 border-slate-200 hover:bg-amber-50/50'
                          }`}
                        >
                          <span>
                            <strong className="mr-1.5">{opt.option_key}:</strong>
                            {opt.option_text}
                          </span>
                          {opt.targeted_misconception_id && (
                            <span className="text-[10px] text-rose-600 font-mono bg-rose-50 px-1.5 py-0.5 rounded">
                              Diagnostic distractor
                            </span>
                          )}
                        </button>
                      ))}
                    </div>
                  </div>

                  {diagnosticAnswer && (
                    <div className="p-3 bg-white rounded-xl border border-amber-300 text-xs text-amber-950">
                      <div className="font-bold text-amber-800 mb-0.5">Probe Answer Recorded</div>
                      <p className="text-[11px] text-slate-600">
                        This response resolves the entropy gap between competing hypotheses and will update the persistent belief state accordingly.
                      </p>
                    </div>
                  )}

                  {result.diagnostic_rationale && (
                    <div className="text-[11px] text-amber-800 font-mono bg-amber-100/50 p-2.5 rounded-lg border border-amber-200">
                      <strong>Selection Rationale:</strong> {result.diagnostic_rationale}
                    </div>
                  )}
                </div>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  );
};
