import React, { useEffect, useState } from 'react';
import { AlertTriangle, BadgeCheck, Database, ExternalLink, FlaskConical, Info, RefreshCw } from 'lucide-react';
import { getTrainingResearchSummary } from '../api/client';
import { TrainingResearchSummary } from '../types';

const MODEL_LABELS: Record<string, string> = {
  bkt: 'Non-forgetting BKT',
  constant_train_prevalence: 'Constant training prevalence',
  smoothed_skill_prevalence: 'Smoothed skill prevalence',
  skill_plus_opportunity_logistic: 'Skill + opportunity logistic',
};

const format = (value: number | undefined, digits = 4) =>
  typeof value === 'number' && Number.isFinite(value) ? value.toFixed(digits) : '—';

const MetricCard: React.FC<{ label: string; value: string; note: string }> = ({ label, value, note }) => (
  <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
    <div className="text-[10px] font-bold uppercase tracking-wider text-slate-400">{label}</div>
    <div className="mt-1 text-2xl font-extrabold text-slate-900">{value}</div>
    <div className="mt-1 text-[11px] text-slate-500">{note}</div>
  </div>
);

const ArtifactHash: React.FC<{ label: string; value?: string | null; verified?: boolean }> = ({ label, value, verified }) => (
  <div className="min-w-0 rounded-lg border border-slate-200 bg-slate-50 p-3">
    <div className="flex items-center justify-between gap-3">
      <span className="text-[11px] font-bold text-slate-700">{label}</span>
      <span className={`shrink-0 rounded-full px-2 py-0.5 text-[9px] font-bold ${verified ? 'bg-emerald-100 text-emerald-800' : 'bg-amber-100 text-amber-900'}`}>
        {verified ? 'SHA verified' : 'Not verified'}
      </span>
    </div>
    <code className="mt-2 block break-all text-[9px] leading-relaxed text-slate-500">{value || 'Artifact hash unavailable'}</code>
  </div>
);

export const TrainingResearchPanel: React.FC = () => {
  const [report, setReport] = useState<TrainingResearchSummary | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let active = true;
    getTrainingResearchSummary()
      .then((data) => { if (active) setReport(data); })
      .catch((err: Error) => { if (active) setError(err.message); });
    return () => { active = false; };
  }, []);

  if (error) {
    return (
      <div role="alert" className="rounded-2xl border border-rose-200 bg-rose-50 p-6 text-sm text-rose-900">
        <div className="font-bold">Training research data could not be loaded.</div>
        <div className="mt-1 text-xs">{error}</div>
      </div>
    );
  }

  if (!report) {
    return (
      <div className="rounded-2xl border border-slate-200 bg-white p-12 text-center">
        <RefreshCw className="mx-auto mb-3 h-8 w-8 animate-spin text-indigo-600" />
        <p className="text-sm font-semibold text-slate-700">Loading training and artifact record…</p>
      </div>
    );
  }

  const bkt = report.assistments_bkt;
  const bktSummary = bkt.summary_across_splits.find((model) => model.name === 'bkt')?.metrics;
  const bktAuc = bktSummary?.auc;
  const bktBrier = bktSummary?.brier_score;
  const bktAccuracy = bktSummary?.accuracy_at_0_5;
  const eediCounts = report.eedi_graph.unique_counts;

  return (
    <div className="space-y-6">
      <section className="rounded-2xl border border-indigo-200 bg-gradient-to-br from-indigo-50 via-white to-slate-50 p-6 shadow-sm">
        <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
          <div className="max-w-3xl">
            <div className="flex items-center gap-2 text-indigo-700">
              <FlaskConical className="h-5 w-5" />
              <span className="text-[11px] font-extrabold uppercase tracking-widest">Research record · {report.status}</span>
            </div>
            <h2 className="mt-2 text-xl font-extrabold tracking-tight text-slate-900">Training results, methods, and artifacts</h2>
            <p className="mt-2 text-sm leading-relaxed text-slate-600">{report.purpose}</p>
          </div>
          <div className="flex max-w-md gap-3 rounded-xl border border-amber-300 bg-amber-50 p-4 text-xs leading-relaxed text-amber-950">
            <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-amber-700" />
            <div><strong>Offline research only.</strong> {report.runtime_boundary.notice} Live state currently uses {report.runtime_boundary.live_state_method.toLowerCase()}</div>
          </div>
        </div>
        <div className="mt-5 grid gap-3 sm:grid-cols-3">
          <div className="rounded-xl border border-slate-200 bg-white/80 p-3 text-xs">
            <div className="font-bold text-slate-800">BKT → live learner state</div>
            <div className="mt-1 text-slate-500">{report.runtime_boundary.bkt_used_for_live_learner_state ? 'Connected' : 'Not connected'}</div>
          </div>
          <div className="rounded-xl border border-slate-200 bg-white/80 p-3 text-xs">
            <div className="font-bold text-slate-800">MAP classifier</div>
            <div className="mt-1 text-slate-500">{report.map_classifier.status.replace(/_/g, ' ')}</div>
          </div>
          <div className="rounded-xl border border-slate-200 bg-white/80 p-3 text-xs">
            <div className="font-bold text-slate-800">Eedi graph → runtime retrieval</div>
            <div className="mt-1 text-slate-500">{report.runtime_boundary.eedi_graph_used_for_runtime_retrieval ? 'Connected' : 'Audited; not connected'}</div>
          </div>
        </div>
      </section>

      <section className="space-y-4">
        <div className="flex flex-col gap-1 sm:flex-row sm:items-end sm:justify-between">
          <div>
            <h3 className="text-base font-extrabold text-slate-900">ASSISTments correctness-only knowledge tracing</h3>
            <p className="mt-1 text-xs text-slate-500">{bkt.dataset} · {bkt.model}</p>
          </div>
          {bkt.source_url && <a className="inline-flex items-center gap-1 text-xs font-semibold text-indigo-700 hover:text-indigo-900" href={bkt.source_url} target="_blank" rel="noreferrer">Official dataset page <ExternalLink className="h-3 w-3" /></a>}
        </div>

        <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
          <MetricCard label="Filtered modeling events" value={Number(bkt.data_audit.rows_after_support_filter || 0).toLocaleString()} note="Across 8 supported fraction/percent skills" />
          <MetricCard label="Mean AUC · 2 splits" value={`${format(bktAuc?.mean_across_splits)} ± ${format(bktAuc?.sd_across_splits)}`} note="Student-level holdouts; SD is descriptive" />
          <MetricCard label="Mean accuracy" value={format(bktAccuracy?.mean_across_splits)} note="At a fixed 0.5 probability threshold" />
          <MetricCard label="Mean Brier score" value={format(bktBrier?.mean_across_splits)} note="Lower is better; uncalibrated probabilities" />
        </div>

        <div className="overflow-hidden rounded-2xl border border-slate-200 bg-white shadow-sm">
          <div className="border-b border-slate-100 p-5">
            <h4 className="text-sm font-bold text-slate-900">Repeated split comparison</h4>
            <p className="mt-1 text-[11px] text-slate-500">Two paired student holdouts (seeds 42 and 7). Values are mean ± split SD, not confidence intervals.</p>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full min-w-[700px] text-left text-xs">
              <thead className="bg-slate-50 text-[10px] uppercase tracking-wider text-slate-500">
                <tr><th className="px-5 py-3">Method</th><th className="px-5 py-3">ROC AUC ↑</th><th className="px-5 py-3">Accuracy ↑</th><th className="px-5 py-3">Brier ↓</th><th className="px-5 py-3">10-bin ECE*</th></tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {bkt.summary_across_splits.map((model) => (
                  <tr key={model.name} className={model.name === 'bkt' ? 'bg-indigo-50/60 font-semibold' : ''}>
                    <td className="px-5 py-3 text-slate-800">{MODEL_LABELS[model.name] || model.name}</td>
                    {['auc', 'accuracy_at_0_5', 'brier_score', 'ece_10_bins_uncalibrated'].map((metric) => (
                      <td key={metric} className="px-5 py-3 font-mono text-slate-700">
                        {format(model.metrics[metric]?.mean_across_splits)} ± {format(model.metrics[metric]?.sd_across_splits)}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="border-t border-slate-100 bg-slate-50 px-5 py-3 text-[10px] leading-relaxed text-slate-500">* ECE is uncalibrated and bin-dependent. BKT mean AUC improvement over the skill + opportunity logistic baseline is {format(bkt.paired_bkt_minus_baselines.skill_plus_opportunity_logistic?.auc)}; mean Brier change is {format(bkt.paired_bkt_minus_baselines.skill_plus_opportunity_logistic?.brier_score)} (negative favors BKT). Two splits do not establish statistical significance or transfer to this app's learners.</div>
        </div>

        <div className="grid gap-4 xl:grid-cols-2">
          <div className="overflow-hidden rounded-2xl border border-slate-200 bg-white shadow-sm">
            <div className="border-b border-slate-100 p-5"><h4 className="text-sm font-bold text-slate-900">Split details</h4><p className="mt-1 text-[11px] text-slate-500">Learner groups are held out from fitting; chronology within each held-out learner is preserved.</p></div>
            <div className="overflow-x-auto">
              <table className="w-full min-w-[480px] text-left text-[11px]">
                <thead className="bg-slate-50 text-slate-500"><tr><th className="px-4 py-2.5">Seed</th><th className="px-4 py-2.5">Train learners / events</th><th className="px-4 py-2.5">Held-out learners / events</th><th className="px-4 py-2.5">BKT AUC</th><th className="px-4 py-2.5">BKT Brier</th></tr></thead>
                <tbody className="divide-y divide-slate-100">{bkt.seeds.map((split) => <tr key={split.seed}><td className="px-4 py-3 font-mono font-bold">{split.seed}</td><td className="px-4 py-3">{split.train_students.toLocaleString()} / {split.train_rows.toLocaleString()}</td><td className="px-4 py-3">{split.test_students.toLocaleString()} / {split.test_rows.toLocaleString()}</td><td className="px-4 py-3 font-mono">{format(split.models.bkt?.auc)}</td><td className="px-4 py-3 font-mono">{format(split.models.bkt?.brier_score)}</td></tr>)}</tbody>
              </table>
            </div>
          </div>
          <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
            <h4 className="text-sm font-bold text-slate-900">Data filters and run setup</h4>
            <div className="mt-3 grid grid-cols-2 gap-2 text-[11px]">
              <div className="rounded-lg bg-slate-50 p-3"><span className="block text-slate-400">Rows after support filter</span><strong className="text-slate-800">{Number(bkt.data_audit.rows_after_support_filter || 0).toLocaleString()}</strong></div>
              <div className="rounded-lg bg-slate-50 p-3"><span className="block text-slate-400">Supported skills</span><strong className="text-slate-800">{bkt.data_audit.skills_after_support_filter}</strong></div>
              <div className="rounded-lg bg-slate-50 p-3"><span className="block text-slate-400">Filter</span><strong className="text-slate-800">{bkt.skill_filter}</strong></div>
              <div className="rounded-lg bg-slate-50 p-3"><span className="block text-slate-400">Split</span><strong className="text-slate-800">{Math.round((bkt.test_ratio || 0) * 100)}% learner holdout</strong></div>
            </div>
            <p className="mt-3 text-[11px] leading-relaxed text-slate-600">The run keeps original problems and attempt-first rows (`first_action = 0`), binary outcomes, and events with learner, skill, problem, and order fields. Low-support skills are removed before fitting. The primary seed 42 split has {Number(bkt.primary_split.train_rows || 0).toLocaleString()} training events and {Number(bkt.primary_split.test_rows_evaluated || 0).toLocaleString()} evaluated events.</p>
            <div className="mt-3 flex flex-wrap gap-1.5">{Object.entries(bkt.versions).map(([name, version]) => <span key={name} className="rounded-full bg-slate-100 px-2.5 py-1 font-mono text-[9px] text-slate-600">{name} {version}</span>)}</div>
          </div>
        </div>

        <div className="overflow-hidden rounded-2xl border border-slate-200 bg-white shadow-sm">
          <div className="border-b border-slate-100 p-5"><h4 className="text-sm font-bold text-slate-900">Per-skill held-out results · seed 42</h4><p className="mt-1 text-[11px] text-slate-500">The very small Percents group is shown with its support count so the unstable estimate is visible.</p></div>
          <div className="overflow-x-auto"><table className="w-full min-w-[680px] text-left text-[11px]"><thead className="bg-slate-50 text-slate-500"><tr><th className="px-4 py-2.5">ASSISTments skill</th><th className="px-4 py-2.5">Test events</th><th className="px-4 py-2.5">AUC</th><th className="px-4 py-2.5">Accuracy</th><th className="px-4 py-2.5">Brier</th><th className="px-4 py-2.5">ECE</th></tr></thead><tbody className="divide-y divide-slate-100">{Object.entries(bkt.skill_metrics).map(([skill, values]) => <tr key={skill} className={values.rows < 100 ? 'bg-amber-50/60' : ''}><td className="px-4 py-3 font-medium text-slate-800">{skill}</td><td className="px-4 py-3 font-mono">{values.rows}</td><td className="px-4 py-3 font-mono">{format(values.auc)}</td><td className="px-4 py-3 font-mono">{format(values.accuracy_at_0_5)}</td><td className="px-4 py-3 font-mono">{format(values.brier_score)}</td><td className="px-4 py-3 font-mono">{format(values.ece_10_bins_uncalibrated)}</td></tr>)}</tbody></table></div>
        </div>

        <details className="rounded-2xl border border-slate-200 bg-white shadow-sm">
          <summary className="cursor-pointer px-5 py-4 text-sm font-bold text-slate-900">Inspect fitted parameters by skill and split</summary>
          <div className="overflow-x-auto border-t border-slate-100"><table className="w-full min-w-[620px] text-left text-[10px]"><thead className="bg-slate-50 text-slate-500"><tr><th className="px-4 py-2.5">Skill</th><th className="px-4 py-2.5">Seed 42 estimates</th><th className="px-4 py-2.5">Seed 7 estimates</th></tr></thead><tbody className="divide-y divide-slate-100">{bkt.skill_parameters.map((entry) => <tr key={entry.skill}><td className="px-4 py-3 font-semibold text-slate-800">{entry.skill}</td>{['42', '7'].map((seed) => <td key={seed} className="px-4 py-3 font-mono text-slate-600">{Object.entries(entry.by_seed[seed] || {}).map(([name, value]) => `${name} ${format(value, 3)}`).join(' · ') || 'Not available'}</td>)}</tr>)}</tbody></table></div>
          <p className="border-t border-slate-100 px-5 py-3 text-[10px] text-slate-500">These are fitted aggregate skill parameters, not student-level records. The serialized checkpoints are hashed for integrity and are not deserialized by the website.</p>
        </details>

        <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
          <div className="flex items-center gap-2"><Database className="h-4 w-4 text-indigo-600" /><h4 className="text-sm font-bold text-slate-900">Tracked model artifacts and provenance</h4></div>
          <p className="mt-1 text-[11px] text-slate-500">Raw ASSISTments rows remain local and Git-ignored. Dataset checksum: <code className="break-all">{bkt.data_sha256 || 'not available'}</code></p>
          <p className="mt-1 text-[10px] text-slate-400">Training run: {bkt.trained_at_utc || 'date unavailable'} · robustness evaluation: {bkt.evaluated_at_utc || 'date unavailable'}</p>
          <div className="mt-4 grid gap-3 md:grid-cols-2">{bkt.seeds.flatMap((split) => split.artifacts.map((artifact) => <ArtifactHash key={`${split.seed}-${artifact.path}`} label={`Seed ${split.seed} · ${artifact.name}`} value={artifact.actual_sha256 || artifact.expected_sha256} verified={artifact.checksum_verified} />))}</div>
          {bkt.limitations.length > 0 && <ul className="mt-4 space-y-1 text-[10px] leading-relaxed text-slate-600">{bkt.limitations.map((item) => <li key={item} className="flex gap-2"><Info className="mt-0.5 h-3 w-3 shrink-0 text-slate-400" />{item}</li>)}</ul>}
        </div>
      </section>

      <section className="grid gap-4 xl:grid-cols-2">
        <div className="rounded-2xl border border-emerald-200 bg-white p-5 shadow-sm">
          <div className="flex items-start justify-between gap-3"><div><div className="flex items-center gap-2 text-emerald-800"><BadgeCheck className="h-4 w-4" /><h3 className="text-sm font-extrabold">Eedi Misconceptions Graph v1.0</h3></div><p className="mt-1 text-[11px] text-slate-500">Structural data audit · curriculum grounding resource</p></div><span className="rounded-full bg-emerald-50 px-2.5 py-1 text-[9px] font-bold text-emerald-800">{report.eedi_graph.license}</span></div>
          <div className="mt-4 grid grid-cols-2 gap-2 sm:grid-cols-3">{[['Edges', report.eedi_graph.row_counts['misconception-map.csv']], ['Constructs', eediCounts.constructs], ['Misconceptions', eediCounts.misconceptions], ['Subtopics', eediCounts.subtopics], ['Topics', eediCounts.topics], ['Subjects', eediCounts.subjects]].map(([label, value]) => <div key={label} className="rounded-lg bg-slate-50 p-3"><div className="text-[9px] uppercase tracking-wider text-slate-400">{label}</div><div className="mt-1 text-lg font-extrabold text-slate-800">{Number(value || 0).toLocaleString()}</div></div>)}</div>
          <p className="mt-3 text-[11px] leading-relaxed text-slate-600">{report.eedi_graph.data_role}. Edge weights validated: {report.eedi_graph.weight_validation.all_groups_sum_to_100_within_tolerance ? 'all checked misconception groups sum to 100% within tolerance.' : 'validation unavailable.'} The graph contains no student responses and has not been mapped to the app taxonomy.</p>
          <p className="mt-2 text-[10px] text-slate-500">{report.eedi_graph.attribution} Pinned revision: <code>{report.eedi_graph.revision}</code></p>
          {report.eedi_graph.source_url && <a className="mt-3 inline-flex items-center gap-1 text-[11px] font-semibold text-indigo-700 hover:text-indigo-900" href={report.eedi_graph.source_url} target="_blank" rel="noreferrer">Pinned dataset source <ExternalLink className="h-3 w-3" /></a>}
          <details className="mt-3"><summary className="cursor-pointer text-[10px] font-semibold text-slate-600">Inspect source file checksums</summary><div className="mt-2 space-y-2">{Object.entries(report.eedi_graph.file_sha256).map(([name, hash]) => <div key={name} className="rounded-lg bg-slate-50 p-2"><div className="text-[9px] font-bold text-slate-600">{name}</div><code className="mt-1 block break-all text-[9px] text-slate-500">{hash}</code></div>)}</div></details>
        </div>

        <div className="rounded-2xl border border-amber-200 bg-white p-5 shadow-sm">
          <div className="flex items-center justify-between gap-3"><div><div className="flex items-center gap-2 text-amber-900"><AlertTriangle className="h-4 w-4" /><h3 className="text-sm font-extrabold">MAP misconception classifier</h3></div><p className="mt-1 text-[11px] text-slate-500">{report.map_classifier.status.replace(/_/g, ' ')}</p></div><span className="rounded-full bg-amber-100 px-2.5 py-1 text-[9px] font-bold text-amber-900">No model or score</span></div>
          <p className="mt-4 text-xs leading-relaxed text-slate-700">{report.map_classifier.reason}</p>
          <div className="mt-3 rounded-lg bg-slate-50 p-3 text-[11px] leading-relaxed text-slate-600"><strong>Prepared method:</strong> {report.map_classifier.method}<br /><strong>Planned validation:</strong> {report.map_classifier.planned_validation}</div>
          <div className="mt-3"><div className="text-[10px] font-bold uppercase tracking-wider text-slate-400">Planned metrics</div><div className="mt-2 flex flex-wrap gap-1.5">{report.map_classifier.planned_metrics.map((metric) => <span key={metric} className="rounded-full bg-amber-50 px-2.5 py-1 text-[9px] font-semibold text-amber-900">{metric}</span>)}</div></div>
        </div>
      </section>

      <section className="rounded-2xl border border-slate-200 bg-slate-50 p-5">
        <h3 className="text-sm font-extrabold text-slate-900">Other datasets and existing demonstration</h3>
        <div className="mt-3 grid gap-3 md:grid-cols-2">{report.other_dataset_status.map((dataset) => <div key={dataset.dataset} className="rounded-xl border border-slate-200 bg-white p-4"><div className="text-xs font-bold text-slate-800">{dataset.dataset}</div><div className="mt-1 text-[10px] font-semibold uppercase tracking-wide text-indigo-700">{dataset.status.replace(/_/g, ' ')}</div><p className="mt-2 text-[11px] text-slate-600">{dataset.role}</p><p className="mt-1 text-[10px] leading-relaxed text-slate-500">{dataset.limitation}</p></div>)}</div>
        <div className="mt-3 flex items-start gap-2 rounded-xl border border-slate-200 bg-white p-3 text-[10px] leading-relaxed text-slate-600"><Info className="mt-0.5 h-3.5 w-3.5 shrink-0 text-slate-400" /><span>The existing <strong>3-Tier Benchmarks</strong> tab remains a separate 48-row synthetic fixture demonstration. Its template-derived scores are not empirical training results and should not be combined with the ASSISTments metrics above.</span></div>
      </section>
    </div>
  );
};
