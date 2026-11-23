import { useState } from 'react';
import { Search, Network, Database } from 'lucide-react';
import { queryKnowledge } from '../api/client';
import { RetrievalResult } from '../types';

export function RetrievalExplorer() {
  const [query, setQuery] = useState('Why do fractions need a common denominator?');
  const [mode, setMode] = useState<RetrievalResult['mode']>('mix');
  const [namespace, setNamespace] = useState('curriculum');
  const [result, setResult] = useState<RetrievalResult | null>(null);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const search = async () => {
    setLoading(true); setError('');
    try { setResult(await queryKnowledge(query, mode, namespace)); }
    catch (error) { setError(error instanceof Error ? error.message : 'Retrieval failed'); }
    finally { setLoading(false); }
  };
  return <section className="space-y-5" aria-label="Retrieval explorer">
    <div className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
      <h2 className="flex items-center gap-2 text-lg font-extrabold text-slate-900"><Network className="h-5 w-5 text-indigo-600" /> Curriculum retrieval explorer</h2>
      <p className="mt-2 text-xs leading-relaxed text-slate-500">Inspect the sources and prerequisite relationships used by the live tutor. Local and global graph retrieval combine with BM25 and deterministic hashed lexical vectors. Rankings measure relevance, not a learner's understanding.</p>
      <form className="mt-5 space-y-3" onSubmit={(event) => { event.preventDefault(); search(); }}>
        <label className="block text-xs font-bold text-slate-700" htmlFor="retrieval-query">Question or reasoning to retrieve</label>
        <textarea id="retrieval-query" value={query} onChange={(event) => setQuery(event.target.value)} maxLength={2000} rows={3} className="w-full rounded-xl border border-slate-300 p-3 text-sm" />
        <div className="flex flex-wrap items-end gap-3">
          <label className="text-xs text-slate-600">Retrieval mode<select aria-label="Retrieval mode" value={mode} onChange={(event) => setMode(event.target.value as RetrievalResult['mode'])} className="ml-2 rounded-lg border border-slate-200 bg-slate-50 p-2">{['mix', 'hybrid', 'local', 'global', 'naive'].map((value) => <option key={value}>{value}</option>)}</select></label>
          <label className="text-xs text-slate-600">Source collection<select aria-label="Source collection" value={namespace} onChange={(event) => setNamespace(event.target.value)} className="ml-2 rounded-lg border border-slate-200 bg-slate-50 p-2"><option value="curriculum">App fraction curriculum</option><option value="eedi_research">Eedi research graph</option></select></label>
          <button disabled={loading || !query.trim()} className="inline-flex items-center gap-2 rounded-xl bg-indigo-600 px-5 py-2 text-xs font-bold text-white disabled:opacity-50"><Search className="h-4 w-4" />{loading ? 'Retrieving…' : 'Retrieve sources'}</button>
        </div>
      </form>
      {error && <p role="alert" className="mt-3 text-xs text-rose-700">{error}</p>}
    </div>
    {result && <>
      <div className="rounded-xl border border-indigo-200 bg-indigo-50 p-4 text-xs text-indigo-900" role="status">{result.abstained ? 'No supporting sources found in this collection. The system abstains.' : `${result.hits.length} sources retrieved · ${result.relationships.length} relationships · ${result.mode} mode`}{result.truncated && ' · Some sources omitted to respect the context budget.'}<code className="mt-2 block break-all text-[10px] text-indigo-500">{result.index_version} · {result.index_fingerprint}</code></div>
      {namespace === 'eedi_research' && <p className="rounded-xl border border-amber-200 bg-amber-50 p-4 text-xs text-amber-900">Eedi is external curriculum grounding under CC BY 4.0. These relationships are not learner responses or verified app taxonomy mappings. If the collection is empty, run the documented Eedi indexing command.</p>}
      <div className="grid gap-4 lg:grid-cols-2">{result.hits.map((hit) => <article key={hit.citation_id} className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm"><h3 className="text-sm font-bold text-slate-900">{hit.title}</h3><p className="mt-2 text-xs leading-relaxed text-slate-600">{hit.text}</p><div className="mt-3 flex flex-wrap gap-1">{hit.channels.map((channel) => <span key={channel} className="rounded-full bg-slate-100 px-2 py-1 text-[9px] text-slate-500">{channel}</span>)}</div><p className="mt-3 break-all text-[10px] text-slate-400">{hit.source} · {hit.source_revision} · {hit.license}</p><code className="mt-1 block text-[9px] text-indigo-500">{hit.citation_id}</code></article>)}</div>
      {result.relationships.length > 0 && <div className="overflow-x-auto rounded-2xl border border-slate-200 bg-white p-5"><h3 className="mb-3 flex items-center gap-2 text-sm font-bold"><Database className="h-4 w-4 text-indigo-600" /> Retrieved relationships</h3><table className="w-full text-left text-[11px]"><thead><tr className="text-slate-400"><th className="p-2">Entity</th><th className="p-2">Relation</th><th className="p-2">Related entity</th></tr></thead><tbody>{result.relationships.map((edge, index) => <tr key={index} className="border-t border-slate-100"><td className="p-2 font-mono">{edge.source_id}</td><td className="p-2 text-indigo-600">{edge.relation}</td><td className="p-2 font-mono">{edge.target_id}</td></tr>)}</tbody></table></div>}
    </>}
  </section>;
}
