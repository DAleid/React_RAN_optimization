'use client';

import React, { useState, useRef, useEffect } from 'react';
import { Send, Bot, User, Loader2, AlertCircle, ChevronDown, ChevronUp } from 'lucide-react';
import { processIntent } from '@/lib/api';
import type { IntentResult } from '@/lib/types';

// ── Sample intents ────────────────────────────────────────────────────────────
const SAMPLES = [
  'Prioritize emergency communications at the central hospital now',
  'Optimize for 50,000 fans at the stadium tonight',
  'Deploy IoT connectivity for 10,000 sensors in the smart factory',
  'Ensure low latency for telemedicine services across all micro cells',
  'Improve gaming experience for users in the downtown area',
];

// ── Single result section ────────────────────────────────────────────────────
function Section({ title, data }: { title: string; data: unknown }) {
  const [open, setOpen] = useState(true);
  if (!data) return null;
  return (
    <div className="border border-border rounded-lg overflow-hidden">
      <button
        onClick={() => setOpen(!open)}
        className="w-full flex items-center justify-between px-4 py-2.5
                   bg-bg-hover text-xs font-semibold text-text-secondary
                   hover:text-text-primary transition-colors"
      >
        {title}
        {open ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
      </button>
      {open && (
        <pre className="px-4 py-3 text-[11px] font-mono text-text-primary overflow-x-auto
                        bg-bg-primary leading-relaxed max-h-48 overflow-y-auto">
          {JSON.stringify(data, null, 2)}
        </pre>
      )}
    </div>
  );
}

// ── Intent result card ────────────────────────────────────────────────────────
function ResultCard({ result, warning }: { result: IntentResult; warning?: string }) {
  const r = result.result;
  return (
    <div className="card-glow space-y-3 animate-fade-in">
      <div className="flex items-center gap-2">
        <Bot size={14} className="text-accent-cyan" />
        <span className="text-sm font-semibold text-text-primary">AI Response</span>
        {r.fallback && (
          <span className="badge badge-warning ml-auto">Fallback mode</span>
        )}
      </div>
      {warning && (
        <div className="flex items-start gap-2 text-xs text-status-warning
                        bg-status-warning/10 border border-status-warning/30 rounded-lg p-2.5">
          <AlertCircle size={12} className="shrink-0 mt-0.5" />
          <span>{warning}</span>
        </div>
      )}
      <div className="space-y-2">
        <Section title="📋 Intent Analysis"    data={r.intent}       />
        <Section title="⚙️ Network Config"     data={r.config}       />
        <Section title="📊 Network Monitor"    data={r.monitor}      />
        <Section title="🔧 Optimization"       data={r.optimization} />
        {r.raw_output && (
          <Section title="📄 Raw Output"       data={r.raw_output}   />
        )}
      </div>
    </div>
  );
}

// ── Main component ────────────────────────────────────────────────────────────
export function IntentPanel() {
  const [input,    setInput]    = useState('');
  const [loading,  setLoading]  = useState(false);
  const [error,    setError]    = useState<string | null>(null);
  const [result,   setResult]   = useState<IntentResult | null>(null);
  const [history,  setHistory]  = useState<{ intent: string; result: IntentResult }[]>([]);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Auto-resize textarea
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = textareaRef.current.scrollHeight + 'px';
    }
  }, [input]);

  const submit = async (text: string) => {
    const t = text.trim();
    if (!t || loading) return;
    setInput('');
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const res = await processIntent(t);
      setResult(res);
      setHistory(h => [{ intent: t, result: res }, ...h.slice(0, 4)]);
    } catch (e: any) {
      setError(e.message ?? 'Failed to process intent');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex flex-col h-full gap-5">

      {/* Input area */}
      <div className="card-glow shrink-0 space-y-3">
        <div className="flex items-center gap-2 mb-1">
          <Bot size={15} className="text-accent-cyan" />
          <span className="text-sm font-semibold text-text-primary">Intent Input</span>
          <span className="badge badge-healthy ml-auto">AI-Powered</span>
        </div>
        <p className="text-xs text-text-secondary">
          Describe your network optimization goal in natural language.
          The AI will parse, plan, monitor, and optimize accordingly.
        </p>

        {/* Textarea */}
        <div className="relative">
          <textarea
            ref={textareaRef}
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={e => {
              if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); submit(input); }
            }}
            placeholder="e.g. Prioritize emergency services at the hospital…"
            rows={2}
            disabled={loading}
            className="w-full bg-bg-primary border border-border rounded-lg px-4 py-3
                       text-sm text-text-primary placeholder-text-muted resize-none
                       focus:outline-none focus:border-accent-cyan transition-colors
                       disabled:opacity-50 pr-14"
          />
          <button
            onClick={() => submit(input)}
            disabled={!input.trim() || loading}
            className="absolute right-3 bottom-3 btn-primary px-2.5 py-1.5"
          >
            {loading ? <Loader2 size={14} className="animate-spin" /> : <Send size={14} />}
          </button>
        </div>

        {/* Sample prompts */}
        <div className="space-y-1">
          <p className="section-title">Quick examples</p>
          <div className="flex flex-wrap gap-1.5">
            {SAMPLES.map(s => (
              <button
                key={s}
                onClick={() => submit(s)}
                disabled={loading}
                className="text-[11px] px-2.5 py-1 rounded-full border border-border
                           text-text-secondary hover:border-accent-cyan hover:text-accent-cyan
                           transition-colors disabled:opacity-40"
              >
                {s.length > 42 ? s.slice(0, 42) + '…' : s}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Loading state */}
      {loading && (
        <div className="card-glow shrink-0 animate-fade-in">
          <div className="flex items-center gap-3">
            <Loader2 size={16} className="text-accent-cyan animate-spin" />
            <div>
              <p className="text-sm font-medium text-text-primary">Processing intent…</p>
              <p className="text-xs text-text-secondary mt-0.5">
                Running 4 AI agents: Intent → Planner → Monitor → Optimizer
              </p>
            </div>
          </div>
          <div className="mt-3 grid grid-cols-4 gap-2">
            {['Intent Parser','Planner','Monitor','Optimizer'].map((a, i) => (
              <div key={a} className="flex flex-col items-center gap-1.5">
                <div className="w-8 h-8 rounded-full border border-accent-cyan/40
                                flex items-center justify-center">
                  <Loader2
                    size={14}
                    className="text-accent-cyan animate-spin"
                    style={{ animationDelay: `${i * 0.2}s` }}
                  />
                </div>
                <span className="text-[10px] text-text-secondary text-center leading-tight">{a}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Error */}
      {error && (
        <div className="card shrink-0 border-status-critical/40 animate-fade-in">
          <div className="flex items-start gap-2 text-sm text-status-critical">
            <AlertCircle size={14} className="shrink-0 mt-0.5" />
            <div>
              <p className="font-medium">Error</p>
              <p className="text-xs text-text-secondary mt-0.5">{error}</p>
              <p className="text-xs text-text-muted mt-1">
                Make sure the Python backend is running: <code className="font-mono">python api/intent.py</code>
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Result */}
      {result && !loading && (
        <div className="shrink-0">
          <ResultCard result={result} warning={result.warning} />
        </div>
      )}

      {/* History */}
      {history.length > 1 && (
        <div className="shrink-0 space-y-2">
          <p className="section-title">Previous intents</p>
          {history.slice(1).map(({ intent }, i) => (
            <button
              key={i}
              onClick={() => submit(intent)}
              disabled={loading}
              className="w-full text-left px-3 py-2 rounded-lg border border-border
                         text-xs text-text-secondary hover:border-accent-cyan/50
                         hover:text-text-primary transition-colors flex items-center gap-2"
            >
              <User size={11} className="shrink-0" />
              <span className="truncate">{intent}</span>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
