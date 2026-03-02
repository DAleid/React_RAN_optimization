import type { IntentResult } from './types';

const BASE = process.env.NEXT_PUBLIC_API_URL ?? '';

export async function processIntent(intent: string): Promise<IntentResult> {
  const res = await fetch(`${BASE}/api/intent`, {
    method:  'POST',
    headers: { 'Content-Type': 'application/json' },
    body:    JSON.stringify({ intent }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ error: res.statusText }));
    throw new Error(err.error ?? 'Intent API error');
  }
  return res.json();
}
