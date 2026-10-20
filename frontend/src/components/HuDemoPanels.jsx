import React, { useMemo, useState } from 'react';
import { CircleCheck, CircleX, Shuffle } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card.jsx';
import { Badge } from './ui/badge.jsx';
import { Button } from './ui/button.jsx';
import { Input } from './ui/input.jsx';

// Panels for the Hungarian eval demo (evaluation/hu_legal/demo.py). Shown only when the
// server answers /api/demo/config.

const MAX_LISTED = 25;

export const isGoldChunk = (evalQuery, chunk) =>
  !!evalQuery && (evalQuery.gold_chunk_ids || []).includes(chunk?.metadata?.chunk_id);

const extractorDetail = (demo, name) => {
  if (name === 'crossencoder' && demo.crossencoder) {
    return `${demo.crossencoder.model.split('/').pop()} ≥ ${demo.crossencoder.threshold}`;
  }
  if (name === 'llm' && demo.llm) return demo.llm.model;
  return '';
};

export const ExtractorToggle = ({ demo, extractor, setExtractor, disabled }) => (
  <div className="space-y-2">
    <div className="flex flex-wrap items-center gap-2">
      <span className="text-sm lg:text-base font-medium text-foreground">Extractor:</span>
      {demo.extractors.map((name) => (
        <Button
          key={name}
          type="button"
          size="sm"
          variant={extractor === name ? 'default' : 'outline'}
          onClick={() => setExtractor(name)}
          disabled={disabled}
          title={extractorDetail(demo, name)}
        >
          {name}
        </Button>
      ))}
      <span className="text-xs lg:text-sm text-muted-foreground">
        {extractorDetail(demo, extractor)}
      </span>
    </div>
    <p className="text-xs lg:text-sm text-muted-foreground">
      Fused BM25 + {demo.dense_model?.split('/').pop()} top {demo.retrieve_k} →{' '}
      {demo.rerank_model?.split('/').pop()} → extraction on the top {demo.extract_k} chunks
    </p>
  </div>
);

const QueryMeta = ({ query }) => (
  <span className="text-xs text-muted-foreground">
    {query.answerable ? 'answerable' : `unanswerable (${query.unanswerable_reason})`}
    {' · '}
    {query.query_type} · {query.legal_domain} · {query.difficulty}
  </span>
);

export const EvalQueryPicker = ({ demo, queries, onPick, disabled }) => {
  const [filter, setFilter] = useState('all');
  const [search, setSearch] = useState('');

  const matches = useMemo(() => {
    const needle = search.trim().toLowerCase();
    return queries.filter((q) => {
      if (filter === 'answerable' && !q.answerable) return false;
      if (filter === 'unanswerable' && q.answerable) return false;
      return (
        !needle || q.query.toLowerCase().includes(needle) || q.qid.toLowerCase().includes(needle)
      );
    });
  }, [queries, filter, search]);

  const pickRandom = () => {
    if (matches.length) onPick(matches[Math.floor(Math.random() * matches.length)]);
  };

  return (
    <details className="border border-border rounded-lg bg-muted/40" open>
      <summary className="cursor-pointer select-none px-4 py-3 text-sm lg:text-base font-medium text-foreground">
        Eval set queries ({demo.split} split, {queries.length})
      </summary>
      <div className="px-4 pb-4 space-y-3">
        <div className="flex gap-2">
          <select
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
            className="border border-input bg-background rounded-md px-2 text-sm"
            aria-label="Filter queries"
          >
            <option value="all">all</option>
            <option value="answerable">answerable</option>
            <option value="unanswerable">unanswerable</option>
          </select>
          <Input
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            // the picker sits inside the question form; Enter here must not submit it
            onKeyDown={(e) => { if (e.key === 'Enter') e.preventDefault(); }}
            placeholder="Search query text or qid"
            className="flex-1"
          />
          <Button type="button" variant="outline" onClick={pickRandom} disabled={disabled || !matches.length}>
            <Shuffle /> Random
          </Button>
        </div>
        <ul className="max-h-64 overflow-y-auto space-y-1">
          {matches.slice(0, MAX_LISTED).map((q) => (
            <li key={q.qid}>
              <button
                type="button"
                onClick={() => onPick(q)}
                disabled={disabled}
                className="w-full text-left px-3 py-2 rounded hover:bg-accent disabled:opacity-50"
              >
                <span className="block text-sm lg:text-base text-foreground">{q.query}</span>
                <QueryMeta query={q} />
              </button>
            </li>
          ))}
        </ul>
        <p className="text-xs text-muted-foreground">
          {matches.length > MAX_LISTED
            ? `Showing ${MAX_LISTED} of ${matches.length} matches`
            : `${matches.length} match${matches.length !== 1 ? 'es' : ''}`}
        </p>
      </div>
    </details>
  );
};

// The retrieved-mode reading of one query: did it answer, and from a gold chunk?
const evalOutcome = (evalQuery, documents) => {
  const gold = (doc) => isGoldChunk(evalQuery, doc);
  const count = (docs) => docs.reduce((n, d) => n + (d.highlights?.length || 0), 0);
  const goldRank = documents.findIndex(gold) + 1;
  const spans = count(documents);
  const goldSpans = count(documents.filter(gold));

  let verdict;
  if (!evalQuery.answerable) {
    verdict = spans
      ? { ok: false, label: 'False answer: the query is unanswerable' }
      : { ok: true, label: 'Correct abstention' };
  } else if (!spans) {
    verdict = {
      ok: false,
      label: goldRank ? 'Abstained although a gold chunk was in context' : 'Abstained: no gold chunk retrieved',
    };
  } else if (goldSpans) {
    verdict = { ok: true, label: 'Answered from a gold chunk' };
  } else {
    verdict = { ok: false, label: 'Answered from non-gold chunks only' };
  }
  return { goldRank, spans, goldSpans, verdict };
};

export const EvalResult = ({ currentQuery }) => {
  const { evalQuery, documents = [], extractor } = currentQuery;
  if (!evalQuery) return null;
  const { goldRank, spans, goldSpans, verdict } = evalOutcome(evalQuery, documents);
  const Icon = verdict.ok ? CircleCheck : CircleX;

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-lg lg:text-xl">
          <Icon className={`w-6 h-6 ${verdict.ok ? 'text-green-600' : 'text-destructive'}`} />
          {verdict.label}
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-2 text-sm lg:text-base">
        <div className="flex flex-wrap gap-2">
          <Badge variant={evalQuery.answerable ? 'default' : 'secondary'}>
            {evalQuery.answerable ? 'answerable' : `unanswerable: ${evalQuery.unanswerable_reason}`}
          </Badge>
          <Badge variant="outline">{evalQuery.query_type}</Badge>
          <Badge variant="outline">{evalQuery.legal_domain}</Badge>
          <Badge variant="outline">{evalQuery.difficulty}</Badge>
          {extractor && <Badge variant="secondary">{extractor}</Badge>}
        </div>
        {evalQuery.answerable && (
          <p>
            Gold chunk in context:{' '}
            <span className="font-medium">{goldRank ? `yes, rank ${goldRank}` : 'no'}</span>
          </p>
        )}
        <p>
          Spans: <span className="font-medium">{spans}</span>
          {evalQuery.answerable && <> ({goldSpans} from gold chunks)</>}
        </p>
        <p className="text-xs text-muted-foreground font-mono break-all">{evalQuery.qid}</p>
      </CardContent>
    </Card>
  );
};
