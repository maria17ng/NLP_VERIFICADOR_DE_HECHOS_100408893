import { FormEvent, useMemo, useState } from "react";

type Speaker = "usuario" | "modelo";
type Verdict = "true" | "false" | "unsure";
type MetricTone = "positive" | "negative" | "neutral";
type SummaryKey = "resumen" | "baseline";

type Message = {
  id: string;
  speaker: Speaker;
  text: string;
  language: string;
  verdict?: Verdict;
  confidence?: number;
  evidence?: string[];
  model?: string;
  stage?: string;
  baselineComparison?: {
    verdict: Verdict | "mixed";
    confidenceDelta: number;
    notes: string;
  };
};

type Metric = {
  label: string;
  value: string;
  trend: string;
  tone: MetricTone;
  description?: string;
};

type SummaryContent = Record<
  SummaryKey,
  {
    title: string;
    bullets: string[];
    footer: string;
  }
>;

type PipelineBadge = {
  label: string;
  value: string;
  detail: string;
};

const API_URL =
  import.meta.env.VITE_VERIFIER_ENDPOINT ?? "http://localhost:8000/api/verify";

type RunResult = {
  messages: Message[];
  metrics: Metric[];
  summary: SummaryContent;
  pipeline: PipelineBadge[];
  comparison?: boolean;
  rag?: RunResult;
  baseline?: RunResult;
  agreement?: boolean;
  time_diff?: number;
};

const verdictCopy: Record<Verdict, string> = {
  true: "Verdadero",
  false: "Falso",
  unsure: "Indeterminado"
};

const verdictTone: Record<Verdict, string> = {
  true: "text-verdict-true",
  false: "text-verdict-false",
  unsure: "text-verdict-unsure"
};

const verdictPill: Record<Verdict, string> = {
  true:
    "bg-gradient-to-r from-verdict-true/20 to-emerald-400/10 text-verdict-true border border-verdict-true/30",
  false:
    "bg-gradient-to-r from-verdict-false/20 to-rose-400/10 text-verdict-false border border-verdict-false/40",
  unsure:
    "bg-gradient-to-r from-verdict-unsure/20 to-amber-400/10 text-verdict-unsure border border-verdict-unsure/40"
};

const summaryTabs: SummaryKey[] = ["resumen", "baseline"];

const defaultBadges: PipelineBadge[] = [
  {
    label: "Pipeline",
    value: "RAG determinista v3",
    detail: "Chunker híbrido + re-ranker MMR + verificador semántico"
  },
  {
    label: "LLM verificador",
    value: "gpt-4.1-mini",
    detail: "Modo determinista, temperatura 0.1, top-p 0.3"
  },
  {
    label: "Vector store",
    value: "Chroma + text-embedding-3-large",
    detail: "MMR λ=0.45, filtros temáticos activos"
  }
];

const trendTone: Record<MetricTone, string> = {
  positive: "text-emerald-300",
  negative: "text-rose-300",
  neutral: "text-slate-300"
};

const toneBg: Record<MetricTone, string> = {
  positive: "bg-emerald-400/10",
  negative: "bg-rose-400/10",
  neutral: "bg-white/5"
};

const formatConfidence = (value?: number) =>
  typeof value === "number" ? `${Math.round(value * 100)}%` : "-";

function App() {
  const [question, setQuestion] = useState("");
  const [conversation, setConversation] = useState<Message[]>([]);
  const [metrics, setMetrics] = useState<Metric[]>([]);
  const [summaryData, setSummaryData] = useState<SummaryContent | null>(null);
  const [summaryMode, setSummaryMode] = useState<SummaryKey>("resumen");
  const [badges, setBadges] = useState<PipelineBadge[]>(() => [...defaultBadges]);
  const [status, setStatus] = useState<"idle" | "loading" | "ready">("idle");
  const [error, setError] = useState<string | null>(null);
  const [compareBaseline, setCompareBaseline] = useState(false);
  const [comparisonData, setComparisonData] = useState<{
    rag: RunResult;
    baseline: RunResult;
    agreement: boolean;
    time_diff: number;
  } | null>(null);

  const primaryVerdict = useMemo(() => {
    const latest = [...conversation].reverse().find((msg) => msg.verdict);
    return latest?.verdict ?? "unsure";
  }, [conversation]);

  const handleSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const trimmed = question.trim();
    if (!trimmed || status === "loading") return;

    setStatus("loading");
    setError(null);
    setSummaryMode("resumen");
    setConversation([]);
    setMetrics([]);
    setSummaryData(null);
    setBadges([...defaultBadges]);
    setComparisonData(null);

    try {
      const result = await fetchVerification(trimmed, compareBaseline);

      if (result.comparison && result.rag && result.baseline) {
        // Modo comparación
        setComparisonData({
          rag: result.rag,
          baseline: result.baseline,
          agreement: result.agreement ?? false,
          time_diff: result.time_diff ?? 0
        });
        setConversation(result.rag.messages);
        setMetrics(result.rag.metrics);
        setSummaryData(result.rag.summary);
        setBadges(result.rag.pipeline);
      } else {
        // Modo normal
        setConversation(result.messages);
        setMetrics(result.metrics);
        setSummaryData(result.summary);
        setBadges(result.pipeline);
      }

      setQuestion("");
      setStatus("ready");
    } catch (err) {
      console.error(err);
      const message = err instanceof Error ? err.message : "No se pudo completar la verificación.";
      setError(message);
      setStatus("idle");
    }
  };

  return (
    <div className="min-h-screen bg-background/95 text-white">
      <div className="mx-auto flex max-w-6xl flex-col gap-6 px-6 pb-12 pt-10">
        <header className="flex flex-col gap-4 rounded-3xl border border-white/5 bg-gradient-to-br from-slate-900/70 via-slate-900/30 to-slate-900/20 p-6 shadow-glow backdrop-blur">
          <div className="flex flex-wrap items-center justify-between gap-4">
            <div>
              <p className="text-sm uppercase tracking-[0.2em] text-white/60">Fact Checker</p>
              <h1 className="mt-1 text-3xl font-semibold text-white">
                Panel de verificación en tiempo real
              </h1>
            </div>
            <span className={`rounded-full px-4 py-2 text-sm font-medium ${verdictPill[primaryVerdict]}`}>
              Veredicto actual: {verdictCopy[primaryVerdict]}
            </span>
          </div>
          <div className="grid gap-3 sm:grid-cols-3">
            {badges.map((badge) => (
              <InfoBadge key={badge.label} label={badge.label} value={badge.value} detail={badge.detail} />
            ))}
          </div>
        </header>

        <main className="grid gap-6 lg:grid-cols-[1.4fr_0.8fr]">
          <section className="rounded-3xl border border-white/5 bg-card/70 p-6 shadow-glow backdrop-blur">
            <div className="flex flex-wrap items-center justify-between gap-3">
              <div>
                <h2 className="text-lg font-semibold text-white/90">Conversación</h2>
                <p className="text-sm text-white/60">{conversation.length} turnos</p>
              </div>
              {status === "loading" && (
                <span className="flex items-center gap-2 text-xs uppercase tracking-[0.3em] text-white/60">
                  <span className="h-2 w-2 animate-ping rounded-full bg-white" />
                  Procesando
                </span>
              )}
            </div>

            <form onSubmit={handleSubmit} className="mt-5 flex flex-col gap-3">
              <div className="flex items-center gap-3 mb-2">
                <label className="flex items-center gap-2 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={compareBaseline}
                    onChange={(e) => setCompareBaseline(e.target.checked)}
                    disabled={status === "loading"}
                    className="w-4 h-4 rounded border-white/20 bg-white/5 text-white focus:ring-white/40"
                  />
                  <span className="text-sm text-white/80">Comparar con sistema baseline (TF-IDF)</span>
                </label>
              </div>
              <div className="flex flex-col gap-3 sm:flex-row">
                <textarea
                  name="question"
                  placeholder="Escribe la afirmación que quieres verificar"
                  className="min-h-[90px] flex-1 resize-none rounded-2xl border border-white/10 bg-white/5 p-4 text-base text-white/90 focus:border-white/40 focus:outline-none"
                  value={question}
                  onChange={(e) => setQuestion(e.target.value)}
                  disabled={status === "loading"}
                />
                <button
                  type="submit"
                  className="rounded-2xl bg-white/90 px-6 py-3 text-base font-semibold text-slate-900 shadow-glow transition hover:bg-white disabled:cursor-not-allowed disabled:bg-white/50"
                  disabled={status === "loading" || question.trim().length === 0}
                >
                  {status === "loading" ? "Verificando…" : "Verificar"}
                </button>
              </div>
            </form>

            {error && (
              <div className="mt-4 rounded-2xl border border-rose-500/30 bg-rose-500/10 p-4 text-sm text-rose-100">
                {error}
              </div>
            )}

            <div className="mt-6 space-y-4 overflow-y-auto pr-2 scrollbar" style={{ maxHeight: "55vh" }}>
              {conversation.length === 0 && status !== "loading" && (
                <EmptyPlaceholder
                  title="Aún no hay turnos"
                  description="Envía una pregunta arriba para activar el pipeline de verificación."
                />
              )}

              {status === "loading" && (
                <div className="space-y-4">
                  {[1, 2].map((skeleton) => (
                    <div key={skeleton} className="animate-pulse rounded-3xl border border-white/5 bg-white/5 p-5">
                      <div className="h-4 w-24 rounded-full bg-white/20" />
                      <div className="mt-3 h-5 w-3/4 rounded-full bg-white/10" />
                      <div className="mt-2 h-5 w-2/3 rounded-full bg-white/10" />
                    </div>
                  ))}
                </div>
              )}

              {conversation.map((message) => (
                <MessageBubble key={message.id} message={message} />
              ))}
            </div>

            {/* Comparación con baseline */}
            {comparisonData && (
              <div className="mt-6 rounded-3xl border border-white/10 bg-gradient-to-br from-blue-900/20 to-purple-900/20 p-6">
                <h3 className="text-lg font-semibold text-white/90 mb-4 flex items-center gap-2">
                  <span className="text-2xl">⚖️</span>
                  Comparación: RAG vs Baseline
                </h3>

                <div className="grid gap-4 sm:grid-cols-2 mb-4">
                  <div className="rounded-2xl border border-white/10 bg-white/5 p-4">
                    <div className="text-xs uppercase tracking-wider text-white/60 mb-2">Sistema RAG</div>
                    <div className={`text-2xl font-bold ${verdictTone[comparisonData.rag.messages.find(m => m.verdict)?.verdict ?? 'unsure']}`}>
                      {verdictCopy[comparisonData.rag.messages.find(m => m.verdict)?.verdict ?? 'unsure']}
                    </div>
                    <div className="text-sm text-white/70 mt-1">
                      Confianza: {formatConfidence(comparisonData.rag.messages.find(m => m.confidence)?.confidence)}
                    </div>
                  </div>

                  <div className="rounded-2xl border border-white/10 bg-white/5 p-4">
                    <div className="text-xs uppercase tracking-wider text-white/60 mb-2">Baseline (TF-IDF)</div>
                    <div className={`text-2xl font-bold ${verdictTone[comparisonData.baseline.messages.find(m => m.verdict)?.verdict ?? 'unsure']}`}>
                      {verdictCopy[comparisonData.baseline.messages.find(m => m.verdict)?.verdict ?? 'unsure']}
                    </div>
                    <div className="text-sm text-white/70 mt-1">
                      Confianza: {formatConfidence(comparisonData.baseline.messages.find(m => m.confidence)?.confidence)}
                    </div>
                  </div>
                </div>

                <div className="flex items-center justify-between rounded-2xl border border-white/10 bg-white/5 p-4">
                  <div className="flex items-center gap-3">
                    <span className="text-3xl">{comparisonData.agreement ? '✅' : '⚠️'}</span>
                    <div>
                      <div className="font-semibold text-white/90">
                        {comparisonData.agreement ? 'Sistemas en acuerdo' : 'Sistemas en desacuerdo'}
                      </div>
                      <div className="text-xs text-white/60">
                        Diferencia de tiempo: {comparisonData.time_diff > 0 ? '+' : ''}{comparisonData.time_diff.toFixed(0)}ms
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </section>

          <aside className="flex h-full flex-col gap-6">
            <section className="rounded-3xl border border-white/5 bg-card/70 p-6 shadow-glow backdrop-blur">
              <h2 className="text-lg font-semibold text-white/90">Salud del pipeline</h2>
              <div className="mt-5 grid gap-4">
                {status === "loading" ? (
                  <div className="space-y-3">
                    {[1, 2].map((id) => (
                      <div key={id} className="h-24 animate-pulse rounded-2xl border border-white/5 bg-white/5" />
                    ))}
                  </div>
                ) : metrics.length === 0 ? (
                  <EmptyPlaceholder
                    title="Sin métricas"
                    description="Las métricas aparecerán tras la primera verificación."
                  />
                ) : (
                  metrics.map((metric) => <MetricCard key={metric.label} metric={metric} />)
                )}
              </div>
            </section>

            <section className="rounded-3xl border border-white/5 bg-card/70 p-6 shadow-glow backdrop-blur">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <h2 className="text-lg font-semibold text-white/90">Resumen & baseline</h2>
                <div className="flex gap-2">
                  {(summaryData ? (Object.keys(summaryData) as SummaryKey[]) : summaryTabs).map((mode) => (
                    <button
                      key={mode}
                      onClick={() => setSummaryMode(mode)}
                      className={`rounded-full px-3 py-1 text-sm capitalize transition ${
                        summaryMode === mode
                          ? "bg-white/90 text-slate-900"
                          : "bg-white/10 text-white/70 hover:bg-white/20"
                      }`}
                      disabled={!summaryData}
                    >
                      {mode}
                    </button>
                  ))}
                </div>
              </div>

              {summaryData ? (
                <SummaryPanel content={summaryData[summaryMode]} />
              ) : status === "loading" ? (
                <div className="mt-4 space-y-3">
                  {[1, 2, 3].map((id) => (
                    <div key={id} className="h-4 animate-pulse rounded-full bg-white/10" />
                  ))}
                </div>
              ) : (
                <EmptyPlaceholder
                  title="Sin resumen"
                  description="Se generará un resumen ejecutivo cuando llegue la primera respuesta."
                />
              )}
            </section>
          </aside>
        </main>
      </div>
    </div>
  );
}

function MessageBubble({ message }: { message: Message }) {
  const isUser = message.speaker === "usuario";
  const verdict = message.verdict;

  return (
    <article
      className={`rounded-3xl border px-4 py-5 shadow-inner transition ${
        isUser
          ? "border-white/10 bg-white/5"
          : "border-white/5 bg-gradient-to-br from-slate-900/80 to-slate-900/40"
      }`}
    >
      <div className="flex flex-wrap items-center justify-between gap-2 text-xs uppercase tracking-widest text-white/50">
        <span>{isUser ? "Usuario" : "Verificador"}</span>
        <span className="font-mono">Idioma: {message.language}</span>
      </div>
      <p className="mt-3 text-base leading-relaxed text-white/90">{message.text}</p>

      {verdict && (
        <div className="mt-4 grid gap-3 text-sm sm:grid-cols-3">
          <div className="rounded-2xl border border-white/10 bg-white/5 px-3 py-2">
            <p className="text-xs uppercase tracking-widest text-white/50">Veredicto</p>
            <p className={`text-base font-semibold ${verdictTone[verdict]}`}>{verdictCopy[verdict]}</p>
          </div>
          <div className="rounded-2xl border border-white/10 bg-white/5 px-3 py-2">
            <p className="text-xs uppercase tracking-widest text-white/50">Confianza</p>
            <p className="text-base font-semibold text-white">{formatConfidence(message.confidence)}</p>
          </div>
          <div className="rounded-2xl border border-white/10 bg-white/5 px-3 py-2">
            <p className="text-xs uppercase tracking-widest text-white/50">Etapa</p>
            <p className="text-sm font-medium text-white/80">{message.stage ?? "-"}</p>
          </div>
        </div>
      )}

      {message.model && (
        <div className="mt-3 text-xs text-white/60">Modelo: {message.model}</div>
      )}

      {message.evidence && (
        <div className="mt-4 rounded-2xl border border-white/10 bg-black/20 p-3">
          <p className="text-xs uppercase tracking-[0.3em] text-white/50">Evidencia</p>
          <ul className="mt-2 list-disc space-y-1 pl-5 text-sm text-white/80">
            {message.evidence.map((item) => (
              <li key={item}>{item}</li>
            ))}
          </ul>
        </div>
      )}

      {message.baselineComparison && (
        <div className="mt-4 rounded-2xl border border-white/10 bg-white/5 p-3">
          <p className="text-xs uppercase tracking-[0.3em] text-white/50">Baseline</p>
          <div className="mt-2 flex flex-wrap items-center gap-2 text-sm text-white/80">
            <span className={`rounded-full px-2 py-0.5 text-xs font-semibold ${verdictPill[message.baselineComparison.verdict as Verdict] ?? "bg-white/10"}`}>
              {message.baselineComparison.verdict === "mixed"
                ? "Veredicto mixto"
                : `Baseline: ${verdictCopy[message.baselineComparison.verdict as Verdict]}`}
            </span>
            <span className="font-mono text-xs text-white/70">
              Δ confianza {message.baselineComparison.confidenceDelta > 0 ? "+" : ""}
              {Math.round(message.baselineComparison.confidenceDelta * 100)}%
            </span>
          </div>
          <p className="mt-2 text-sm text-white/80">{message.baselineComparison.notes}</p>
        </div>
      )}
    </article>
  );
}

function MetricCard({ metric }: { metric: Metric }) {
  return (
    <article className="rounded-2xl border border-white/5 bg-white/5 p-4">
      <p className="text-xs uppercase tracking-[0.3em] text-white/60">{metric.label}</p>
      <div className="mt-2 flex items-end justify-between">
        <p className="text-3xl font-semibold text-white">{metric.value}</p>
        <span className={`rounded-full px-3 py-1 text-xs font-semibold ${toneBg[metric.tone]} ${trendTone[metric.tone]}`}>
          {metric.trend}
        </span>
      </div>
      {metric.description && <p className="mt-3 text-sm text-white/70">{metric.description}</p>}
    </article>
  );
}

function SummaryPanel({ content }: { content: SummaryContent[SummaryKey] }) {
  return (
    <div className="mt-4 space-y-4">
      <h3 className="text-base font-semibold text-white/90">{content.title}</h3>
      <ul className="space-y-3 text-sm text-white/80">
        {content.bullets.map((bullet) => (
          <li key={bullet} className="flex gap-3">
            <span className="mt-1 text-lg text-verdict-unsure">•</span>
            <p>{bullet}</p>
          </li>
        ))}
      </ul>
      <p className="text-xs uppercase tracking-[0.3em] text-white/50">{content.footer}</p>
    </div>
  );
}

function InfoBadge({
  label,
  value,
  detail
}: {
  label: string;
  value: string;
  detail: string;
}) {
  return (
    <div className="group relative overflow-hidden rounded-2xl border border-white/10 bg-white/5 px-4 py-3">
      <p className="text-xs uppercase tracking-[0.3em] text-white/50">{label}</p>
      <p className="mt-1 text-sm font-medium text-white">{value}</p>
      <div className="pointer-events-none absolute left-1/2 top-full z-20 mt-2 hidden w-60 -translate-x-1/2 rounded-2xl border border-white/10 bg-slate-900/95 p-3 text-xs text-white/80 shadow-glow transition group-hover:block">
        {detail}
      </div>
    </div>
  );
}

function EmptyPlaceholder({
  title,
  description
}: {
  title: string;
  description: string;
}) {
  return (
    <div className="rounded-3xl border border-dashed border-white/15 bg-white/5 p-6 text-center">
      <p className="text-sm font-semibold text-white/80">{title}</p>
      <p className="mt-2 text-sm text-white/60">{description}</p>
    </div>
  );
}

async function fetchVerification(question: string, compareBaseline: boolean = false): Promise<RunResult> {
  const response = await fetch(API_URL, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, compare_baseline: compareBaseline })
  });

  let payload: unknown = null;
  try {
    payload = await response.json();
  } catch (error) {
    // El backend siempre debería regresar JSON; si no, detenemos la ejecución
    throw new Error("Respuesta inválida del verificador");
  }

  if (!response.ok) {
    const detail =
      typeof payload === "object" && payload && "detail" in payload
        ? String((payload as { detail: unknown }).detail)
        : "Error al llamar al verificador";
    throw new Error(detail);
  }

  return payload as RunResult;
}

export default App;
