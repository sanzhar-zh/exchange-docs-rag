"use client";

import { ReactNode, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

const API_BASE = process.env.API_BASE ?? "http://localhost:8000";
const GITHUB = "https://github.com/sanzhar-zh";

// Short labels, full questions. The questions are deliberately wordy - answering
// them as a person would phrase them is the point - but a row of buttons carrying
// the whole sentence wraps after every one and leaves the row half empty.
const EXAMPLES: { label: string; question: string }[] = [
  {
    label: "take profit and stop loss",
    question:
      "how do I attach a take profit and stop loss to an existing Bybit position",
  },
  {
    label: "unknown order error",
    question:
      "what does Binance return when the order I asked about does not exist",
  },
  {
    label: "websocket auth",
    question: "how do I authenticate a Bybit websocket connection",
  },
  { label: "off topic", question: "what is the capital of France" },
];

type Source = {
  n: number;
  exchange: string;
  section: string;
  source: string;
  excerpt: string;
};

type Answer = {
  answer: string;
  sources: Source[];
  provider: string;
};

// Matches [1] and the grouped form the models also produce, [1, 4].
const CITATION = /(\[\d+(?:,\s*\d+)*\])/;

/**
 * Turn citation markers into chips, leaving every other character alone.
 *
 * The model is told to cite after each claim, so unstyled brackets appear several
 * times a sentence and crowd the text out. This runs over the strings react-markdown
 * has already produced, which is why the answer keeps its headings, lists and code
 * spans: markdown is rendered normally and only the markers inside it are replaced.
 */
function decorate(node: ReactNode): ReactNode {
  if (typeof node === "string") {
    return node.split(CITATION).map((part, i) =>
      CITATION.test(part) ? (
        <span key={i}>
          {part
            .slice(1, -1)
            .split(",")
            .map((number, j) => (
              <span className="cite" key={j}>
                {number.trim()}
              </span>
            ))}
        </span>
      ) : (
        <span key={i}>{part}</span>
      ),
    );
  }
  if (Array.isArray(node)) {
    return node.map((child, i) => (
      <span key={i}>{decorate(child as ReactNode)}</span>
    ));
  }
  return node;
}

// Headings are levelled to h3: the model picks its own depth, sometimes starting at
// h1, and a heading inside a card should not outrank the card's own label.
const MARKDOWN_COMPONENTS = {
  p: ({ children }: { children?: ReactNode }) => <p>{decorate(children)}</p>,
  li: ({ children }: { children?: ReactNode }) => <li>{decorate(children)}</li>,
  h1: ({ children }: { children?: ReactNode }) => <h3>{decorate(children)}</h3>,
  h2: ({ children }: { children?: ReactNode }) => <h3>{decorate(children)}</h3>,
  h3: ({ children }: { children?: ReactNode }) => <h3>{decorate(children)}</h3>,
  h4: ({ children }: { children?: ReactNode }) => <h3>{decorate(children)}</h3>,
};

function SourceCard({ source }: { source: Source }) {
  const [open, setOpen] = useState(false);
  const exchange = source.exchange.startsWith("binance") ? "binance" : "bybit";

  return (
    <div className="source">
      <div className="source-head">
        <span className="num">{source.n}</span>
        <span className={`badge ${exchange}`}>{exchange}</span>
        <span className="section" title={source.section}>
          {source.section}
        </span>
      </div>
      <div className="path mono">{source.source}</div>
      <div className={`excerpt mono ${open ? "open" : ""}`}>
        {source.excerpt}
      </div>
      <button className="link" onClick={() => setOpen(!open)}>
        {open ? "show less" : "show the passage"}
      </button>
    </div>
  );
}

export default function Home() {
  const [question, setQuestion] = useState("");
  const [result, setResult] = useState<Answer | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  async function ask(text: string) {
    if (!text.trim() || loading) return;

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await fetch(`${API_BASE}/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: text }),
      });
      if (!response.ok) {
        // The API sends its own explanation for a failure it understood, such as
        // an exhausted provider quota. Fall back to the status only if it did not.
        const body = await response.json().catch(() => null);
        throw new Error(body?.detail ?? `the API answered ${response.status}`);
      }
      setResult(await response.json());
    } catch (cause) {
      // fetch reports a dead server and a blocked cross-origin response the same
      // way, as a bare TypeError, so this cannot claim the server is down - an
      // unhandled 500 reaches the browser without CORS headers and lands here too.
      setError(
        cause instanceof TypeError
          ? `no response from the API at ${API_BASE}. Check that uvicorn is ` +
            `running and look at its log for an error.`
          : String(cause),
      );
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="shell">
      <header>
        <div className="title-row">
          <h1>Exchange Docs RAG</h1>
          <a className="byline" href={GITHUB} target="_blank" rel="noreferrer">
            github.com/sanzhar-zh
          </a>
        </div>
        <p>
          Questions about the Binance Spot and Bybit v5 APIs, answered only from
          their documentation, with the passages used shown alongside.
        </p>
      </header>

      <form
        onSubmit={(event) => {
          event.preventDefault();
          ask(question);
        }}
      >
        <input
          type="text"
          value={question}
          placeholder="how do I cancel an order on Bybit"
          onChange={(event) => setQuestion(event.target.value)}
        />
        <button
          className="primary"
          type="submit"
          disabled={loading || !question.trim()}
        >
          {loading ? "Searching" : "Ask"}
        </button>
      </form>

      <div className="examples">
        {EXAMPLES.map((example) => (
          <button
            className="chip"
            key={example.label}
            disabled={loading}
            title={example.question}
            onClick={() => {
              setQuestion(example.question);
              ask(example.question);
            }}
          >
            {example.label}
          </button>
        ))}
      </div>

      {loading && (
        <div className="note">
          <span className="spinner" />
          searching 4941 passages, then asking the model
        </div>
      )}

      {error && <div className="note error">{error}</div>}

      {result && (
        <>
          <section className="panel">
            <h2>Answer</h2>
            <div className="answer">
              <ReactMarkdown
                remarkPlugins={[remarkGfm]}
                components={MARKDOWN_COMPONENTS}
              >
                {result.answer}
              </ReactMarkdown>
            </div>
            <div className="provider">answered by {result.provider}</div>
          </section>

          <section className="panel">
            <h2>Retrieved passages</h2>
            {result.sources.map((source) => (
              <SourceCard key={source.n} source={source} />
            ))}
          </section>
        </>
      )}
    </main>
  );
}
