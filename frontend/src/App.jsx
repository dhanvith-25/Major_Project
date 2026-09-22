import { useState } from "react";

const API = import.meta.env.VITE_API_URL || "http://127.0.0.1:8081";

async function api(path, options) {
  const response = await fetch(`${API}${path}`, options);
  const body = await response.json();
  if (!response.ok) throw new Error(body.detail || "Request failed");
  return body;
}

export default function App() {
  const [claim, setClaim] = useState("");
  const [result, setResult] = useState(null);
  const [webQuery, setWebQuery] = useState("");
  const [webResult, setWebResult] = useState(null);
  const [webLoading, setWebLoading] = useState(false);
  const [report, setReport] = useState(null);
  const [reportLoading, setReportLoading] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function predict(event) {
    event.preventDefault();
    if (claim.trim().length < 2) return setError("Enter at least two characters.");
    setLoading(true); setError("");
    try { setResult(await api("/health/predict", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ claim }) })); }
    catch (err) { setError(err.message); }
    finally { setLoading(false); }
  }

  async function verifyWithSources(event) {
    event.preventDefault();
    if (webQuery.trim().length < 2) return setError("Enter a question or claim to verify.");
    setWebLoading(true); setError(""); setWebResult(null);
    try { setWebResult(await api("/query", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ query: webQuery }) })); }
    catch (err) { setError(err.message); }
    finally { setWebLoading(false); }
  }

  async function fetchReport() {
    setReportLoading(true);
    setError("");
    try {
      const result = await api("/report?limit=10", { method: "GET" });
      setReport(result);
    } catch (err) {
      setError(err.message);
    } finally {
      setReportLoading(false);
    }
  }

  return <div className="shell">
    <header className="topbar"><div className="brand"><span className="brandMark">H</span><div><strong>Healthcare-Ai</strong><small>health intelligence</small></div></div><span className="prototype">RESEARCH PROTOTYPE</span></header>
    <main>
      <section className="hero"><p className="eyebrow">CLAIM VERIFICATION</p><h1>Is this health information<br /><em>supported by the model?</em></h1><p className="intro">Write one health claim and the classifier will return a prediction, confidence, and the full probability breakdown.</p></section>
      <section className="workspace">
        <div className="panel inputPanel">
          <div className="panelTitle"><span className="step">01</span><div><h2>Choose or write a claim</h2><p>Start with a dataset example or enter text below.</p></div></div>
          <label htmlFor="claim">Health claim</label>
          <textarea id="claim" value={claim} onChange={(event) => { setClaim(event.target.value); setResult(null); }} placeholder="Example: Antibiotics cure the common cold" />
          <button className="primary" onClick={predict} disabled={loading}>{loading ? "Analyzing claim…" : "Predict claim"}<span>→</span></button>
          {error && <p className="error">{error}</p>}
        </div>
        <div className="panel resultPanel">
          <div className="panelTitle"><span className="step">02</span><div><h2>Prediction result</h2><p>The model's classification for this claim.</p></div></div>
          {!result ? <div className="empty"><div className="emptyIcon">✦</div><p>Your result will appear here</p><small>Submit a claim to see its classification.</small></div> : <div className="prediction"><div className={`verdict ${result.verdict}`}><span className="dot" />{result.verdict}</div><div className="confidence"><div><span>Confidence</span><strong>{result.confidence}%</strong></div><div className="meter"><i style={{ width: `${result.confidence}%` }} /></div></div><div className="probabilities"><span>Probability breakdown</span>{Object.entries(result.probabilities).map(([label, value]) => <div className="prob" key={label}><b className={label}>{label}</b><div className="bar"><i className={label} style={{ width: `${value}%` }} /></div><strong>{value}%</strong></div>)}</div><p className="modelNote">{result.warning}</p></div>}
        </div>
      </section>
      <section className="webPanel">
        <div className="panelTitle"><span className="step">03</span><div><h2>Verify with live web sources</h2><p>The LLM searches the web, compares evidence, and returns cited sources.</p></div></div>
        <form className="webForm" onSubmit={verifyWithSources}><input value={webQuery} onChange={(event) => setWebQuery(event.target.value)} placeholder="Example: Do antibiotics cure the common cold?" /><button className="primary" type="submit" disabled={webLoading}>{webLoading ? "Searching sources…" : "Verify with sources"}<span>↗</span></button></form>
        {webResult && <div className="webResult"><div className={`verdict ${webResult.verdict}`}><span className="dot" />{webResult.verdict}</div><section className="detailBlock"><h3>Detailed answer</h3><p className="answer">{webResult.answer || "No answer was generated."}</p></section><div className="webMeta"><span>Validation score <strong>{webResult.validation_score ?? "—"}</strong></span><span>Validation status <strong>{webResult.validation_status || "—"}</strong></span></div><section className="detailBlock explanation"><h3>Why this verdict?</h3><p>{webResult.explanation || webResult.validation_reason || "The validator did not provide an explanation."}</p><p className="reason"><strong>Reason:</strong> {webResult.validation_reason || "Not provided."}</p></section>{webResult.evidence?.length > 0 && <section className="evidence"><h3>Supporting details</h3>{webResult.evidence.map((item, index) => <article className="evidenceItem" key={`${item.url}-${index}`}><div><b className={item.relationship}>{item.relationship}</b><span>{item.relevance ?? 0}% relevance</span></div><p>{item.reason || "This source was used by the validator."}</p></article>)}</section>}{webResult.sources?.length > 0 && <div className="sources"><h3>Supporting source links</h3>{webResult.sources.map((source, index) => <a href={source.url} target="_blank" rel="noreferrer" key={`${source.url}-${index}`}><strong>{source.title || source.url}</strong><small>{source.url}</small></a>)}</div>}</div>}
      </section>
      <section className="webPanel">
        <div className="panelTitle"><span className="step">04</span><div><h2>Request a query report</h2><p>Generate a summary of the recent verification queries and their verdicts.</p></div></div>
        <button className="primary" onClick={fetchReport} disabled={reportLoading}>{reportLoading ? "Generating report…" : "Generate report"}<span>▣</span></button>
        {report && <div className="webResult reportBox">
          <div className="webMeta"><span>Total queries <strong>{report.total_queries}</strong></span><span>Generated <strong>{new Date(report.generated_at).toLocaleString()}</strong></span></div>
          <section className="detailBlock"><h3>Verdict summary</h3><div className="summaryGrid">{Object.entries(report.summary || {}).map(([label, value]) => <div key={label} className="summaryItem"><strong>{label}</strong><span>{value}</span></div>)}</div></section>
          <section className="detailBlock"><h3>Recent query history</h3>{report.queries?.slice(0, 5).map((item) => <div className="reportItem" key={item.id}><p><strong>{item.query}</strong></p><small>{item.verdict} · {item.route} · {item.status}</small><p>{item.answer || "No answer recorded."}</p></div>)}</section>
        </div>}
      </section>
      <footer><span>---------</span><span>TF-IDF + Logistic Regression</span><span>Not medical advice</span></footer>
    </main>
  </div>;
}
