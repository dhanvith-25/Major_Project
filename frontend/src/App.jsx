import { useEffect, useState } from "react";

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
  const [symptomText, setSymptomText] = useState("");
  const [symptomImage, setSymptomImage] = useState(null);
  const [symptomResult, setSymptomResult] = useState(null);
  const [symptomLoading, setSymptomLoading] = useState(false);
  const [report, setReport] = useState(null);
  const [reportLoading, setReportLoading] = useState(false);
  const [personalReports, setPersonalReports] = useState([]);
  const [selectedPersonalReport, setSelectedPersonalReport] = useState(null);
  const [personalSummary, setPersonalSummary] = useState(null);
  const [personalFile, setPersonalFile] = useState(null);
  const [personalSearch, setPersonalSearch] = useState("");
  const [personalQuestion, setPersonalQuestion] = useState("");
  const [personalAnswer, setPersonalAnswer] = useState(null);
  const [personalLoading, setPersonalLoading] = useState(false);
  const [personalUploading, setPersonalUploading] = useState(false);
  const [personalAsking, setPersonalAsking] = useState(false);
  const [personalError, setPersonalError] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => { loadPersonalRecords(); }, []);

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

  async function checkSymptoms(event) {
    event.preventDefault();
    if (!symptomText.trim() && !symptomImage) return setError("Add symptoms or upload an image to analyze.");
    setSymptomLoading(true); setError(""); setSymptomResult(null);
    try {
      const form = new FormData();
      if (symptomText.trim()) form.append("symptoms", symptomText);
      if (symptomImage) form.append("image", symptomImage);
      const result = await fetch(`${API}/symptom-check`, { method: "POST", body: form });
      const body = await result.json();
      if (!result.ok) throw new Error(body.detail || "Symptom check failed");
      setSymptomResult(body);
    } catch (err) {
      setError(err.message);
    } finally {
      setSymptomLoading(false);
    }
  }

  async function loadPersonalRecords(searchTerm = "") {
    setPersonalLoading(true);
    setPersonalError("");
    try {
      const query = searchTerm.trim() ? `?query=${encodeURIComponent(searchTerm.trim())}` : "";
      const [reports, summary] = await Promise.all([
        api(`/personal-reports${query}`),
        api(`/personal-reports/summary${query}`),
      ]);
      setPersonalReports(reports);
      setPersonalSummary(summary);
      setSelectedPersonalReport((current) => {
        if (!reports.length) return null;
        return reports.find((report) => report.id === current?.id) || reports[0];
      });
    } catch (err) {
      setPersonalError(err.message);
    } finally {
      setPersonalLoading(false);
    }
  }

  async function uploadPersonalReport(event) {
    event.preventDefault();
    if (!personalFile) return setPersonalError("Choose a PDF or image report first.");
    setPersonalUploading(true);
    setPersonalError("");
    try {
      const form = new FormData();
      form.append("file", personalFile);
      const saved = await api("/personal-reports/upload", { method: "POST", body: form });
      setPersonalFile(null);
      setSelectedPersonalReport(saved);
      await loadPersonalRecords(personalSearch);
    } catch (err) {
      setPersonalError(err.message);
    } finally {
      setPersonalUploading(false);
    }
  }

  async function askAboutPersonalRecords(event) {
    event.preventDefault();
    if (personalQuestion.trim().length < 2) return setPersonalError("Ask a question about a saved test or result.");
    setPersonalAsking(true);
    setPersonalError("");
    try {
      setPersonalAnswer(await api("/personal-reports/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: personalQuestion }),
      }));
    } catch (err) {
      setPersonalError(err.message);
    } finally {
      setPersonalAsking(false);
    }
  }

  const reportGroups = personalReports.reduce((groups, report) => {
    const type = report.test_type || "Other reports";
    groups[type] = groups[type] || [];
    groups[type].push(report);
    return groups;
  }, {});

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
        <div className="panelTitle"><span className="step">03</span><div><h2>Misinformation Detection</h2><p>The LLM searches the web, compares evidence, and returns cited sources.</p></div></div>
        <form className="webForm" onSubmit={verifyWithSources}><input value={webQuery} onChange={(event) => setWebQuery(event.target.value)} placeholder="Example: Do antibiotics cure the common cold?" /><button className="primary" type="submit" disabled={webLoading}>{webLoading ? "Searching sources…" : "Verify with sources"}<span>↗</span></button></form>
        {webResult && <div className="webResult"><div className={`verdict ${webResult.verdict}`}><span className="dot" />{webResult.verdict}</div><section className="detailBlock"><h3>Detailed answer</h3><p className="answer">{webResult.answer || "No answer was generated."}</p></section><div className="webMeta"><span>Validation score <strong>{webResult.validation_score ?? "—"}</strong></span><span>Validation status <strong>{webResult.validation_status || "—"}</strong></span></div><section className="detailBlock explanation"><h3>Why this verdict?</h3><p>{webResult.explanation || webResult.validation_reason || "The validator did not provide an explanation."}</p><p className="reason"><strong>Reason:</strong> {webResult.validation_reason || "Not provided."}</p></section>{webResult.evidence?.length > 0 && <section className="evidence"><h3>Supporting details</h3>{webResult.evidence.map((item, index) => <article className="evidenceItem" key={`${item.url}-${index}`}><div><b className={item.relationship}>{item.relationship}</b><span>{item.relevance ?? 0}% relevance</span></div><p>{item.reason || "This source was used by the validator."}</p></article>)}</section>}{webResult.sources?.length > 0 && <div className="sources"><h3>Supporting source links</h3>{webResult.sources.map((source, index) => <a href={source.url} target="_blank" rel="noreferrer" key={`${source.url}-${index}`}><strong>{source.title || source.url}</strong><small>{source.url}</small></a>)}</div>}</div>}
      </section>
      <section className="webPanel symptomPanel">
        <div className="panelTitle"><span className="step">04</span><div><h2>Symptom checker</h2><p>Upload an image or type symptoms and review the likely causes.</p></div></div>
        <form className="symptomForm" onSubmit={checkSymptoms}>
          <label htmlFor="symptomImage">Image upload</label>
          <input id="symptomImage" type="file" accept="image/*" onChange={(event) => setSymptomImage(event.target.files?.[0] || null)} />
          <label htmlFor="symptomText">Symptoms</label>
          <textarea id="symptomText" value={symptomText} onChange={(event) => setSymptomText(event.target.value)} placeholder="Example: fever, cough, sore throat, body aches" />
          <button className="primary" type="submit" disabled={symptomLoading}>{symptomLoading ? "Checking symptoms…" : "Analyze symptoms"}<span>✦</span></button>
        </form>
        {symptomResult && <div className="webResult">
          <section className="detailBlock"><h3>Summary</h3><p className="answer">{symptomResult.summary}</p></section>
          <div className="webMeta"><span>Detected symptoms <strong>{symptomResult.detected_symptoms?.length ?? 0}</strong></span><span>OCR status <strong>{symptomResult.ocr_status || "—"}</strong></span></div>
          {symptomResult.ocr_text && <section className="detailBlock"><h3>OCR text</h3><p className="reason">{symptomResult.ocr_text}</p></section>}
          {symptomResult.ocr_warning && <section className="detailBlock"><h3>OCR note</h3><p className="reason">{symptomResult.ocr_warning}</p></section>}
          {symptomResult.suggested_causes?.length > 0 && <section className="detailBlock"><h3>Suggested causes</h3>{symptomResult.suggested_causes.map((item, index) => <div className="reportItem symptomCause" key={`${item.cause}-${index}`}><strong>{item.cause}</strong><small>{item.confidence} confidence</small><p>{item.matched_symptoms.join(", ") || "No symptoms matched"}</p></div>)}</section>}
        </div>}
      </section>
      <section className="personalPanel" id="personal-records">
        <div className="panelTitle"><span className="step">05</span><div><h2>Personal health records</h2><p>Keep laboratory reports on this device, review extracted values, and compare changes over time.</p></div></div>
        <div className="personalToolbar">
          <form className="personalUpload" onSubmit={uploadPersonalReport}>
            <label htmlFor="personalReport">Upload report <span>PDF, PNG, JPG, WEBP · up to 10 MB</span></label>
            <div className="uploadControls"><input id="personalReport" type="file" accept="application/pdf,image/png,image/jpeg,image/webp" onChange={(event) => setPersonalFile(event.target.files?.[0] || null)} /><button className="primary compact" type="submit" disabled={personalUploading}>{personalUploading ? "Extracting…" : "Add personal report"}<span>↑</span></button></div>
          </form>
          <form className="personalSearch" onSubmit={(event) => { event.preventDefault(); loadPersonalRecords(personalSearch); }}><label htmlFor="personalSearch">Search reports</label><div className="searchControls"><input id="personalSearch" value={personalSearch} onChange={(event) => setPersonalSearch(event.target.value)} placeholder="Try sugar, Vitamin D, glucose…" /><button type="submit">Search</button></div></form>
        </div>
        {personalError && <p className="error">{personalError}</p>}
        <div className="personalLayout">
          <aside className="reportSidebar" aria-label="Personal reports by test type">
            <div className="sidebarHeader"><strong>Your reports</strong><span>{personalReports.length}</span></div>
            {personalLoading ? <p className="sidebarEmpty">Loading your local records…</p> : Object.keys(reportGroups).length === 0 ? <p className="sidebarEmpty">No reports yet. Upload a laboratory report to build your personal timeline.</p> : Object.entries(reportGroups).map(([type, reports]) => <section className="reportGroup" key={type}><h3>{type}<small>{reports.length}</small></h3>{reports.map((report) => <button key={report.id} className={`reportSelect ${selectedPersonalReport?.id === report.id ? "selected" : ""}`} onClick={() => setSelectedPersonalReport(report)}><strong>{report.reported_date || "Undated report"}</strong><span>{report.original_filename}</span><small>{report.measurements?.length || 0} extracted values</small></button>)}</section>)}
          </aside>
          <div className="personalContent">
            {!selectedPersonalReport ? <div className="personalEmpty"><div className="emptyIcon">▤</div><p>Choose a report to see its details.</p><small>Files and extracted data remain in local application storage.</small></div> : <>
              <section className="reportDetails"><div className="reportDetailsTitle"><div><p className="eyebrow">{selectedPersonalReport.test_type}</p><h3>{selectedPersonalReport.report_test_name || selectedPersonalReport.original_filename}</h3></div><span className="localBadge">Stored locally</span></div><div className="recordMeta"><span>Patient <strong>{selectedPersonalReport.patient_name || "Not found"}</strong></span><span>Doctor <strong>{selectedPersonalReport.doctor_name || "Not found"}</strong></span><span>Report date <strong>{selectedPersonalReport.reported_date || "Not found"}</strong></span><span>Extraction <strong>{selectedPersonalReport.extraction_method === "tesseract_ocr" ? "Image OCR" : "PDF text"}</strong></span></div>
                <h4>Extracted test details</h4>
                {selectedPersonalReport.measurements?.length ? <div className="measurementTable"><div className="measurementHead"><span>Test</span><span>Value</span><span>Reference range</span><span>Status</span></div>{selectedPersonalReport.measurements.map((measurement) => <div className="measurementRow" key={measurement.id || `${measurement.test_name}-${measurement.value_raw}`}><strong>{measurement.test_name}</strong><span>{measurement.value_raw} {measurement.unit || ""}</span><span>{measurement.reference_range || "Not available"}</span><b className={`measureStatus ${measurement.status}`}>{measurement.status}</b></div>)}</div> : <p className="noMeasurements">No value rows could be reliably extracted. The original report is still saved locally.</p>}
              </section>
              <section className="trendSection"><div className="trendTitle"><div><p className="eyebrow">Combined view</p><h3>Trends across matching reports</h3></div><span>{personalSummary?.total_reports || 0} reports · {personalSummary?.total_measurements || 0} values</span></div>{personalSummary?.metrics?.length ? <div className="trendGrid">{personalSummary.metrics.slice(0, 8).map((metric) => <article className="trendCard" key={`${metric.test_name}-${metric.unit || ""}`}><h4>{metric.test_name}</h4><p className={`trendDirection ${metric.direction.replaceAll(" ", "-")}`}>{metric.direction}</p><strong>{metric.latest?.value_raw} {metric.unit || ""}</strong><small>Latest · {metric.latest?.reported_date || "date unavailable"}</small><div><span>Lowest {metric.lowest?.value_raw ?? "—"}</span><span>Highest {metric.highest?.value_raw ?? "—"}</span></div>{metric.latest?.status !== "unknown" && <b className={`measureStatus ${metric.latest.status}`}>{metric.latest.status}</b>}</article>)}</div> : <p className="noMeasurements">Upload reports with numeric test results to generate a trend summary.</p>}</section>
              <section className="askRecords"><div><p className="eyebrow">Change explorer</p><h3>What changed in my tests?</h3><p>Ask a factual question such as “How has my blood sugar changed?”</p></div><form onSubmit={askAboutPersonalRecords}><input value={personalQuestion} onChange={(event) => setPersonalQuestion(event.target.value)} placeholder="Ask about a saved result…" /><button className="primary compact" type="submit" disabled={personalAsking}>{personalAsking ? "Comparing…" : "Show changes"}<span>→</span></button></form>{personalAnswer && <div className="recordAnswer"><strong>Local record summary</strong><p>{personalAnswer.answer}</p></div>}</section>
            </>}
          </div>
        </div>
      </section>
      <section className="webPanel">
        <div className="panelTitle"><span className="step">06</span><div><h2>Request a query report</h2><p>Generate a summary of the recent verification queries and their verdicts.</p></div></div>
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
