import { useState } from "react";
import StructureModal from "./StructureModal";
import "./App.css";

const API_BASE = "http://127.0.0.1:8001";

export default function App() {
  const [mode, setMode] = useState("dataset"); // "dataset" | "elements" | "formulas"
  const [datasetN, setDatasetN] = useState(100);
  const [elementsInput, setElementsInput] = useState("Ga,N,Al,In,O");
  const [formulasInput, setFormulasInput] = useState("GaN\nAl2O3\nGa2O3\nInP");
  const [maxStoich, setMaxStoich] = useState(3);
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [selectedFormula, setSelectedFormula] = useState(null);
  const [sortKey, setSortKey] = useState("stability_proba");

  async function runScreen() {
    setLoading(true);
    setError(null);
    setResults(null);
    try {
      let body;
      if (mode === "dataset") body = { dataset_first_n: Number(datasetN) };
      else if (mode === "elements")
        body = { elements: elementsInput.split(",").map((s) => s.trim()).filter(Boolean), max_stoich: Number(maxStoich) };
      else body = { formulas: formulasInput.split("\n").map((s) => s.trim()).filter(Boolean) };

      const res = await fetch(`${API_BASE}/api/screen`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || "Request failed");
      }
      const data = await res.json();
      setResults(data.results);
      // dataset mode returns real materials ranked for solar suitability
      setSortKey(mode === "dataset" ? "solar_score" : "stability_proba");
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }

  const isSolar = results && results.some((r) => r.solar_score != null);

  const sorted = results
    ? [...results].sort((a, b) => {
        if (sortKey === "formation_energy_pred_ev_atom") return a[sortKey] - b[sortKey];
        if (sortKey === "solar_score") return (b.solar_score ?? 0) - (a.solar_score ?? 0);
        return b[sortKey] - a[sortKey];
      })
    : null;

  const modes = [
    { id: "dataset", label: "Dataset sample" },
    { id: "elements", label: "From elements" },
    { id: "formulas", label: "Custom formulas" },
  ];

  return (
    <div className="sheet">
      <header className="masthead">
        <div className="masthead-rule">
          <span>MATERIALS&nbsp;INFORMATICS</span>
          <span>COMPOSITION-ONLY&nbsp;ML</span>
        </div>
        <div className="title-row">
          <svg className="logo" viewBox="0 0 64 64" aria-hidden="true">
            {/* benzene ring — hexagonal carbon ring with aromatic inner circle */}
            <polygon points="32,6 54,19 54,45 32,58 10,45 10,19"
                     fill="none" stroke="var(--accent)" strokeWidth="3" strokeLinejoin="round" />
            <circle cx="32" cy="32" r="13" fill="none" stroke="var(--accent)" strokeWidth="2.5" />
            {/* six vertex atoms */}
            <g fill="var(--ink)">
              <circle cx="32" cy="6" r="3.2" /><circle cx="54" cy="19" r="3.2" />
              <circle cx="54" cy="45" r="3.2" /><circle cx="32" cy="58" r="3.2" />
              <circle cx="10" cy="45" r="3.2" /><circle cx="10" cy="19" r="3.2" />
            </g>
          </svg>
          <h1>Material Screener</h1>
        </div>
        <p className="lede">
          Predicts <em>formation energy</em>, <em>thermodynamic stability</em>, and the{" "}
          <em>metal / semiconductor / insulator</em> class for any chemical formula. Select an entry
          to inspect its measured crystal structure, where one exists in the Materials Project.
        </p>
        <div className="masthead-stats">
          <div><span className="stat-k">Formation energy</span><span className="stat-v">R² 0.955 · MAE 0.112 eV/atom</span></div>
          <div><span className="stat-k">Stability</span><span className="stat-v">ROC-AUC 0.977 · 94.3% acc.</span></div>
          <div><span className="stat-k">Semiconductor class</span><span className="stat-v">84.1% acc. (3-class)</span></div>
        </div>
      </header>

      <section className="block">
        <div className="block-label"><span className="num">01</span> Input</div>

        <div className="seg">
          {modes.map((m) => (
            <button key={m.id} className={mode === m.id ? "seg-on" : ""} onClick={() => setMode(m.id)}>
              {m.label}
            </button>
          ))}
        </div>

        {mode === "dataset" && (
          <div className="field-row">
            <label className="field">
              <span className="field-k">Top N solar-absorber semiconductors</span>
              <input type="number" min="1" max="200" value={datasetN} onChange={(e) => setDatasetN(e.target.value)} />
            </label>
            <p className="field-note">
              Ranks real MP materials for single-junction solar cells: measured band gap near the
              1.34 eV optimum (PBE-corrected), direct-gap bonus, stability, and earth-abundant /
              non-toxic elements. Every entry has a viewable measured structure.
            </p>
          </div>
        )}

        {mode === "elements" && (
          <div className="field-row">
            <label className="field grow">
              <span className="field-k">Element set</span>
              <input value={elementsInput} onChange={(e) => setElementsInput(e.target.value)} placeholder="Ga,N,Al,In,O" />
            </label>
            <label className="field">
              <span className="field-k">Max stoichiometry</span>
              <input type="number" min="1" max="6" value={maxStoich} onChange={(e) => setMaxStoich(e.target.value)} />
            </label>
          </div>
        )}

        {mode === "formulas" && (
          <div className="field-row">
            <label className="field grow">
              <span className="field-k">Formulas — one per line</span>
              <textarea rows={4} value={formulasInput} onChange={(e) => setFormulasInput(e.target.value)} />
            </label>
          </div>
        )}

        <button className="run" onClick={runScreen} disabled={loading}>
          {loading ? "Computing…" : "Run screening →"}
        </button>

        {error && <div className="err">{error}</div>}
      </section>

      {sorted && (
        <section className="block">
          <div className="block-label">
            <span className="num">02</span> Results
            <span className="count">{sorted.length} entr{sorted.length !== 1 ? "ies" : "y"}</span>
            <span className="sorter">
              ordered by
              <select value={sortKey} onChange={(e) => setSortKey(e.target.value)}>
                {isSolar && <option value="solar_score">solar suitability</option>}
                <option value="stability_proba">stability</option>
                <option value="formation_energy_pred_ev_atom">formation energy</option>
              </select>
            </span>
          </div>

          <div className="grid">
            {sorted.map((r, i) => (
              <button
                key={r.formula}
                className={`spec ${r.is_promising ? "spec-flag" : ""}`}
                onClick={() => setSelectedFormula(r.formula)}
              >
                <div className="spec-top">
                  <span className="spec-idx">{String(i + 1).padStart(3, "0")}</span>
                  {r.is_promising && <span className="spec-tag">candidate</span>}
                </div>
                <div className="spec-formula">{r.formula}</div>
                {r.bandgap_class && (
                  <div className={`spec-class spec-class-${r.bandgap_class}`}>
                    {r.bandgap_class}
                    <i>{(r.bandgap_confidence * 100).toFixed(0)}%</i>
                  </div>
                )}
                <dl className="spec-data">
                  {r.solar_score != null && (
                    <>
                      <dt>band gap</dt>
                      <dd>{r.true_gap.toFixed(2)} <i>eV {r.is_gap_direct ? "· direct" : "· indirect"}</i></dd>
                    </>
                  )}
                  <dt>ΔHf</dt>
                  <dd>{r.formation_energy_pred_ev_atom != null ? r.formation_energy_pred_ev_atom.toFixed(3) : "—"} <i>eV/atom</i></dd>
                  <dt>P(stable)</dt>
                  <dd>{r.stability_proba != null ? (r.stability_proba * 100).toFixed(1) : "—"}<i>%</i></dd>
                </dl>
                {r.solar_score != null
                  ? <div className="meter solar"><span style={{ width: `${Math.min(r.solar_score / 1.25 * 100, 100)}%` }} /></div>
                  : <div className="meter"><span style={{ width: `${r.stability_proba * 100}%` }} /></div>}
                <div className="spec-foot">inspect structure →</div>
              </button>
            ))}
          </div>
        </section>
      )}

      {selectedFormula && (
        <StructureModal formula={selectedFormula} onClose={() => setSelectedFormula(null)} />
      )}
    </div>
  );
}
