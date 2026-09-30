import { useEffect, useRef, useState } from "react";

const API_BASE = "http://127.0.0.1:8001";

export default function StructureModal({ formula, onClose }) {
  const viewerRef = useRef(null);
  const [state, setState] = useState({ loading: true, error: null, data: null });
  const [desc, setDesc] = useState({ loading: false, text: null });

  useEffect(() => {
    let cancelled = false;
    setState({ loading: true, error: null, data: null });
    setDesc({ loading: false, text: null });

    fetch(`${API_BASE}/api/structure/${encodeURIComponent(formula)}`)
      .then(async (res) => {
        if (!res.ok) {
          const body = await res.json().catch(() => ({}));
          throw new Error(body.detail || "Structure not available");
        }
        return res.json();
      })
      .then((data) => {
        if (cancelled) return;
        setState({ loading: false, error: null, data });
        // lazily fetch the Robocrystallographer description (slow, ~5s)
        setDesc({ loading: true, text: null });
        fetch(`${API_BASE}/api/description/${data.material_id}`)
          .then((r) => (r.ok ? r.json() : null))
          .then((d) => { if (!cancelled) setDesc({ loading: false, text: d?.description || null }); })
          .catch(() => { if (!cancelled) setDesc({ loading: false, text: null }); });
      })
      .catch((err) => {
        if (!cancelled) setState({ loading: false, error: err.message, data: null });
      });

    return () => { cancelled = true; };
  }, [formula]);

  useEffect(() => {
    if (!state.data || !viewerRef.current || !window.$3Dmol) return;

    // 3Dmol measures the container's size at creation time — if the modal
    // is still animating in (CSS transition), the div can report 0x0.
    // Deferring one frame ensures layout has settled first.
    const raf = requestAnimationFrame(() => {
      const el = viewerRef.current;
      if (!el) return;
      el.innerHTML = "";
      const viewer = window.$3Dmol.createViewer(el, { backgroundColor: "#0b0e14" });
      viewer.addModel(state.data.cif, "cif");
      viewer.setStyle({}, { sphere: { scale: 0.32 }, stick: { radius: 0.12 } });
      viewer.addUnitCell();
      viewer.zoomTo();
      viewer.render();
      viewer.resize();
    });

    return () => cancelAnimationFrame(raf);
  }, [state.data]);

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <div>
            <h2>{formula}</h2>
            {state.data?.name && <div className="modal-name">{state.data.name}</div>}
          </div>
          <button className="modal-close" onClick={onClose}>&times;</button>
        </div>

        {state.loading && <div className="modal-status">Fetching structure from Materials Project…</div>}

        {state.error && (
          <div className="modal-status error">
            <div className="modal-status-icon">⚠</div>
            <div>
              <strong>No 3D structure available</strong>
              <p>{state.error}</p>
              <p className="modal-hint">
                This formula isn't a known Materials Project entry, so there's no real, computed
                crystal structure to show — predictions for stability/formation energy are
                composition-only and don't imply a known 3D arrangement.
              </p>
            </div>
          </div>
        )}

        {state.data && (
          <>
            <div ref={viewerRef} className="viewer-3d" />
            {state.data.crystal_system && (
              <div className="modal-struct">
                <span>{state.data.crystal_system}</span>
                {state.data.spacegroup && (
                  <span>space group {state.data.spacegroup}
                    {state.data.sg_number ? ` (#${state.data.sg_number})` : ""}</span>
                )}
              </div>
            )}
            <div className="modal-footer">
              <span>material_id: <code>{state.data.material_id}</code></span>
              <span>{state.data.n_atoms} atoms in cell</span>
              <a href={state.data.mp_url} target="_blank" rel="noreferrer">View on Materials Project ↗</a>
            </div>
            {(desc.loading || desc.text) && (
              <div className="modal-desc">
                <div className="modal-desc-label">Structure description · Robocrystallographer</div>
                {desc.loading
                  ? <p className="modal-desc-loading">Analyzing bonding & generating description…</p>
                  : <p>{desc.text}</p>}
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}
