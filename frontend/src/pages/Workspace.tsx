import { useEffect } from "react";
import { useSearchParams, useNavigate } from "react-router-dom";
import { Clock, RotateCcw, Download, BarChart3 } from "lucide-react";
import { useSession } from "../store/useSession";
import { downloadUrl } from "../api/client";
import Sidebar from "../components/Sidebar";
import ImageViewer from "../components/ImageViewer";
import OverlayViewer from "../components/OverlayViewer";
import MetricsTable from "../components/MetricsTable";
import UncertaintyPanel from "../components/UncertaintyPanel";

const TABS = ["original", "preprocessed", "mask", "overlay", "uncertainty"] as const;
const TAB_LABELS: Record<string, string> = {
  original: "Original",
  preprocessed: "Prétraité",
  mask: "Masque",
  overlay: "Overlay",
  uncertainty: "Incertitude",
};

export default function Workspace() {
  const [params] = useSearchParams();
  const navigate = useNavigate();
  const {
    imageId, currentResultId, maskId, groundTruthId,
    activeTab, setActiveTab, pipeline, revertToStep,
    setModality, loading, uncertaintyData, metrics,
  } = useSession();

  useEffect(() => {
    const mod = params.get("modality");
    if (mod === "brain" || mod === "endoscopy") {
      setModality(mod);
    }
  }, [params, setModality]);

  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLSelectElement) return;
      if (e.key === "o") setActiveTab("overlay");
      if (e.key === "m") setActiveTab("mask");
      if (e.key === "r") setActiveTab("original");
      if (e.key === "u") setActiveTab("uncertainty");
      const num = parseInt(e.key);
      if (num >= 1 && num <= pipeline.length) {
        revertToStep(pipeline[num - 1].id);
      }
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [setActiveTab, revertToStep, pipeline]);

  const viewerId = activeTab === "original" ? imageId
    : activeTab === "mask" ? maskId
    : activeTab === "preprocessed" ? currentResultId
    : null;

  return (
    <div className="flex" style={{ height: "calc(100vh - 48px)" }}>
      <Sidebar />

      <main className="flex-1 flex flex-col min-w-0 p-4 gap-2">
        <div className="flex gap-1">
          {TABS.map((tab) => (
            <button
              key={tab}
              onClick={() => setActiveTab(tab)}
              className={`px-3 py-1 text-xs rounded border transition-colors duration-150 ${
                activeTab === tab
                  ? "border-accent text-accent bg-blue-50"
                  : "border-border text-text-secondary bg-surface hover:bg-bg"
              }`}
            >
              {TAB_LABELS[tab]}
            </button>
          ))}
        </div>

        {activeTab === "overlay" && imageId && maskId ? (
          <OverlayViewer baseId={imageId} maskId={maskId} groundTruthId={groundTruthId} />
        ) : activeTab === "uncertainty" && imageId && uncertaintyData ? (
          <UncertaintyPanel imageId={imageId} groundTruthId={groundTruthId} data={uncertaintyData} />
        ) : (
          <ImageViewer imageId={viewerId} label={TAB_LABELS[activeTab]} />
        )}

        {loading && (
          <div className="h-1 bg-border rounded overflow-hidden">
            <div className="h-full bg-accent animate-pulse w-1/2" />
          </div>
        )}
      </main>

      <aside className="w-[320px] shrink-0 border-l border-border bg-surface overflow-y-auto p-4 flex flex-col">
        <div className="flex items-center gap-2 mb-3">
          <Clock size={16} strokeWidth={1.5} className="text-text-secondary" />
          <span className="text-xs font-medium text-text">Historique du pipeline</span>
        </div>

        <div className="flex-1 flex flex-col gap-2 overflow-y-auto">
          {pipeline.length === 0 && (
            <span className="text-xs text-text-secondary">Aucune étape appliquée</span>
          )}
          {pipeline.map((step, i) => (
            <div key={step.id} className="border border-border rounded p-3 bg-bg">
              <div className="flex items-center justify-between mb-1">
                <span className="text-xs font-medium text-text">
                  {i + 1}. {step.method}
                </span>
                <span className="text-[10px] font-mono text-text-secondary">
                  {step.durationMs}ms
                </span>
              </div>
              <div className="text-[10px] text-text-secondary mb-2">{step.type}</div>
              {Object.keys(step.params).length > 0 && (
                <div className="text-[10px] text-text-secondary mb-2 font-mono">
                  {Object.entries(step.params).map(([k, v]) => `${k}=${v}`).join(", ")}
                </div>
              )}
              <button
                onClick={() => revertToStep(step.id)}
                className="flex items-center gap-1 text-[10px] text-accent hover:underline"
              >
                <RotateCcw size={10} strokeWidth={1.5} />
                Revenir ici
              </button>
            </div>
          ))}
        </div>

        {metrics && (
          <div className="mt-3 border border-border rounded p-3 bg-bg">
            <div className="text-xs font-medium text-text mb-2">Métriques</div>
            <MetricsTable metrics={metrics} />
          </div>
        )}

        <div className="mt-4 flex flex-col gap-2">
          {maskId && (
            <a
              href={downloadUrl(maskId)}
              download
              className="flex items-center justify-center gap-2 px-3 py-1.5 text-xs border border-border rounded bg-surface hover:bg-bg transition-colors duration-150"
            >
              <Download size={14} strokeWidth={1.5} />
              Exporter résultat
            </a>
          )}
          {imageId && (
            <button
              onClick={() => navigate(`/compare?image_id=${imageId}`)}
              className="flex items-center justify-center gap-2 px-3 py-1.5 text-xs border border-accent text-accent rounded hover:bg-blue-50 transition-colors duration-150"
            >
              <BarChart3 size={14} strokeWidth={1.5} />
              Comparer les méthodes
            </button>
          )}
        </div>
      </aside>
    </div>
  );
}
