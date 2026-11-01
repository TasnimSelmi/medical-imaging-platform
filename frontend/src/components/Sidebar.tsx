import { useState, useRef, useCallback } from "react";
import {
  Upload, Image, Sliders, Grid3X3, Cpu, Target, Scissors,
  Scan, Activity, BarChart3,
} from "lucide-react";
import { toast } from "sonner";
import { useSession } from "../store/useSession";
import {
  uploadImage, preprocess, segmentClassical, segmentDeep,
  segmentDeepMC, applyMorphology, detectEdges, evaluate,
  getHistogram, thumbnailUrl,
} from "../api/client";
import MethodPicker from "./MethodPicker";
import ParamSlider from "./ParamSlider";
import HistogramChart from "./HistogramChart";

const PREPROCESS_METHODS = [
  { key: "gaussian", label: "Filtre gaussien" },
  { key: "median", label: "Filtre médian" },
  { key: "mean", label: "Filtre moyenneur" },
  { key: "bilateral", label: "Filtre bilatéral" },
  { key: "nlm", label: "Non-local means" },
  { key: "clahe", label: "CLAHE" },
  { key: "histogram_eq", label: "Égalisation d'histogramme" },
  { key: "zscore", label: "Normalisation Z-score" },
  { key: "n4_bias", label: "Correction N4 (IRM)" },
];

const MORPHOLOGY_METHODS = [
  { key: "erosion", label: "Érosion" },
  { key: "dilation", label: "Dilatation" },
  { key: "opening", label: "Ouverture" },
  { key: "closing", label: "Fermeture" },
  { key: "gradient", label: "Gradient morphologique" },
];

const EDGE_METHODS = [
  { key: "sobel", label: "Sobel" },
  { key: "laplace", label: "Laplacien" },
  { key: "canny", label: "Canny" },
];

const CLASSICAL_METHODS = [
  { key: "otsu", label: "Otsu" },
  { key: "adaptive_threshold", label: "Seuillage adaptatif" },
  { key: "watershed", label: "Ligne de partage des eaux" },
  { key: "chan_vese", label: "Chan-Vese" },
  { key: "active_contour", label: "Contour actif géodésique" },
  { key: "kmeans", label: "K-means" },
  { key: "meanshift", label: "Mean Shift" },
  { key: "region_growing", label: "Croissance de régions" },
];

export default function Sidebar() {
  const {
    imageId, setImage, setCurrentResult, setMask, setActiveTab,
    addStep, loading, setLoading, modality, setModality,
    setGroundTruth, currentResultId, maskId, groundTruthId,
    setUncertaintyData, setMetrics,
  } = useSession();
  // Only the endoscopy U-Net is trained; no brain MRI model ships with the platform
  const dlAvailable = modality === "endoscopy";

  const fileRef = useRef<HTMLInputElement>(null);
  const gtRef = useRef<HTMLInputElement>(null);
  const [nPasses, setNPasses] = useState(30);
  const [histData, setHistData] = useState<Record<string, number[]> | null>(null);
  const [preprocessParams, setPreprocessParams] = useState<Record<string, number>>({});
  const [morphParams, setMorphParams] = useState<Record<string, number>>({});
  const [edgeParams, setEdgeParams] = useState<Record<string, number>>({});
  const [segParams, setSegParams] = useState<Record<string, number>>({});

  const handleUpload = useCallback(async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setLoading(true);
    try {
      const res = await uploadImage(file);
      setImage(res.image_id, res.width, res.height);
      toast.success("Image chargée");
      const hist = await getHistogram(res.image_id);
      setHistData(hist.histograms);
    } catch {
      toast.error("Erreur lors du chargement");
    } finally {
      setLoading(false);
    }
  }, [setImage, setLoading]);

  const handlePreprocess = useCallback(async (method: string) => {
    const srcId = currentResultId;
    if (!srcId) return;
    setLoading(true);
    try {
      const res = await preprocess(method, srcId, preprocessParams);
      setCurrentResult(res.result_id);
      setActiveTab("preprocessed");
      addStep({ type: "preprocess", method, params: { ...preprocessParams }, resultId: res.result_id, durationMs: res.duration_ms });
    } catch {
      toast.error("Erreur prétraitement");
    } finally {
      setLoading(false);
    }
  }, [currentResultId, preprocessParams, setCurrentResult, setActiveTab, addStep, setLoading]);

  const handleMorphology = useCallback(async (method: string) => {
    const srcId = currentResultId;
    if (!srcId) return;
    setLoading(true);
    try {
      const res = await applyMorphology(method, srcId, morphParams);
      setCurrentResult(res.result_id);
      setActiveTab("preprocessed");
      addStep({ type: "morphology", method, params: { ...morphParams }, resultId: res.result_id, durationMs: res.duration_ms });
    } catch {
      toast.error("Erreur morphologie");
    } finally {
      setLoading(false);
    }
  }, [currentResultId, morphParams, setCurrentResult, setActiveTab, addStep, setLoading]);

  const handleEdges = useCallback(async (method: string) => {
    const srcId = currentResultId;
    if (!srcId) return;
    setLoading(true);
    try {
      const res = await detectEdges(method, srcId, edgeParams);
      setCurrentResult(res.result_id);
      setActiveTab("preprocessed");
      addStep({ type: "edges", method, params: { ...edgeParams }, resultId: res.result_id, durationMs: res.duration_ms });
    } catch {
      toast.error("Erreur détection de contours");
    } finally {
      setLoading(false);
    }
  }, [currentResultId, edgeParams, setCurrentResult, setActiveTab, addStep, setLoading]);

  const handleClassicalSeg = useCallback(async (method: string) => {
    const srcId = currentResultId;
    if (!srcId) return;
    setLoading(true);
    try {
      const res = await segmentClassical(method, srcId, segParams);
      setMask(res.mask_id);
      addStep({ type: "segment-classical", method, params: { ...segParams }, resultId: res.mask_id, durationMs: res.duration_ms });
    } catch {
      toast.error("Erreur segmentation");
    } finally {
      setLoading(false);
    }
  }, [currentResultId, segParams, setMask, addStep, setLoading]);

  const handleDeepSeg = useCallback(async () => {
    const srcId = currentResultId;
    if (!srcId) return;
    setLoading(true);
    try {
      const res = await segmentDeep(srcId, modality);
      setMask(res.mask_id);
      addStep({ type: "segment-deep", method: "U-Net", params: {}, resultId: res.mask_id, durationMs: res.duration_ms });
    } catch {
      toast.error("Modèle non disponible");
    } finally {
      setLoading(false);
    }
  }, [currentResultId, modality, setMask, addStep, setLoading]);

  const handleMCDropout = useCallback(async () => {
    const srcId = currentResultId;
    if (!srcId) return;
    setLoading(true);
    try {
      const res = await segmentDeepMC(srcId, modality, nPasses);
      setMask(res.mask_id);
      setUncertaintyData({
        maskId: res.mask_id,
        stdMapId: res.std_map_id,
        entropyMapId: res.entropy_map_id,
        confidence: res.confidence_score,
        nPasses: res.n_passes,
        durationMs: res.duration_ms,
      });
      setActiveTab("uncertainty");
      addStep({ type: "segment-mc", method: "U-Net MC Dropout", params: { n_passes: nPasses }, resultId: res.mask_id, durationMs: res.duration_ms });
    } catch {
      toast.error("Modèle non disponible");
    } finally {
      setLoading(false);
    }
  }, [currentResultId, modality, nPasses, setMask, setUncertaintyData, setActiveTab, addStep, setLoading]);

  const handleGtUpload = useCallback(async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    try {
      const res = await uploadImage(file);
      setGroundTruth(res.image_id);
      toast.success("Ground truth chargé");
    } catch {
      toast.error("Erreur chargement GT");
    }
  }, [setGroundTruth]);

  const handleEvaluate = useCallback(async () => {
    if (!maskId || !groundTruthId) return;
    setLoading(true);
    try {
      const res = await evaluate(maskId, groundTruthId);
      setMetrics(res);
      toast.success(`Dice: ${res.dice.toFixed(4)} — IoU: ${res.iou.toFixed(4)}`);
    } catch {
      toast.error("Erreur évaluation");
    } finally {
      setLoading(false);
    }
  }, [maskId, groundTruthId, setLoading, setMetrics]);

  return (
    <aside className="w-[280px] shrink-0 border-r border-border bg-surface overflow-y-auto p-4 flex flex-col gap-5">
      {/* Image (TP1) */}
      <section>
        <SectionTitle icon={<Image size={16} strokeWidth={1.5} />} title="Image" />
        {imageId && (
          <img src={thumbnailUrl(imageId)} alt="" className="w-full rounded border border-border mb-2" />
        )}
        <input ref={fileRef} type="file" accept="image/*,.dcm,.nii,.nii.gz" className="hidden" onChange={handleUpload} />
        <button onClick={() => fileRef.current?.click()} className="btn-outline w-full flex items-center justify-center gap-2">
          <Upload size={14} strokeWidth={1.5} />
          <span>{imageId ? "Réimporter" : "Charger une image"}</span>
        </button>
        {histData && <HistogramChart data={histData} />}
      </section>

      {imageId && (
        <>
          {/* Prétraitement (TP2) */}
          <section>
            <SectionTitle icon={<Sliders size={16} strokeWidth={1.5} />} title="Prétraitement" />
            <div className="mb-2 flex flex-col gap-1">
              <ParamSlider label="Sigma" param="sigma" value={preprocessParams.sigma ?? 1} min={0.1} max={10} step={0.1} onChange={(v) => setPreprocessParams((p) => ({ ...p, sigma: v }))} />
              <ParamSlider label="Taille noyau" param="ksize" value={preprocessParams.ksize ?? 5} min={3} max={21} step={2} onChange={(v) => setPreprocessParams((p) => ({ ...p, ksize: v }))} />
              <ParamSlider label="Clip limit" param="clip_limit" value={preprocessParams.clip_limit ?? 2} min={0.5} max={10} step={0.5} onChange={(v) => setPreprocessParams((p) => ({ ...p, clip_limit: v }))} />
            </div>
            <MethodPicker methods={PREPROCESS_METHODS} onSelect={handlePreprocess} loading={loading} />
          </section>

          {/* Morphologie & Contours (TP2) */}
          <section>
            <SectionTitle icon={<Grid3X3 size={16} strokeWidth={1.5} />} title="Morphologie" />
            <div className="mb-2">
              <ParamSlider label="Taille noyau" param="ksize" value={morphParams.ksize ?? 3} min={3} max={15} step={2} onChange={(v) => setMorphParams((p) => ({ ...p, ksize: v }))} />
              <ParamSlider label="Itérations" param="iterations" value={morphParams.iterations ?? 1} min={1} max={10} step={1} onChange={(v) => setMorphParams((p) => ({ ...p, iterations: v }))} />
            </div>
            <MethodPicker methods={MORPHOLOGY_METHODS} onSelect={handleMorphology} loading={loading} />
          </section>

          <section>
            <SectionTitle icon={<Scan size={16} strokeWidth={1.5} />} title="Contours" />
            <div className="mb-2">
              <ParamSlider label="Low" param="low" value={edgeParams.low ?? 50} min={0} max={255} step={5} onChange={(v) => setEdgeParams((p) => ({ ...p, low: v }))} />
              <ParamSlider label="High" param="high" value={edgeParams.high ?? 150} min={0} max={255} step={5} onChange={(v) => setEdgeParams((p) => ({ ...p, high: v }))} />
            </div>
            <MethodPicker methods={EDGE_METHODS} onSelect={handleEdges} loading={loading} />
          </section>

          {/* Segmentation classique (TP2 + TP3) */}
          <section>
            <SectionTitle icon={<Scissors size={16} strokeWidth={1.5} />} title="Segmentation classique" />
            <div className="mb-2 flex flex-col gap-1">
              <ParamSlider label="K (clusters)" param="k" value={segParams.k ?? 3} min={2} max={8} step={1} onChange={(v) => setSegParams((p) => ({ ...p, k: v }))} />
              <ParamSlider label="Block size" param="block_size" value={segParams.block_size ?? 11} min={3} max={51} step={2} onChange={(v) => setSegParams((p) => ({ ...p, block_size: v }))} />
            </div>
            <MethodPicker methods={CLASSICAL_METHODS} onSelect={handleClassicalSeg} loading={loading} />
          </section>

          {/* Segmentation DL (TP4) */}
          <section>
            <SectionTitle icon={<Cpu size={16} strokeWidth={1.5} />} title="Segmentation DL" />
            <div className="flex items-center gap-2 mb-2">
              <select
                value={modality}
                onChange={(e) => setModality(e.target.value as "endoscopy" | "brain")}
                className="flex-1 text-xs border border-border rounded px-2 py-1 bg-surface"
              >
                <option value="endoscopy">Endoscopie</option>
                <option value="brain">IRM cérébrale</option>
              </select>
            </div>
            {!dlAvailable && (
              <p className="text-[10px] text-text-secondary mb-2">
                Modèle disponible pour l'endoscopie uniquement.
              </p>
            )}
            <button onClick={handleDeepSeg} disabled={loading || !dlAvailable} className="btn-accent w-full mb-2">
              U-Net
            </button>
            <div className="flex items-center gap-2 mb-2">
              <label className="text-[10px] text-text-secondary whitespace-nowrap">N passes</label>
              <input
                type="number"
                min={3}
                max={100}
                value={nPasses}
                onChange={(e) => setNPasses(parseInt(e.target.value) || 30)}
                className="w-16 text-xs border border-border rounded px-2 py-1 bg-surface font-mono"
              />
            </div>
            <button onClick={handleMCDropout} disabled={loading || !dlAvailable} className="btn-outline w-full flex items-center justify-center gap-2">
              <Activity size={14} strokeWidth={1.5} />
              U-Net + Incertitude (MC Dropout)
            </button>
          </section>

          {/* Évaluation */}
          <section>
            <SectionTitle icon={<Target size={16} strokeWidth={1.5} />} title="Évaluation" />
            <input ref={gtRef} type="file" accept="image/*" className="hidden" onChange={handleGtUpload} />
            <button onClick={() => gtRef.current?.click()} className="btn-outline w-full text-xs mb-2">
              Charger masque GT (optionnel)
            </button>
            {groundTruthId && maskId && (
              <button onClick={handleEvaluate} disabled={loading} className="btn-outline w-full flex items-center justify-center gap-2 text-xs">
                <BarChart3 size={14} strokeWidth={1.5} />
                Calculer métriques
              </button>
            )}
          </section>
        </>
      )}

      <style>{`
        .btn-outline {
          font-size: 0.75rem;
          padding: 6px 12px;
          border: 1px solid #e5e5e5;
          border-radius: 4px;
          background: #fff;
          cursor: pointer;
          transition: background 150ms ease;
        }
        .btn-outline:hover { background: #fafafa; }
        .btn-outline:disabled { opacity: 0.5; cursor: default; }
        .btn-accent:disabled { opacity: 0.5; cursor: default; }
        .btn-accent {
          font-size: 0.75rem;
          padding: 6px 12px;
          border: 1px solid #1e40af;
          border-radius: 4px;
          background: #1e40af;
          color: #fff;
          cursor: pointer;
          transition: opacity 150ms ease;
        }
        .btn-accent:hover { opacity: 0.9; }
        .btn-accent:disabled { opacity: 0.5; cursor: default; }
      `}</style>
    </aside>
  );
}

function SectionTitle({ icon, title }: { icon: React.ReactNode; title: string }) {
  return (
    <div className="flex items-center gap-2 mb-2">
      <span className="text-text-secondary">{icon}</span>
      <span className="text-xs font-medium text-text">{title}</span>
    </div>
  );
}
