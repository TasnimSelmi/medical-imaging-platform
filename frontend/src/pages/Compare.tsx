import { useState, useEffect } from "react";
import { useSearchParams } from "react-router-dom";
import { BarChart, Bar, Cell, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts";
import { toast } from "sonner";
import { comparemethods, imageUrl, type CompareResult } from "../api/client";
import { useSession } from "../store/useSession";

const ALL_METHODS = [
  "otsu", "adaptive_threshold", "watershed", "chan_vese", "active_contour",
  "kmeans", "meanshift", "region_growing", "unet",
];

const METHOD_LABELS: Record<string, string> = {
  otsu: "Otsu",
  adaptive_threshold: "Seuillage adaptatif",
  watershed: "Ligne de partage des eaux",
  chan_vese: "Chan-Vese",
  active_contour: "Contour actif géodésique",
  kmeans: "K-means",
  meanshift: "Mean Shift",
  region_growing: "Croissance de régions",
  unet: "U-Net (Deep Learning)",
};

const label = (m: string) => METHOD_LABELS[m] ?? m;

export default function Compare() {
  const [params] = useSearchParams();
  const { groundTruthId, modality } = useSession();
  const imageId = params.get("image_id");
  const [results, setResults] = useState<CompareResult[]>([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!imageId) return;
    setLoading(true);
    const methods = modality === "endoscopy" ? ALL_METHODS : ALL_METHODS.filter((m) => m !== "unet");
    comparemethods(imageId, methods, groundTruthId ?? undefined, modality)
      .then((res) => {
        setResults(res.results);
        if (modality === "endoscopy" && !res.results.some((r) => r.method === "unet")) {
          toast.warning("Modèle U-Net non chargé : comparaison limitée aux méthodes classiques");
        }
      })
      .catch(() => toast.error("Erreur de comparaison"))
      .finally(() => setLoading(false));
  }, [imageId, groundTruthId, modality]);

  const chartData = results
    .filter((r) => r.metrics)
    .map((r) => ({ method: label(r.method), Dice: r.metrics!.dice, IoU: r.metrics!.iou, dl: r.method === "unet" }));

  const sorted = [...results].sort((a, b) => (b.metrics?.dice ?? 0) - (a.metrics?.dice ?? 0));

  if (!imageId) {
    return (
      <div className="flex items-center justify-center p-8">
        <span className="text-sm text-text-secondary">Aucune image sélectionnée</span>
      </div>
    );
  }

  return (
    <div className="p-4" style={{ minHeight: "calc(100vh - 48px)" }}>
      <h2 className="text-sm font-semibold text-text mb-4">Comparaison multi-méthodes</h2>

      {loading && <div className="h-1 bg-border rounded overflow-hidden mb-4"><div className="h-full bg-accent animate-pulse w-1/2" /></div>}

      <div className="grid grid-cols-3 gap-3 mb-6">
        <div className="border border-border rounded-md overflow-hidden">
          <div className="px-3 py-1.5 text-xs text-text-secondary border-b border-border">Original</div>
          <img src={imageUrl(imageId)} alt="original" className="w-full object-contain" />
        </div>
        {groundTruthId && (
          <div className="border border-border rounded-md overflow-hidden">
            <div className="px-3 py-1.5 text-xs text-text-secondary border-b border-border">Ground Truth</div>
            <img src={imageUrl(groundTruthId)} alt="ground truth" className="w-full object-contain" />
          </div>
        )}
        {results.map((r) => (
          <div key={r.method} className={`border rounded-md overflow-hidden ${r.method === "unet" ? "border-accent" : "border-border"}`}>
            <div className="px-3 py-1.5 text-xs border-b border-border flex justify-between">
              <span className="text-text">{label(r.method)}</span>
              {r.metrics && (
                <span className="font-mono text-text-secondary">
                  D:{r.metrics.dice.toFixed(3)}
                </span>
              )}
            </div>
            <img src={imageUrl(r.mask_id)} alt={r.method} className="w-full object-contain" />
          </div>
        ))}
      </div>

      {chartData.length > 0 && (
        <div className="border border-border rounded-md p-4 bg-surface mb-6">
          <div className="text-xs font-medium text-text mb-3">Dice par méthode</div>
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e5e5e5" />
              <XAxis dataKey="method" tick={{ fontSize: 10 }} />
              <YAxis domain={[0, 1]} tick={{ fontSize: 10 }} />
              <Tooltip contentStyle={{ fontSize: 11 }} />
              <Bar dataKey="Dice" radius={[2, 2, 0, 0]}>
                {chartData.map((d) => (
                  <Cell key={d.method} fill={d.dl ? "#b45309" : "#1e40af"} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}

      {results.length > 0 && (
        <div className="border border-border rounded-md overflow-hidden bg-surface">
          <table className="w-full text-xs">
            <thead>
              <tr className="border-b border-border bg-bg">
                <th className="text-left px-3 py-2 font-medium text-text">Méthode</th>
                <th className="text-right px-3 py-2 font-medium text-text">Durée (ms)</th>
                {groundTruthId && <th className="text-right px-3 py-2 font-medium text-text">IoU</th>}
                {groundTruthId && <th className="text-right px-3 py-2 font-medium text-text">Dice</th>}
              </tr>
            </thead>
            <tbody>
              {sorted.map((r) => (
                <tr key={r.method} className="border-b border-border last:border-0">
                  <td className="px-3 py-2 text-text">{label(r.method)}</td>
                  <td className="px-3 py-2 text-right font-mono text-text-secondary">{r.duration_ms}</td>
                  {groundTruthId && <td className="px-3 py-2 text-right font-mono">{r.metrics?.iou?.toFixed(4) ?? "—"}</td>}
                  {groundTruthId && <td className="px-3 py-2 text-right font-mono">{r.metrics?.dice?.toFixed(4) ?? "—"}</td>}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
