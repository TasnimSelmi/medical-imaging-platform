import { useSession } from "../store/useSession";
import { imageUrl } from "../api/client";

function confidenceColor(c: number): string {
  if (c >= 0.8) return "#16a34a";
  if (c >= 0.6) return "#ea580c";
  return "#dc2626";
}

function confidenceLabel(c: number): string {
  if (c >= 0.8) return "confiant";
  if (c >= 0.6) return "modérément confiant";
  return "incertain";
}

export default function Uncertainty() {
  const { imageId, groundTruthId, uncertaintyData } = useSession();

  if (!uncertaintyData || !imageId) {
    return (
      <div className="flex items-center justify-center p-8" style={{ minHeight: "calc(100vh - 48px)" }}>
        <div className="text-center">
          <p className="text-sm text-text-secondary mb-2">
            Aucune donnée d'incertitude disponible.
          </p>
          <p className="text-xs text-text-secondary">
            Lancez une segmentation MC Dropout depuis l'espace de travail.
          </p>
        </div>
      </div>
    );
  }

  const color = confidenceColor(uncertaintyData.confidence);

  return (
    <div className="p-4" style={{ minHeight: "calc(100vh - 48px)" }}>
      <h2 className="text-sm font-semibold text-text mb-1">
        Quantification d'incertitude — MC Dropout
      </h2>
      <p className="text-xs text-text-secondary mb-4">
        {uncertaintyData.nPasses} passes stochastiques · {uncertaintyData.durationMs} ms
      </p>

      <div className="flex items-center gap-3 mb-6">
        <span
          className="inline-flex items-center px-3 py-1 rounded text-xs font-medium text-white"
          style={{ backgroundColor: color }}
        >
          Confiance : {uncertaintyData.confidence.toFixed(2)}
        </span>
        <span className="text-xs text-text-secondary">
          Le modèle est <strong>{confidenceLabel(uncertaintyData.confidence)}</strong> sur cette prédiction.
        </span>
      </div>

      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3 mb-6">
        <Card label="Image originale" src={imageUrl(imageId)} />
        {groundTruthId ? (
          <Card label="Ground truth" src={imageUrl(groundTruthId)} />
        ) : (
          <div className="border border-border rounded-md flex items-center justify-center bg-bg p-4">
            <span className="text-[10px] text-text-secondary">Pas de GT</span>
          </div>
        )}
        <Card label="Prédiction moyenne" src={imageUrl(uncertaintyData.maskId)} />
        <Card label="Carte d'écart-type (σ)" src={imageUrl(uncertaintyData.stdMapId)} />
      </div>

      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
        <Card label="Carte d'entropie" src={imageUrl(uncertaintyData.entropyMapId)} />
      </div>

      <div className="mt-6 border border-border rounded-md p-4 bg-surface">
        <h3 className="text-xs font-medium text-text mb-2">Interprétation</h3>
        <ul className="text-xs text-text-secondary space-y-1 list-disc pl-4">
          <li>
            La <strong>carte d'écart-type</strong> montre l'incertitude épistémique pixel par pixel.
            Les zones claires correspondent aux régions où le modèle hésite.
          </li>
          <li>
            La <strong>carte d'entropie</strong> mesure l'incertitude de la distribution prédictive.
            Valeur maximale = log(2) ≈ 0.693 pour une prédiction à 50/50.
          </li>
          <li>
            Le <strong>score de confiance</strong> résume la certitude globale sur la région segmentée.
          </li>
        </ul>
      </div>
    </div>
  );
}

function Card({ label, src }: { label: string; src: string }) {
  return (
    <div className="border border-border rounded-md overflow-hidden bg-surface">
      <div className="px-2 py-1.5 text-[10px] text-text-secondary border-b border-border">{label}</div>
      <img src={src} alt={label} className="w-full object-contain" />
    </div>
  );
}
