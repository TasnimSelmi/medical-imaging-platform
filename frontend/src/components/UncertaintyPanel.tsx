import { imageUrl } from "../api/client";
import type { UncertaintyData } from "../store/useSession";

interface Props {
  imageId: string;
  groundTruthId?: string | null;
  data: UncertaintyData;
}

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

export default function UncertaintyPanel({ imageId, groundTruthId, data }: Props) {
  const color = confidenceColor(data.confidence);

  return (
    <div className="flex-1 flex flex-col gap-3 overflow-auto">
      <div className="flex items-center gap-3 px-1">
        <span
          className="inline-flex items-center gap-1.5 px-3 py-1 rounded text-xs font-medium text-white"
          style={{ backgroundColor: color }}
        >
          Confiance : {data.confidence.toFixed(2)}
        </span>
        <span className="text-xs text-text-secondary">
          Le modèle est <strong>{confidenceLabel(data.confidence)}</strong> sur cette prédiction.
        </span>
      </div>
      <div className="text-[10px] text-text-secondary px-1">
        {data.nPasses} passes · {data.durationMs} ms
      </div>

      <div className="grid grid-cols-2 gap-2">
        <Card label="Image originale" src={imageUrl(imageId)} />
        {groundTruthId && <Card label="Ground truth" src={imageUrl(groundTruthId)} />}
        <Card label="Prédiction moyenne" src={imageUrl(data.maskId)} />
        <Card label="Carte d'écart-type (σ)" src={imageUrl(data.stdMapId)} />
        <Card label="Carte d'entropie" src={imageUrl(data.entropyMapId)} />
      </div>
    </div>
  );
}

function Card({ label, src }: { label: string; src: string }) {
  return (
    <div className="border border-border rounded-md overflow-hidden bg-surface">
      <div className="px-2 py-1 text-[10px] text-text-secondary border-b border-border">{label}</div>
      <img src={src} alt={label} className="w-full object-contain" />
    </div>
  );
}
