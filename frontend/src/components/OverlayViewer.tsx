import { useState } from "react";
import { TransformWrapper, TransformComponent } from "react-zoom-pan-pinch";
import { imageUrl } from "../api/client";

interface Props {
  baseId: string;
  maskId: string;
  groundTruthId?: string | null;
}

export default function OverlayViewer({ baseId, maskId, groundTruthId }: Props) {
  const [opacity, setOpacity] = useState(0.5);

  return (
    <div className="flex-1 flex flex-col bg-bg border border-border rounded-md overflow-hidden">
      <div className="px-3 py-1.5 flex items-center gap-3 border-b border-border">
        <span className="text-xs text-text-secondary">Opacité</span>
        <input
          type="range"
          min={0}
          max={1}
          step={0.05}
          value={opacity}
          onChange={(e) => setOpacity(parseFloat(e.target.value))}
          className="flex-1 h-0.5 accent-accent"
        />
        <span className="text-xs font-mono text-text-secondary w-8">{Math.round(opacity * 100)}%</span>
      </div>
      <TransformWrapper>
        <TransformComponent wrapperClass="!w-full !h-full" contentClass="!w-full !h-full flex items-center justify-center">
          <div className="relative">
            <img src={imageUrl(baseId)} alt="base" className="max-w-full max-h-full object-contain" />
            <img
              src={imageUrl(maskId)}
              alt="mask"
              className="absolute inset-0 max-w-full max-h-full object-contain"
              style={{ opacity, mixBlendMode: "multiply", filter: "hue-rotate(0deg) saturate(3) brightness(1.2)" }}
            />
            {groundTruthId && (
              <img
                src={imageUrl(groundTruthId)}
                alt="ground truth"
                className="absolute inset-0 max-w-full max-h-full object-contain"
                style={{ opacity: opacity * 0.6, mixBlendMode: "screen" }}
              />
            )}
          </div>
        </TransformComponent>
      </TransformWrapper>
    </div>
  );
}
