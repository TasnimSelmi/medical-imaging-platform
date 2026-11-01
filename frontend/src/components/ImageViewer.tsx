import { TransformWrapper, TransformComponent } from "react-zoom-pan-pinch";
import { imageUrl } from "../api/client";

interface Props {
  imageId: string | null;
  label?: string;
}

export default function ImageViewer({ imageId, label }: Props) {
  if (!imageId) {
    return (
      <div className="flex-1 flex items-center justify-center bg-bg border border-border rounded-md">
        <span className="text-sm text-text-secondary">Aucune image chargée</span>
      </div>
    );
  }

  return (
    <div className="flex-1 flex flex-col bg-bg border border-border rounded-md overflow-hidden">
      {label && (
        <div className="px-3 py-1.5 text-xs text-text-secondary border-b border-border">
          {label}
        </div>
      )}
      <TransformWrapper>
        <TransformComponent wrapperClass="!w-full !h-full" contentClass="!w-full !h-full flex items-center justify-center">
          <img
            src={imageUrl(imageId)}
            alt="medical scan"
            className="max-w-full max-h-full object-contain"
          />
        </TransformComponent>
      </TransformWrapper>
    </div>
  );
}
