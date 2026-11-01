import { create } from "zustand";

export interface PipelineStep {
  id: string;
  type: "preprocess" | "morphology" | "edges" | "segment-classical" | "segment-deep" | "segment-mc";
  method: string;
  params: Record<string, number>;
  resultId: string;
  durationMs: number;
  timestamp: number;
}

export interface UncertaintyData {
  maskId: string;
  stdMapId: string;
  entropyMapId: string;
  confidence: number;
  nPasses: number;
  durationMs: number;
}

interface SessionState {
  imageId: string | null;
  imageWidth: number;
  imageHeight: number;
  currentResultId: string | null;
  maskId: string | null;
  groundTruthId: string | null;
  modality: "endoscopy" | "brain";
  activeTab: "original" | "preprocessed" | "mask" | "overlay" | "uncertainty";
  pipeline: PipelineStep[];
  loading: boolean;
  uncertaintyData: UncertaintyData | null;
  metrics: { iou: number; dice: number; pearson: number; mse: number; nmi: number } | null;

  setImage: (id: string, w: number, h: number) => void;
  setCurrentResult: (id: string) => void;
  setMask: (id: string) => void;
  setGroundTruth: (id: string | null) => void;
  setModality: (m: "endoscopy" | "brain") => void;
  setActiveTab: (tab: SessionState["activeTab"]) => void;
  addStep: (step: Omit<PipelineStep, "id" | "timestamp">) => void;
  revertToStep: (stepId: string) => void;
  setLoading: (v: boolean) => void;
  setUncertaintyData: (data: UncertaintyData | null) => void;
  setMetrics: (m: SessionState["metrics"]) => void;
  reset: () => void;
}

const initial = {
  imageId: null as string | null,
  imageWidth: 0,
  imageHeight: 0,
  currentResultId: null as string | null,
  maskId: null as string | null,
  groundTruthId: null as string | null,
  modality: "endoscopy" as const,
  activeTab: "original" as const,
  pipeline: [] as PipelineStep[],
  loading: false,
  uncertaintyData: null as UncertaintyData | null,
  metrics: null as SessionState["metrics"],
};

export const useSession = create<SessionState>((set) => ({
  ...initial,

  setImage: (id, w, h) =>
    set({ imageId: id, imageWidth: w, imageHeight: h, currentResultId: id, pipeline: [], maskId: null, activeTab: "original", uncertaintyData: null, metrics: null }),
  setCurrentResult: (id) => set({ currentResultId: id }),
  setMask: (id) => set({ maskId: id, activeTab: "overlay" }),
  setGroundTruth: (id) => set({ groundTruthId: id }),
  setModality: (m) => set({ modality: m }),
  setActiveTab: (tab) => set({ activeTab: tab }),
  addStep: (step) =>
    set((s) => ({
      pipeline: [
        ...s.pipeline,
        { ...step, id: crypto.randomUUID().slice(0, 8), timestamp: Date.now() },
      ],
    })),
  revertToStep: (stepId) =>
    set((s) => {
      const idx = s.pipeline.findIndex((p) => p.id === stepId);
      if (idx < 0) return s;
      const step = s.pipeline[idx];
      return {
        pipeline: s.pipeline.slice(0, idx + 1),
        currentResultId: step.resultId,
        activeTab: step.type === "preprocess" || step.type === "morphology" || step.type === "edges" ? "preprocessed" : "mask",
      };
    }),
  setLoading: (v) => set({ loading: v }),
  setUncertaintyData: (data) => set({ uncertaintyData: data }),
  setMetrics: (m) => set({ metrics: m }),
  reset: () => set(initial),
}));
