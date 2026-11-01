import axios from "axios";

const api = axios.create({
  baseURL: "/api",
});

export async function uploadImage(file: File) {
  const form = new FormData();
  form.append("file", file);
  const { data } = await api.post<{ image_id: string; width: number; height: number }>("/upload", form);
  return data;
}

export async function preprocess(method: string, imageId: string, params: Record<string, number> = {}) {
  const { data } = await api.post<{ result_id: string; duration_ms: number }>(`/preprocess/${method}`, {
    image_id: imageId,
    params,
  });
  return data;
}

export async function applyMorphology(method: string, imageId: string, params: Record<string, number> = {}) {
  const { data } = await api.post<{ result_id: string; duration_ms: number }>(`/morphology/${method}`, {
    image_id: imageId,
    params,
  });
  return data;
}

export async function detectEdges(method: string, imageId: string, params: Record<string, number> = {}) {
  const { data } = await api.post<{ result_id: string; duration_ms: number }>(`/edges/${method}`, {
    image_id: imageId,
    params,
  });
  return data;
}

export async function segmentClassical(method: string, imageId: string, params: Record<string, number> = {}) {
  const { data } = await api.post<{ mask_id: string; duration_ms: number; threshold?: number }>(
    `/segment/classical/${method}`,
    { image_id: imageId, params }
  );
  return data;
}

export async function segmentDeep(imageId: string, modality: "endoscopy" | "brain") {
  const { data } = await api.post<{ mask_id: string; duration_ms: number }>("/segment/deep/unet", {
    image_id: imageId,
    modality,
  });
  return data;
}

export interface MCDropoutResult {
  mask_id: string;
  std_map_id: string;
  entropy_map_id: string;
  confidence_score: number;
  n_passes: number;
  duration_ms: number;
}

export async function segmentDeepMC(imageId: string, modality: "endoscopy" | "brain", nPasses: number = 30) {
  const { data } = await api.post<MCDropoutResult>("/segment/deep/unet_mc", {
    image_id: imageId,
    modality,
    n_passes: nPasses,
  });
  return data;
}

export async function getHistogram(imageId: string) {
  const { data } = await api.get<{ histograms: Record<string, number[]> }>(`/preprocess/histogram/${imageId}`);
  return data;
}

export async function computeOpticalFlow(imageId1: string, imageId2: string) {
  const { data } = await api.post<{ flow_viz_id: string; mean_magnitude: number; duration_ms: number }>(
    "/registration/optical_flow",
    { image_id_1: imageId1, image_id_2: imageId2 }
  );
  return data;
}

export async function evaluate(maskPredId: string, maskTruthId: string) {
  const { data } = await api.post<{ iou: number; dice: number; pearson: number; mse: number; nmi: number }>(
    "/evaluate",
    { mask_pred_id: maskPredId, mask_truth_id: maskTruthId }
  );
  return data;
}

export interface CompareResult {
  method: string;
  mask_id: string;
  duration_ms: number;
  metrics?: { iou: number; dice: number };
}

export async function comparemethods(imageId: string, methods: string[], groundTruthId?: string, modality: "endoscopy" | "brain" = "endoscopy") {
  const { data } = await api.post<{ results: CompareResult[] }>("/compare", {
    image_id: imageId,
    methods,
    ground_truth_id: groundTruthId,
    modality,
  });
  return data;
}

export function imageUrl(id: string) {
  return `/api/image/${id}`;
}

export function thumbnailUrl(id: string) {
  return `/api/image/${id}/thumbnail`;
}

export function downloadUrl(id: string) {
  return `/api/download/${id}?format=png`;
}
