import type { NVImage } from "@niivue/niivue";
import { uint8ArrayToBase64 } from "@/lib/niivue-helpers";
import type { AiClass, AiSessionSummary, AnnotationMode, EditingMode } from "@/store/ai-slice";

export type AiSessionSettings = {
  model_id: string;
  annotation_mode: AnnotationMode;
  classes: AiClass[];
  modality: "ct" | "other";
  editing_mode?: EditingMode;
  active_label_id?: number | null;
  state_version?: number;
};

export async function postAiJson(url: string, body: unknown) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!response.ok) throw new Error(`${response.status}: ${await response.text()}`);
  return response.json();
}

export async function saveAiSettings(sessionId: string, settings: AiSessionSettings): Promise<AiSessionSummary> {
  return postAiJson(`/ai/session/${encodeURIComponent(sessionId)}/settings`, settings);
}

export async function encodeAiDrawing(volume: NVImage, bitmap: Uint8Array) {
  return uint8ArrayToBase64(await volume.saveToUint8Array("drawing.nii.gz", bitmap));
}

export async function uploadAiDrawing(volume: NVImage, bitmap: Uint8Array, sessionName: string) {
  const annotationPath = "annotations.nii.gz";
  const bytes = await volume.saveToUint8Array(annotationPath, bitmap);
  await postAiJson("/data/nii", {
    filename: `ai-sessions/${sessionName}/${annotationPath}`,
    data: uint8ArrayToBase64(bytes),
  });
  return annotationPath;
}
