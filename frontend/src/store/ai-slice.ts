import type { StateCreator } from "zustand";

export type AiClass = { id: number; name: string; color: string; locked?: boolean };
export type AnnotationMode = "binary" | "multiclass";
export type EditingMode = "joint" | "per_label";
export type AiLabelActions = {
  setMode: (mode: EditingMode) => Promise<void>;
  select: (id: number) => Promise<void>;
  add: (entry: AiClass) => Promise<void>;
  update: (id: number, changes: Partial<AiClass>) => Promise<void>;
  lock: (id: number, locked: boolean) => Promise<void>;
  remove: (id: number) => Promise<void>;
  clear: () => Promise<void>;
};

export type AiSessionSummary = {
  session_id: string;
  session_name: string;
  created_at: string;
  volume_path: string | null;
  volume_path_root: "data" | "session" | null;
  annotation_path: string | null;
  result_path: string | null;
  last_inference_ml_id: string | null;
  last_inference_at: string | null;
  model_id?: string;
  annotation_mode?: AnnotationMode;
  classes?: AiClass[];
  modality?: "ct" | "other";
  editing_mode?: EditingMode;
  active_label_id?: number | null;
  state_version?: number;
  label_edits?: Record<string, {
    context_id?: string;
    annotation_path?: string | null; mask_path?: string | null; prior_path?: string | null;
  }>;
};

export type AiActiveSession = {
  session_id: string;
  session_name: string;
  result_path?: string | null;
  result_model_id?: string | null;
};

export interface AiSlice {
  aiEnabled: boolean | null;
  aiSessions: AiSessionSummary[];
  aiActiveSession: AiActiveSession | null;
  aiModelId: string;
  aiAnnotationMode: AnnotationMode;
  aiEditingMode: EditingMode;
  aiActiveLabelId: number | null;
  aiBinaryState: AiSessionSummary | null;
  aiLabelActions: AiLabelActions | null;
  aiLabelSaving: boolean;
  aiLabelDrawingReady: boolean;
  aiClasses: AiClass[];
  aiModality: "ct" | "other";
  aiAutoUpdate: boolean;
  aiPredictionVisible: boolean;
  aiPredictionOpacity: number;
  aiRevision: number;
  aiClearRequest: number;
  aiBusy: boolean;
  aiError: string | null;
  setAiModel: (id: string, mode: AnnotationMode) => void;
  setAiClasses: (classes: AiClass[]) => void;
  setAiModality: (modality: "ct" | "other") => void;
  setAiAutoUpdate: (enabled: boolean) => void;
  setAiPredictionVisible: (visible: boolean) => void;
  setAiPredictionOpacity: (opacity: number) => void;
  markAiDrawingChanged: () => void;
  clearAiAnnotations: () => void;
  setAiBusy: (busy: boolean) => void;
  setAiError: (error: string | null) => void;
  setAiEnabled: (enabled: boolean | null) => void;
  setAiSessions: (sessions: AiSessionSummary[]) => void;
  setAiActiveSession: (session: AiActiveSession | null) => void;
}

export const createAiSlice: StateCreator<AiSlice> = (set) => ({
  aiEnabled: null,
  aiSessions: [],
  aiActiveSession: null,
  aiModelId: "",
  aiAnnotationMode: "binary",
  aiEditingMode: "joint",
  aiActiveLabelId: null,
  aiBinaryState: null,
  aiLabelActions: null,
  aiLabelSaving: false,
  aiLabelDrawingReady: true,
  aiClasses: [{ id: 1, name: "Class 1", color: "#b6ff00" }],
  aiModality: "ct",
  aiAutoUpdate: true,
  aiPredictionVisible: true,
  aiPredictionOpacity: 0.5,
  aiRevision: 0,
  aiClearRequest: 0,
  aiBusy: false,
  aiError: null,
  setAiModel: (aiModelId, aiAnnotationMode) => set({ aiModelId, aiAnnotationMode }),
  setAiClasses: (aiClasses) => set({ aiClasses }),
  setAiModality: (aiModality) => set({ aiModality }),
  setAiAutoUpdate: (aiAutoUpdate) => set({ aiAutoUpdate }),
  setAiPredictionVisible: (aiPredictionVisible) => set({ aiPredictionVisible }),
  setAiPredictionOpacity: (aiPredictionOpacity) => set({ aiPredictionOpacity }),
  markAiDrawingChanged: () => set((s) => ({ aiRevision: s.aiRevision + 1 })),
  clearAiAnnotations: () => set((s) => ({ aiClearRequest: s.aiClearRequest + 1 })),
  setAiBusy: (aiBusy) => set({ aiBusy }),
  setAiError: (aiError) => set({ aiError }),
  setAiEnabled: (aiEnabled) => set({ aiEnabled }),
  setAiSessions: (aiSessions) => set({ aiSessions }),
  setAiActiveSession: (aiActiveSession) => set({ aiActiveSession, aiRevision: 0 }),
});
