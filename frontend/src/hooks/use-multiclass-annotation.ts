import { useCallback, useEffect, useRef } from "react";
import { NVImage, type Niivue } from "@niivue/niivue";
import { useFreeBrowseStore } from "@/store";
import { annotationColormap, BACKGROUND_LABEL, retainPredictionLabels } from "@/lib/ai-labels";
import {
  postAiJson, saveAiSettings, uploadAiDrawing, type AiSessionSettings,
} from "@/lib/ai-session-api";
import { LatestPredictionQueue } from "@/lib/latest-prediction-queue";
import { useBinaryEditing } from "@/hooks/use-binary-editing";

type PredictionInput = {
  volume: NVImage;
  bitmap: Uint8Array;
  sessionId: string;
  sessionName: string;
  revision: number;
  contextId: string;
  settings: AiSessionSettings;
};

export function useMulticlassAnnotation(nvRef: React.RefObject<Niivue | null>) {
  const binary = useBinaryEditing(nvRef);
  const session = useFreeBrowseStore((s) => s.aiActiveSession);
  const mode = useFreeBrowseStore((s) =>
    s.aiEditingMode === "per_label" ? "per_label" : s.aiAnnotationMode);
  const modelId = useFreeBrowseStore((s) => s.aiModelId);
  const classes = useFreeBrowseStore((s) => s.aiClasses);
  const modality = useFreeBrowseStore((s) => s.aiModality);
  const revision = useFreeBrowseStore((s) => s.aiRevision);
  const clearRequest = useFreeBrowseStore((s) => s.aiClearRequest);
  const autoUpdate = useFreeBrowseStore((s) => s.aiAutoUpdate);
  const opacity = useFreeBrowseStore((s) => s.aiPredictionOpacity);
  const visible = useFreeBrowseStore((s) => s.aiPredictionVisible);
  const volumeVersion = useFreeBrowseStore((s) => s.volumeVersion);
  const overlay = useRef<NVImage | null>(null);
  const sessionVolume = useRef<NVImage | null>(null);
  const handledClear = useRef(clearRequest);
  const deletionRevision = useRef(-1);
  const previousClasses = useRef({
    sessionId: session?.session_id, ids: classes.map((entry) => entry.id),
  });
  const predictionContext = useRef(crypto.randomUUID());
  const queue = useRef<LatestPredictionQueue<PredictionInput, NVImage> | null>(null);

  const removePrediction = useCallback(() => {
    const nv = nvRef.current;
    if (nv && overlay.current && nv.volumes.includes(overlay.current)) {
      nv.removeVolume(overlay.current);
    }
    overlay.current = null;
  }, [nvRef]);

  useEffect(() => {
    const predictionQueue = new LatestPredictionQueue<PredictionInput, NVImage>(
      async (input) => {
        const annotationPath = await uploadAiDrawing(input.volume, input.bitmap, input.sessionName);
        await postAiJson(`/ai/session/${input.sessionId}/set_annots`, {
          annotation_path: annotationPath,
        });
        await saveAiSettings(input.sessionId, input.settings);
        const response = await postAiJson(
          `/ai/session/${input.sessionId}/infer/${encodeURIComponent(input.settings.model_id)}`,
          {
            annotation_mode: "multiclass", modality: input.settings.modality,
            prediction_context_id: input.contextId,
          },
        );
        return NVImage.loadFromUrl({
          url: `/data/ai-sessions/${input.sessionName}/${response.result_path}?t=${Date.now()}`,
          name: "result.nii.gz", opacity: 0.5,
        });
      },
      (image, input) => {
        const nv = nvRef.current;
        const state = useFreeBrowseStore.getState();
        if (!nv || nv.volumes[0] !== input.volume || state.aiRevision !== input.revision ||
          state.aiActiveSession?.session_id !== input.sessionId ||
          state.aiModelId !== input.settings.model_id ||
          input.settings.classes.some((entry) =>
            !state.aiClasses.some(({ id }) => id === entry.id)) ||
          state.aiAnnotationMode !== "multiclass") return;
        removePrediction();
        image.setColormapLabel(annotationColormap(state.aiClasses, true));
        image.opacity = state.aiPredictionVisible ? state.aiPredictionOpacity : 0;
        overlay.current = image;
        nv.addVolume(image);
        state.incrementVolumeVersion();
      },
      (error) => useFreeBrowseStore.getState().setAiError(
        error instanceof Error ? error.message : String(error),
      ),
      (busy) => useFreeBrowseStore.getState().setAiBusy(busy),
    );
    queue.current = predictionQueue;
    return () => { predictionQueue.invalidate(); queue.current = null; };
  }, [nvRef, removePrediction]);

  const requestPrediction = useCallback(async () => {
    const nv = nvRef.current;
    const state = useFreeBrowseStore.getState();
    if (!state.aiLabelDrawingReady) return;
    if (state.aiEditingMode === "per_label") return binary.requestPrediction();
    const active = state.aiActiveSession;
    if (!nv?.drawBitmap || !nv.volumes[0] || !active ||
      state.aiAnnotationMode !== "multiclass") return;
    // Capture before any asynchronous work; later strokes must not change an in-flight request.
    const bitmap = nv.drawBitmap.slice();
    if (!state.aiRevision && !bitmap.some((value) => value !== 0)) return;
    state.setAiError(null);
    queue.current?.enqueue({
      volume: nv.volumes[0], bitmap, sessionId: active.session_id, sessionName: active.session_name,
      revision: state.aiRevision,
      contextId: predictionContext.current,
      settings: {
        model_id: state.aiModelId, annotation_mode: "multiclass",
        classes: state.aiClasses, modality: state.aiModality,
      },
    });
  }, [nvRef, binary.requestPrediction]);

  useEffect(() => {
    predictionContext.current = crypto.randomUUID();
    deletionRevision.current = -1;
    queue.current?.invalidate();
    removePrediction();
    sessionVolume.current = session ? nvRef.current?.volumes[0] ?? null : null;
    useFreeBrowseStore.getState().setAiError(null);
  }, [session, modelId, modality, mode, nvRef, removePrediction]);

  useEffect(() => {
    if (session && sessionVolume.current && nvRef.current?.volumes[0] !== sessionVolume.current) {
      queue.current?.invalidate();
      removePrediction();
      useFreeBrowseStore.getState().setAiActiveSession(null);
    }
  }, [volumeVersion, session, nvRef, removePrediction]);

  useEffect(() => {
    if (!session?.result_path || session.result_model_id !== modelId ||
      mode !== "multiclass") return;
    const nv = nvRef.current;
    if (!nv?.drawBitmap?.some((value) => value !== 0)) return;
    const savedRevision = useFreeBrowseStore.getState().aiRevision;
    const volume = nv.volumes[0];
    let cancelled = false;
    void NVImage.loadFromUrl({
      url: `/data/ai-sessions/${session.session_name}/${session.result_path}?t=${Date.now()}`,
      name: "result.nii.gz",
    }).then((image) => {
      const state = useFreeBrowseStore.getState();
      if (cancelled || state.aiRevision !== savedRevision ||
        nv.volumes[0] !== volume || overlay.current) return;
      retainPredictionLabels(image, state.aiClasses);
      image.setColormapLabel(annotationColormap(state.aiClasses, true));
      image.opacity = state.aiPredictionVisible ? state.aiPredictionOpacity : 0;
      overlay.current = image;
      nv.addVolume(image);
      state.incrementVolumeVersion();
    }).catch((error) => {
      if (!cancelled) useFreeBrowseStore.getState().setAiError(String(error));
    });
    return () => { cancelled = true; };
  }, [session, modelId, mode, nvRef]);

  useEffect(() => {
    if (!session || mode !== "multiclass") return;
    const nv = nvRef.current;
    if (!nv) return;
    const previous = nv.onMouseUp;
    const onMouseUp: typeof nv.onMouseUp = (event) => {
      previous?.(event);
      if (!event.mouseButtonRightDown && !event.mouseButtonCenterDown &&
        nv.opts.drawingEnabled && nv.drawBitmap) {
        useFreeBrowseStore.getState().markAiDrawingChanged();
      }
    };
    nv.onMouseUp = onMouseUp;
    return () => { if (nv.onMouseUp === onMouseUp) nv.onMouseUp = previous; };
  }, [session, mode, nvRef]);

  useEffect(() => {
    const previous = previousClasses.current;
    const ids = new Set(classes.map((entry) => entry.id));
    previousClasses.current = { sessionId: session?.session_id, ids: [...ids] };
    if (!session || mode !== "multiclass" || previous.sessionId !== session.session_id ||
      !previous.ids.some((id) => !ids.has(id))) return;
    predictionContext.current = crypto.randomUUID();
    queue.current?.invalidate();
    if (overlay.current) retainPredictionLabels(overlay.current, classes);
    const nv = nvRef.current;
    if (nv?.drawBitmap) {
      for (let index = 0; index < nv.drawBitmap.length; index++) {
        const value = nv.drawBitmap[index];
        if (value !== BACKGROUND_LABEL && !ids.has(value)) nv.drawBitmap[index] = 0;
      }
      // Undo snapshots contain deleted marks, so start a new history at the remaining drawing.
      nv.drawClearAllUndoBitmaps();
      nv.drawAddUndoBitmap();
      nv.refreshDrawing(true);
    }
    const state = useFreeBrowseStore.getState();
    deletionRevision.current = state.aiRevision + 1;
    state.markAiDrawingChanged();
  }, [classes, session, mode, nvRef]);

  useEffect(() => {
    if (!session || mode !== "multiclass") return;
    queue.current?.invalidate();
    // Deletion changes the drawing but must not schedule an inference, even in a batched edit.
    if (!autoUpdate || revision <= deletionRevision.current) return;
    const timer = window.setTimeout(() => { void requestPrediction(); }, 250);
    return () => window.clearTimeout(timer);
  }, [revision, autoUpdate, modelId, modality, session, mode, requestPrediction]);

  useEffect(() => {
    if (clearRequest === handledClear.current) return;
    handledClear.current = clearRequest;
    predictionContext.current = crypto.randomUUID();
    if (!session || mode !== "multiclass") return;
    const nv = nvRef.current;
    queue.current?.invalidate();
    removePrediction();
    if (nv?.drawBitmap) {
      nv.drawBitmap.fill(0);
      nv.drawAddUndoBitmap();
      nv.refreshDrawing(true);
      useFreeBrowseStore.getState().markAiDrawingChanged();
    }
  }, [clearRequest, session, mode, nvRef, removePrediction]);

  useEffect(() => {
    if (!session || mode !== "multiclass") return;
    const nv = nvRef.current;
    if (!nv) return;
    nv.setDrawColormap(annotationColormap(classes) as unknown as string);
    nv.setDrawOpacity(1);
    if (overlay.current) {
      overlay.current.setColormapLabel(annotationColormap(classes, true));
      overlay.current.opacity = visible ? opacity : 0;
      nv.updateGLVolume();
    }
  }, [classes, opacity, visible, session, mode, nvRef]);

  useEffect(() => {
    if (!session || !modelId || mode === "per_label") return;
    const timer = window.setTimeout(() => {
      const state = useFreeBrowseStore.getState();
      if (state.aiEditingMode !== "joint" || state.aiLabelSaving) return;
      void saveAiSettings(session.session_id, {
        model_id: modelId, annotation_mode: mode, classes, modality,
      }).catch((error) => useFreeBrowseStore.getState().setAiError(String(error)));
    }, 300);
    return () => window.clearTimeout(timer);
  }, [session, modelId, mode, classes, modality]);

  const cancelPendingPrediction = useCallback(() => queue.current?.invalidate(), []);
  return { requestPrediction, cancelPendingPrediction, saveLabelEdits: binary.save };
}
