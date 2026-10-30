import { useCallback, useEffect, useRef } from "react";
import { NVImage, type Niivue } from "@niivue/niivue";
import { useFreeBrowseStore } from "@/store";
import type { AiLabelActions, AiSessionSummary } from "@/store/ai-slice";
import { annotationColormap } from "@/lib/ai-labels";
import { encodeAiDrawing, postAiJson, uploadAiDrawing } from "@/lib/ai-session-api";

type History = { bitmap: Uint8Array; undo: Uint8Array[]; index: number };

export function useBinaryEditing(nvRef: React.RefObject<Niivue | null>) {
  const session = useFreeBrowseStore((s) => s.aiActiveSession);
  const mode = useFreeBrowseStore((s) => s.aiEditingMode);
  const revision = useFreeBrowseStore((s) => s.aiRevision);
  const classes = useFreeBrowseStore((s) => s.aiClasses);
  const opacity = useFreeBrowseStore((s) => s.aiPredictionOpacity);
  const visible = useFreeBrowseStore((s) => s.aiPredictionVisible);
  const overlay = useRef<NVImage | null>(null);
  const history = useRef(new Map<number, History>());
  const savedMarks = useRef(new Map<number, Uint8Array>());
  const serial = useRef(Promise.resolve());
  const pending = useRef(0);
  const inferenceGeneration = useRef(0);
  const prediction = useRef<Promise<void> | null>(null);
  const predictionKey = useRef<string | null>(null);
  const queuedPrediction = useRef<string | null>(null);
  const lastRequestedKey = useRef<string | null>(null);

  const currentPredictionKey = useCallback(() => {
    const state = useFreeBrowseStore.getState();
    const labelId = state.aiActiveLabelId;
    if (!state.aiActiveSession || !labelId || state.aiEditingMode !== "per_label" ||
      !state.aiLabelDrawingReady || state.aiClasses.find((item) => item.id === labelId)?.locked) {
      return null;
    }
    return JSON.stringify([
      state.aiActiveSession.session_id, state.aiModelId, state.aiModality, labelId,
      state.aiBinaryState?.label_edits?.[String(labelId)]?.context_id, state.aiRevision,
    ]);
  }, []);

  const removeOverlay = useCallback(() => {
    const nv = nvRef.current;
    if (nv && overlay.current && nv.volumes.includes(overlay.current)) nv.removeVolume(overlay.current);
    overlay.current = null;
  }, [nvRef]);

  const applyManifest = useCallback((manifest: AiSessionSummary) => {
    const state = useFreeBrowseStore.getState();
    if (state.aiActiveSession?.session_id !== manifest.session_id) return;
    useFreeBrowseStore.setState({
      aiBinaryState: manifest, aiEditingMode: manifest.editing_mode ?? "joint",
      aiActiveLabelId: manifest.active_label_id ?? null,
      aiClasses: JSON.stringify(state.aiClasses) === JSON.stringify(manifest.classes)
        ? state.aiClasses : manifest.classes ?? [],
    });
  }, []);

  const displayPrediction = useCallback(async (manifest: AiSessionSummary) => {
    if (!manifest.result_path) { removeOverlay(); return; }
    const nv = nvRef.current;
    const volume = nv?.volumes[0];
    const image = await NVImage.loadFromUrl({
      url: `/data/ai-sessions/${manifest.session_name}/${manifest.result_path}?t=${Date.now()}`,
      name: "result.nii.gz",
    });
    const state = useFreeBrowseStore.getState();
    if (!nv || nv.volumes[0] !== volume || state.aiEditingMode !== "per_label" ||
      state.aiActiveSession?.session_id !== manifest.session_id ||
      state.aiBinaryState?.state_version !== manifest.state_version) return;
    removeOverlay();
    image.setColormapLabel(annotationColormap(state.aiClasses, true));
    image.opacity = state.aiPredictionVisible ? state.aiPredictionOpacity : 0;
    overlay.current = image;
    nv.addVolume(image);
    state.incrementVolumeVersion();
  }, [nvRef, removeOverlay]);

  const runSerialized = useCallback(<T,>(action: () => Promise<T>): Promise<T> => {
    pending.current++;
    useFreeBrowseStore.setState({ aiLabelSaving: true });
    const result = serial.current.then(action);
    serial.current = result.then(() => undefined, () => undefined);
    return result.finally(() => {
      pending.current--;
      if (!pending.current) useFreeBrowseStore.setState({ aiLabelSaving: false });
    });
  }, []);

  const saveCurrent = useCallback(async (force = false) => {
    const state = useFreeBrowseStore.getState();
    const nv = nvRef.current;
    const active = state.aiActiveSession;
    const labelId = state.aiActiveLabelId;
    if (state.aiEditingMode === "per_label" && !state.aiLabelDrawingReady) {
      throw new Error("Select a label to reload its drawing before editing or saving");
    }
    if (state.aiEditingMode !== "per_label" || !active || !labelId || !nv?.drawBitmap ||
      state.aiClasses.find((item) => item.id === labelId)?.locked) return;
    const bitmap = nv.drawBitmap.slice();
    history.current.set(labelId, {
      bitmap, undo: nv.drawUndoBitmaps?.map((item) => item.slice()) ?? [],
      index: nv.currentDrawUndoBitmap,
    });
    const saved = savedMarks.current.get(labelId);
    if (!force && saved && saved.length === bitmap.length &&
      saved.every((value, i) => value === bitmap[i])) {
      return;
    }
    const data = await encodeAiDrawing(nv.volumes[0], bitmap);
    const saveDrawing = () => postAiJson(`/ai/session/${active.session_id}/binary/annotations`, {
      label_id: labelId, state_version: useFreeBrowseStore.getState().aiBinaryState?.state_version ?? 0,
      data,
    });
    let manifest: AiSessionSummary;
    try { manifest = await saveDrawing(); }
    catch (error) {
      if (!String(error).includes("409")) throw error;
      const response = await fetch(`/ai/session/${active.session_id}`);
      if (!response.ok) throw error;
      const latest: AiSessionSummary = await response.json();
      applyManifest(latest);
      if (latest.editing_mode !== "per_label" || latest.active_label_id !== labelId ||
        latest.classes?.find((item) => item.id === labelId)?.locked) {
        useFreeBrowseStore.setState({ aiLabelDrawingReady: false });
        nv.setDrawingEnabled(false);
        throw error;
      }
      manifest = await saveDrawing();
    }
    savedMarks.current.set(labelId, bitmap);
    applyManifest(manifest);
  }, [nvRef, applyManifest]);

  const saveSettings = useCallback(async (changes: Record<string, unknown> = {}, publish = true) => {
    const state = useFreeBrowseStore.getState();
    const active = state.aiActiveSession;
    if (!active) throw new Error("Open an AI session first");
    const manifest = await postAiJson(`/ai/session/${active.session_id}/settings`, {
      model_id: state.aiModelId, annotation_mode: state.aiAnnotationMode,
      classes: state.aiClasses, modality: state.aiModality, editing_mode: state.aiEditingMode,
      active_label_id: state.aiActiveLabelId, state_version: state.aiBinaryState?.state_version ?? 0,
      ...changes,
    });
    if (publish) applyManifest(manifest);
    return manifest as AiSessionSummary;
  }, [applyManifest]);

  const loadDrawing = useCallback(async (manifest: AiSessionSummary) => {
    const nv = nvRef.current;
    if (!nv || useFreeBrowseStore.getState().aiActiveSession?.session_id !== manifest.session_id) return;
    useFreeBrowseStore.setState({ aiLabelDrawingReady: false });
    const labelId = manifest.active_label_id;
    const cached = manifest.editing_mode === "per_label" && labelId
      ? history.current.get(labelId) : undefined;
    const path = manifest.editing_mode === "per_label" && labelId
      ? manifest.label_edits?.[String(labelId)]?.annotation_path : manifest.annotation_path;
    if (cached && nv.drawBitmap) {
      nv.drawBitmap.set(cached.bitmap);
      nv.drawUndoBitmaps = cached.undo.map((item) => item.slice());
      nv.currentDrawUndoBitmap = cached.index;
    } else if (path) {
      const response = await fetch(`/data/ai-sessions/${manifest.session_name}/${path}?t=${Date.now()}`);
      if (!response.ok) throw new Error("Could not load the saved label drawing");
      const image = await nv.niftiArray2NVImage(new Uint8Array(await response.arrayBuffer()));
      if (!nv.loadDrawing(image)) throw new Error("The saved label drawing does not match the scan");
    } else {
      if (!nv.drawBitmap) nv.createEmptyDrawing();
      nv.drawBitmap!.fill(0);
      nv.drawClearAllUndoBitmaps();
      nv.drawAddUndoBitmap();
    }
    if (manifest.editing_mode === "joint" && nv.drawBitmap) {
      const ids = new Set(manifest.classes?.map((item) => item.id));
      for (let index = 0; index < nv.drawBitmap.length; index++) {
        const value = nv.drawBitmap[index];
        if (value !== 255 && !ids.has(value)) nv.drawBitmap[index] = 0;
      }
      nv.drawClearAllUndoBitmaps();
      nv.drawAddUndoBitmap();
    }
    if (labelId && nv.drawBitmap) savedMarks.current.set(labelId, nv.drawBitmap.slice());
    applyManifest(manifest);
    const state = useFreeBrowseStore.getState();
    const penValue = manifest.editing_mode === "per_label" ? labelId ?? 255 : 255;
    nv.setPenValue(penValue, false);
    nv.setDrawingEnabled(!state.aiClasses.find((item) => item.id === labelId)?.locked);
    state.setDrawingOptions((previous) => ({ ...previous, penValue, mode: "pen" }));
    nv.refreshDrawing(true);
    useFreeBrowseStore.setState({ aiLabelDrawingReady: true });
  }, [nvRef, applyManifest]);

  const switchDrawing = useCallback(async (changes: Record<string, unknown>) => {
    const previous = useFreeBrowseStore.getState();
    const nv = nvRef.current;
    const bitmap = nv?.drawBitmap?.slice();
    const undo = nv?.drawUndoBitmaps?.map((item) => item.slice());
    const undoIndex = nv?.currentDrawUndoBitmap;
    const manifest = await saveSettings(changes, false);
    try {
      await loadDrawing(manifest);
      return manifest;
    } catch (error) {
      const rollback = await postAiJson(`/ai/session/${manifest.session_id}/settings`, {
        model_id: previous.aiModelId, annotation_mode: previous.aiAnnotationMode,
        classes: previous.aiClasses, modality: previous.aiModality,
        editing_mode: previous.aiEditingMode, active_label_id: previous.aiActiveLabelId,
        state_version: manifest.state_version,
      });
      applyManifest(rollback);
      if (nv && bitmap) {
        nv.drawBitmap = bitmap;
        if (undo) nv.drawUndoBitmaps = undo;
        if (undoIndex !== undefined) nv.currentDrawUndoBitmap = undoIndex;
        nv.setPenValue(previous.drawingOptions.penValue, false);
        previous.setDrawingOptions(() => previous.drawingOptions);
        nv.refreshDrawing(true);
      }
      useFreeBrowseStore.setState({ aiLabelDrawingReady: previous.aiLabelDrawingReady });
      throw error;
    }
  }, [nvRef, saveSettings, loadDrawing, applyManifest]);

  const requestPrediction = useCallback((): Promise<void> => {
    const key = currentPredictionKey();
    if (!key) return Promise.resolve();
    lastRequestedKey.current = key;
    if (prediction.current && predictionKey.current === key && !queuedPrediction.current) {
      return prediction.current;
    }
    queuedPrediction.current = key;
    if (prediction.current) return prediction.current;
    useFreeBrowseStore.getState().setAiBusy(true);
    const drain = async () => {
      let failure: unknown;
      while (queuedPrediction.current) {
        const next = queuedPrediction.current;
        queuedPrediction.current = null;
        predictionKey.current = next;
        const generation = inferenceGeneration.current;
        try {
          await runSerialized(async () => {
            if (next === currentPredictionKey()) await saveCurrent();
          });
          if (next !== currentPredictionKey()) continue;
          const state = useFreeBrowseStore.getState();
          const version = state.aiBinaryState?.state_version ?? 0;
          const manifest = await postAiJson(
            `/ai/session/${state.aiActiveSession!.session_id}/infer/${encodeURIComponent(state.aiModelId)}`, {
              annotation_mode: "multiclass", editing_mode: "per_label",
              active_label_id: state.aiActiveLabelId, state_version: version,
            },
          );
          const current = useFreeBrowseStore.getState();
          if (current.aiActiveSession?.session_id !== manifest.session_id ||
            current.aiBinaryState?.state_version !== version) continue;
          // Advance the server revision even if a newer local stroke superseded the result.
          applyManifest(manifest);
          if (generation !== inferenceGeneration.current || next !== currentPredictionKey()) continue;
          await displayPrediction(manifest);
          useFreeBrowseStore.getState().setAiError(null);
          failure = undefined;
        } catch (error) {
          if (generation === inferenceGeneration.current && next === currentPredictionKey()) {
            failure = error;
          }
        }
      }
      if (failure) throw failure;
    };
    prediction.current = drain().finally(() => {
      prediction.current = null;
      predictionKey.current = null;
      useFreeBrowseStore.getState().setAiBusy(false);
    });
    return prediction.current;
  }, [currentPredictionKey, runSerialized, saveCurrent, applyManifest, displayPrediction]);

  useEffect(() => {
    const safely = (action: () => Promise<void>) => runSerialized(async () => {
      inferenceGeneration.current++;
      const nv = nvRef.current;
      nv?.setDrawingEnabled(false);
      try {
        await action();
        useFreeBrowseStore.getState().setAiError(null);
      } catch (error) {
        useFreeBrowseStore.getState().setAiError(String(error));
        if (String(error).includes("409") && session) {
          const response = await fetch(`/ai/session/${session.session_id}`);
          if (response.ok) {
            const manifest = await response.json();
            applyManifest(manifest);
            await loadDrawing(manifest);
            await displayPrediction(manifest);
          }
        }
      } finally {
        const state = useFreeBrowseStore.getState();
        const locked = state.aiClasses.find((item) => item.id === state.aiActiveLabelId)?.locked;
        nv?.setDrawingEnabled(state.drawingOptions.mode === "pen" && !locked &&
          state.aiLabelDrawingReady);
      }
    });
    const actions: AiLabelActions = {
      setMode: (editingMode) => safely(async () => {
        const state = useFreeBrowseStore.getState();
        if (editingMode === state.aiEditingMode) return;
        if (state.aiEditingMode === "per_label") await saveCurrent();
        else if (session && nvRef.current?.drawBitmap) {
          const path = await uploadAiDrawing(
            nvRef.current.volumes[0], nvRef.current.drawBitmap.slice(), session.session_name,
          );
          await postAiJson(`/ai/session/${session.session_id}/set_annots`, { annotation_path: path });
        }
        if (session && state.aiEditingMode === "joint") {
          const response = await fetch(`/ai/session/${session.session_id}`);
          if (!response.ok) throw new Error("Could not load the current AI session");
          useFreeBrowseStore.setState({ aiBinaryState: await response.json() });
        }
        const manifest = await switchDrawing({
          editing_mode: editingMode, active_label_id: state.aiActiveLabelId ?? state.aiClasses[0]?.id,
        });
        if (editingMode === "per_label") await displayPrediction(manifest);
        else {
          removeOverlay();
          if (session) useFreeBrowseStore.getState().setAiActiveSession({
            ...session, result_path: manifest.result_path,
            result_model_id: manifest.last_inference_ml_id,
          });
        }
      }),
      select: (id) => safely(async () => {
        if (useFreeBrowseStore.getState().aiLabelDrawingReady) await saveCurrent();
        await switchDrawing({ active_label_id: id });
      }),
      add: (entry) => safely(async () => {
        await saveCurrent();
        const state = useFreeBrowseStore.getState();
        const manifest = await saveSettings({
          classes: [...state.aiClasses, entry], active_label_id: entry.id,
        });
        await loadDrawing(manifest);
      }),
      update: async (id, changes) => {
        const before = useFreeBrowseStore.getState();
        const entry = before.aiClasses.find((item) => item.id === id);
        if (!entry || Object.entries(changes).every(([key, value]) =>
          entry[key as keyof typeof entry] === value)) return;
        // Names and colors can wait; they must not invalidate a prediction in progress.
        await prediction.current?.catch(() => undefined);
        await safely(async () => {
          const state = useFreeBrowseStore.getState();
          if (state.aiActiveSession?.session_id !== before.aiActiveSession?.session_id) return;
          await saveSettings({
            classes: state.aiClasses.map((item) => item.id === id ? { ...item, ...changes } : item),
          });
        });
      },
      lock: (id, locked) => safely(async () => {
        await saveCurrent();
        const state = useFreeBrowseStore.getState();
        const manifest = await saveSettings({
          classes: state.aiClasses.map((item) => item.id === id ? { ...item, locked } : item),
        });
        await displayPrediction(manifest);
      }),
      remove: (id) => safely(async () => {
        await saveCurrent();
        const state = useFreeBrowseStore.getState();
        const remaining = state.aiClasses.filter((item) => item.id !== id);
        const manifest = await saveSettings({
          classes: remaining,
          active_label_id: state.aiActiveLabelId === id ? remaining[0]?.id : state.aiActiveLabelId,
        });
        history.current.delete(id);
        savedMarks.current.delete(id);
        await loadDrawing(manifest);
        await displayPrediction(manifest);
      }),
      clear: () => safely(async () => {
        const state = useFreeBrowseStore.getState();
        if (!session || !state.aiActiveLabelId) return;
        const manifest = await postAiJson(`/ai/session/${session.session_id}/binary/clear`, {
          label_id: state.aiActiveLabelId, state_version: state.aiBinaryState?.state_version ?? 0,
        });
        history.current.delete(state.aiActiveLabelId);
        savedMarks.current.delete(state.aiActiveLabelId);
        applyManifest(manifest);
        await loadDrawing(manifest);
        await displayPrediction(manifest);
      }),
    };
    useFreeBrowseStore.setState({ aiLabelActions: actions });
    return () => {
      if (useFreeBrowseStore.getState().aiLabelActions === actions) {
        useFreeBrowseStore.setState({ aiLabelActions: null });
      }
    };
  }, [session, nvRef, runSerialized, saveCurrent, saveSettings, switchDrawing, loadDrawing,
    displayPrediction, removeOverlay, applyManifest]);

  useEffect(() => {
    history.current.clear();
    savedMarks.current.clear();
    inferenceGeneration.current++;
    queuedPrediction.current = null;
    lastRequestedKey.current = null;
    removeOverlay();
    const state = useFreeBrowseStore.getState();
    if (session && state.aiEditingMode === "per_label" && state.aiBinaryState) {
      const labelId = state.aiActiveLabelId;
      if (labelId && nvRef.current?.drawBitmap) {
        savedMarks.current.set(labelId, nvRef.current.drawBitmap.slice());
      }
      void displayPrediction(state.aiBinaryState).catch((error) => state.setAiError(String(error)));
    }
    return removeOverlay;
  }, [session, nvRef, displayPrediction, removeOverlay]);

  useEffect(() => {
    if (!session || mode !== "per_label") return;
    const nv = nvRef.current;
    if (!nv) return;
    const previous = nv.onMouseUp;
    const onMouseUp: typeof nv.onMouseUp = (event) => {
      previous?.(event);
      if (nv.opts.drawingEnabled && !event.mouseButtonRightDown && !event.mouseButtonCenterDown) {
        useFreeBrowseStore.getState().markAiDrawingChanged();
      }
    };
    nv.onMouseUp = onMouseUp;
    return () => { if (nv.onMouseUp === onMouseUp) nv.onMouseUp = previous; };
  }, [session, mode, nvRef]);

  useEffect(() => {
    if (!session || mode !== "per_label" || revision === 0) return;
    inferenceGeneration.current++;
    const scheduledKey = currentPredictionKey();
    const timer = window.setTimeout(() => {
      const state = useFreeBrowseStore.getState();
      if (scheduledKey !== currentPredictionKey()) return;
      if (state.aiAutoUpdate && lastRequestedKey.current === currentPredictionKey()) return;
      const action = state.aiAutoUpdate ? requestPrediction() : runSerialized(saveCurrent);
      void action.catch((error) => useFreeBrowseStore.getState().setAiError(String(error)));
    }, 250);
    return () => window.clearTimeout(timer);
  }, [revision, session, mode, requestPrediction, runSerialized, saveCurrent, currentPredictionKey]);

  useEffect(() => {
    if (mode !== "per_label") return;
    const nv = nvRef.current;
    nv?.setDrawColormap(annotationColormap(classes) as unknown as string);
    if (overlay.current && nv) {
      overlay.current.setColormapLabel(annotationColormap(classes, true));
      overlay.current.opacity = visible ? opacity : 0;
      nv.updateGLVolume();
    }
  }, [classes, opacity, visible, mode, nvRef]);

  const save = useCallback(() => runSerialized(async () => {
    inferenceGeneration.current++;
    await saveCurrent(true);
    await saveSettings();
  }), [runSerialized, saveCurrent, saveSettings]);
  return { requestPrediction, save };
}
