import { useCallback, useEffect } from "react";
import type { Niivue } from "@niivue/niivue";
import { useFreeBrowseStore } from "@/store";
import {
  requestImagingUploadConfirmation,
  requestSessionDeleteConfirmation,
} from "@/lib/confirmations";
import { uint8ArrayToBase64 } from "@/lib/niivue-helpers";
import type { AiSessionSummary } from "@/store/ai-slice";
import { annotationColormap, BACKGROUND_LABEL, supportsPerLabelEditing } from "@/lib/ai-labels";
import { saveAiSettings } from "@/lib/ai-session-api";
import { useMulticlassAnnotation } from "@/hooks/use-multiclass-annotation";

const SESSION_NAME_RE = /^[A-Za-z0-9_-]+$/;

async function fetchSessionList(): Promise<AiSessionSummary[]> {
  const res = await fetch("/ai/session/list");
  if (!res.ok) throw new Error(`GET /ai/session/list failed: ${res.status}`);
  return res.json();
}

function ensureNiiName(name: string | undefined | null): string {
  if (!name) return "volume.nii.gz";
  if (name.endsWith(".nii.gz") || name.endsWith(".nii")) return name;
  return `${name}.nii.gz`;
}

function stripDataPrefix(url: string): string | null {
  const path = url.startsWith("/data/") ? url.slice(6) :
    url.startsWith("data/") ? url.slice(5) : null;
  return path?.startsWith("ai-sessions/") ? null : path;
}

async function fetchArrayBuffer(url: string): Promise<Uint8Array> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`GET ${url} failed: ${res.status}`);
  return new Uint8Array(await res.arrayBuffer());
}

async function exportAndUploadDrawing(
  nv: Niivue,
  sessionName: string,
): Promise<string | null> {
  if (!nv.drawBitmap || nv.volumes.length === 0) return null;
  // NVImage.saveToUint8Array(filename, drawing8) returns the drawing as a
  // gzipped NIfTI (when filename ends in .gz), without triggering a download.
  const annotRel = "annotations.nii.gz";
  const bitmap = nv.drawBitmap.slice();
  const bytes = await nv.volumes[0].saveToUint8Array(annotRel, bitmap);
  const targetPath = `ai-sessions/${sessionName}/${annotRel}`;
  const res = await fetch("/data/nii", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      filename: targetPath,
      data: uint8ArrayToBase64(bytes),
    }),
  });
  if (!res.ok)
    throw new Error(`annotations upload failed: ${res.status} ${res.statusText}`);
  return annotRel;
}

async function postSetAnnots(
  sessionId: string,
  annotationRelPath: string,
): Promise<void> {
  const res = await fetch(
    `/ai/session/${encodeURIComponent(sessionId)}/set_annots`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ annotation_path: annotationRelPath }),
    },
  );
  if (!res.ok) {
    const msg = await res.text();
    throw new Error(`set_annots failed (${res.status}): ${msg}`);
  }
}

const RESULT_FILENAME = "result.nii.gz";
const AI_RESULT_COLORMAP_NAME = "sky_blue";

const AI_RESULT_COLORMAP = {
  R: [0, 91, 91],
  G: [0, 163, 163],
  B: [0, 201, 201],
  A: [0, 64, 128],
  I: [0, 128, 255],
};

const AI_PROMPT_COLORMAP = {
  R: [0, 0, 255],
  G: [0, 255, 0],
  B: [0, 0, 0],
  A: [0, 255, 255],
  I: [0, 1, 2],
  labels: ["background", "positive", "negative"],
};

export function useAiSession(nvRef: React.RefObject<Niivue | null>) {
  const { requestPrediction, cancelPendingPrediction, saveLabelEdits } = useMulticlassAnnotation(nvRef);
  const aiEnabled = useFreeBrowseStore((s) => s.aiEnabled);
  const setAiSessions = useFreeBrowseStore((s) => s.setAiSessions);
  const setAiActiveSession = useFreeBrowseStore((s) => s.setAiActiveSession);
  const aiActiveSession = useFreeBrowseStore((s) => s.aiActiveSession);
  const setDrawingOptions = useFreeBrowseStore((s) => s.setDrawingOptions);
  const setActiveTab = useFreeBrowseStore((s) => s.setActiveTab);
  const setShowUploader = useFreeBrowseStore((s) => s.setShowUploader);
  const incrementVolumeVersion = useFreeBrowseStore(
    (s) => s.incrementVolumeVersion,
  );

  const refreshSessions = useCallback(async () => {
    try {
      const sessions = await fetchSessionList();
      setAiSessions(sessions);
    } catch (err) {
      console.error("Failed to refresh AI sessions:", err);
    }
  }, [setAiSessions]);

  useEffect(() => {
    if (aiEnabled) void refreshSessions();
  }, [aiEnabled, refreshSessions]);

  const enterDrawMode = useCallback(() => {
    const nv = nvRef.current;
    if (!nv) return;
    const state = useFreeBrowseStore.getState();
    const drawingOptions = state.drawingOptions;
    if (state.aiAnnotationMode === "multiclass") {
      const penValue = state.aiEditingMode === "per_label"
        ? state.aiActiveLabelId ?? BACKGROUND_LABEL : BACKGROUND_LABEL;
      if (!nv.drawBitmap) nv.createEmptyDrawing();
      nv.setDrawColormap(annotationColormap(state.aiClasses) as unknown as string);
      nv.setPenValue(penValue, false);
      nv.setDrawOpacity(1);
      nv.opts.clickToSegment = false;
      nv.setDrawingEnabled(!state.aiClasses.find((item) => item.id === state.aiActiveLabelId)?.locked);
      setDrawingOptions((prev) => ({
        ...prev, enabled: true, mode: "pen", opacity: 1, penValue,
        penFill: false, penErases: false,
      }));
      setActiveTab("aiAnnotation");
      return;
    }
    nv.setDrawingEnabled(false);
    const penValue = drawingOptions.penValue === 2 ? 2 : 1;
    // NiiVue accepts a ColorMap object here, but its type signature only exposes names.
    nv.setDrawColormap(AI_PROMPT_COLORMAP as unknown as string);
    nv.setPenValue(penValue, drawingOptions.penFill);
    nv.setDrawOpacity(1.0);
    setDrawingOptions((prev) => ({
      ...prev,
      enabled: true,
      mode: "none",
      opacity: 1.0,
      penValue,
    }));
    setActiveTab("aiAnnotation");
  }, [nvRef, setDrawingOptions, setActiveTab]);

  const exitDrawModeLocal = useCallback(() => {
    const nv = nvRef.current;
    if (nv) {
      nv.setDrawingEnabled(false);
      nv.opts.clickToSegment = false;
      nv.closeDrawing();
    }
    setDrawingOptions((prev) => ({ ...prev, enabled: false, mode: "none" }));
    setAiActiveSession(null);
    incrementVolumeVersion();
  }, [nvRef, setDrawingOptions, setAiActiveSession, incrementVolumeVersion]);

  const handleNewSession = useCallback(
    async (sessionName: string): Promise<void> => {
      const trimmed = sessionName.trim();
      if (!trimmed) throw new Error("Session name is required");
      if (!SESSION_NAME_RE.test(trimmed))
        throw new Error(
          "Session name may only contain letters, digits, underscores, and hyphens",
        );

      const nv = nvRef.current;
      if (!nv || nv.volumes.length === 0)
        throw new Error("Load a volume before creating a session");

      const volume = nv.volumes[0];
      const backendRelPath =
        typeof volume.url === "string" ? stripDataPrefix(volume.url) : null;
      const needsUpload = !backendRelPath;

      if (needsUpload) {
        const ok = await requestImagingUploadConfirmation();
        if (!ok) throw new Error("Upload cancelled");
      }

      const newRes = await fetch("/ai/session/new", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_name: trimmed }),
      });
      if (!newRes.ok) {
        const msg = await newRes.text();
        throw new Error(`Create session failed (${newRes.status}): ${msg}`);
      }
      const { session_id, session_name } = await newRes.json();

      let volumePathForSetVolume: string;
      if (needsUpload) {
        const basename = ensureNiiName(volume.name);
        const targetPath = `ai-sessions/${trimmed}/${basename}`;
        const uint8Array = await volume.saveToUint8Array(basename);
        const base64Data = uint8ArrayToBase64(uint8Array);

        const uploadRes = await fetch("/data/nii", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ filename: targetPath, data: base64Data }),
        });
        if (!uploadRes.ok)
          throw new Error(
            `Volume upload failed: ${uploadRes.status} ${uploadRes.statusText}`,
          );

        volumePathForSetVolume = basename;
      } else {
        volumePathForSetVolume = backendRelPath;
      }

      const setVolRes = await fetch(
        `/ai/session/${encodeURIComponent(session_id)}/set_volume`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            volume_path: volumePathForSetVolume,
            volume_path_root: needsUpload ? "session" : "data",
          }),
        },
      );
      if (!setVolRes.ok) {
        const msg = await setVolRes.text();
        throw new Error(`set_volume failed (${setVolRes.status}): ${msg}`);
      }

      const state = useFreeBrowseStore.getState();
      const classes = [{ id: 1, name: "Class 1", color: "#b6ff00" }];
      const editingMode = state.aiAnnotationMode === "multiclass" && supportsPerLabelEditing(state.aiModelId)
        ? "per_label" : "joint";
      const activeLabelId = editingMode === "per_label" ? classes[0].id : null;
      const saved = state.aiModelId ? await saveAiSettings(session_id, {
        model_id: state.aiModelId, annotation_mode: state.aiAnnotationMode,
        classes, modality: state.aiModality, editing_mode: editingMode,
        active_label_id: activeLabelId, state_version: 0,
      }) : null;
      useFreeBrowseStore.setState({
        aiEditingMode: saved?.editing_mode ?? editingMode,
        aiActiveLabelId: saved?.active_label_id ?? activeLabelId,
        aiBinaryState: saved, aiLabelDrawingReady: true,
      });
      state.setAiClasses(saved?.classes ?? classes);
      state.setAiAutoUpdate(true);
      nv.closeDrawing();
      setAiActiveSession({ session_id, session_name });
      enterDrawMode();
      await refreshSessions();
    },
    [nvRef, setAiActiveSession, enterDrawMode, refreshSessions],
  );

  const handleLoadSession = useCallback(
    async (sessionId: string): Promise<void> => {
      const nv = nvRef.current;
      if (!nv) throw new Error("Viewer not ready");

      const sessions = await fetchSessionList();
      setAiSessions(sessions);
      const summary = sessions.find((s) => s.session_id === sessionId);
      if (!summary) throw new Error(`Unknown session: ${sessionId}`);
      if (!summary.volume_path)
        throw new Error("Session has no volume set; cannot load");

      const volumeUrl =
        summary.volume_path_root === "session"
          ? `/data/ai-sessions/${summary.session_name}/${summary.volume_path}`
          : `/data/${summary.volume_path}`;

      // Ensure the niivue canvas is mounted + attached (first-load-in-session path).
      // Same trick use-file-loading.ts:196 uses before addVolumeFromUrl.
      setShowUploader(false);

      // Atomic scene swap — niivue manages the teardown internally.
      await nv.loadVolumes([{ url: volumeUrl }]);
      incrementVolumeVersion();

      const state = useFreeBrowseStore.getState();
      useFreeBrowseStore.setState({
        aiEditingMode: summary.editing_mode ?? "joint",
        aiActiveLabelId: summary.active_label_id ?? null,
        aiBinaryState: summary,
        aiLabelDrawingReady: true,
      });
      state.setAiModel(summary.model_id ?? summary.last_inference_ml_id ?? state.aiModelId,
        summary.annotation_mode ?? "binary");
      state.setAiClasses(summary.classes ?? [{ id: 1, name: "Class 1", color: "#b6ff00" }]);
      state.setAiModality(summary.modality ?? "ct");
      state.setAiAutoUpdate(true);

      const annotationPath = summary.editing_mode === "per_label" && summary.active_label_id
        ? summary.label_edits?.[String(summary.active_label_id)]?.annotation_path
        : summary.annotation_path;
      if (annotationPath) {
        const annotUrl = `/data/ai-sessions/${summary.session_name}/${annotationPath}`;
        const bytes = await fetchArrayBuffer(annotUrl);
        const nvimage = await nv.niftiArray2NVImage(bytes);
        if (!nv.loadDrawing(nvimage)) throw new Error("Saved annotations do not match this scan");
        if (summary.editing_mode !== "per_label" && summary.annotation_mode === "multiclass" &&
          nv.drawBitmap) {
          const ids = new Set(summary.classes?.map((item) => item.id));
          for (let index = 0; index < nv.drawBitmap.length; index++) {
            const value = nv.drawBitmap[index];
            if (value !== BACKGROUND_LABEL && !ids.has(value)) nv.drawBitmap[index] = 0;
          }
          nv.drawClearAllUndoBitmaps();
          nv.drawAddUndoBitmap();
          nv.refreshDrawing(true);
        }
      }

      setAiActiveSession({
        session_id: summary.session_id,
        session_name: summary.session_name,
        result_path: summary.result_path,
        result_model_id: summary.last_inference_ml_id,
      });
      enterDrawMode();
    },
    [nvRef, setShowUploader, incrementVolumeVersion, setAiActiveSession, setAiSessions, enterDrawMode],
  );

  const handleRunSegmentation = useCallback(
    async (mlId: string, labelValue: number): Promise<void> => {
      if (useFreeBrowseStore.getState().aiAnnotationMode === "multiclass") {
        await requestPrediction();
        return;
      }
      const nv = nvRef.current;
      const active = useFreeBrowseStore.getState().aiActiveSession;
      if (!nv || !active || nv.volumes.length === 0)
        throw new Error("No active session or volume");

      const annotRel = await exportAndUploadDrawing(nv, active.session_name);
      if (!annotRel)
        throw new Error("No drawing layer to send as annotations");

      await postSetAnnots(active.session_id, annotRel);

      const inferRes = await fetch(
        `/ai/session/${encodeURIComponent(active.session_id)}/infer/${encodeURIComponent(mlId)}`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ label_value: labelValue }),
        },
      );
      if (!inferRes.ok) {
        const msg = await inferRes.text();
        throw new Error(`inference failed (${inferRes.status}): ${msg}`);
      }
      await inferRes.json();

      // Replace any prior result overlay before loading the fresh one.
      const lastIdx = nv.volumes.length - 1;
      if (lastIdx > 0) {
        const last = nv.volumes[lastIdx];
        const lastName = last?.name ?? "";
        const lastUrl = last?.url ?? "";
        if (lastName === RESULT_FILENAME || lastUrl.includes(`/${RESULT_FILENAME}`)) {
          nv.removeVolumeByIndex(lastIdx);
        }
      }

      const resultUrl =
        `/data/ai-sessions/${active.session_name}/${RESULT_FILENAME}` +
        `?t=${Date.now()}`;
      nv.addColormap(AI_RESULT_COLORMAP_NAME, AI_RESULT_COLORMAP);
      await nv.addVolumeFromUrl({
        url: resultUrl,
        name: RESULT_FILENAME,
        colormap: AI_RESULT_COLORMAP_NAME,
        opacity: 0.5,
      });
      incrementVolumeVersion();
    },
    [nvRef, incrementVolumeVersion, requestPrediction],
  );

  const handleExitAndSaveSession = useCallback(async (): Promise<void> => {
    const nv = nvRef.current;
    const active = useFreeBrowseStore.getState().aiActiveSession;
    if (nv && active) {
      const state = useFreeBrowseStore.getState();
      const wasDrawing = nv.opts.drawingEnabled;
      cancelPendingPrediction();
      state.setAiAutoUpdate(false);
      nv.setDrawingEnabled(false);
      try {
        if (!state.aiLabelDrawingReady) {
          throw new Error("Select a label to reload its drawing before saving the session");
        }
        if (state.aiEditingMode === "per_label") await saveLabelEdits();
        else {
          const annotRel = await exportAndUploadDrawing(nv, active.session_name);
          if (annotRel) await postSetAnnots(active.session_id, annotRel);
          if (state.aiModelId) await saveAiSettings(active.session_id, {
            model_id: state.aiModelId, annotation_mode: state.aiAnnotationMode,
            classes: state.aiClasses, modality: state.aiModality,
          });
        }
      } catch (err) {
        nv.setDrawingEnabled(wasDrawing);
        state.setAiAutoUpdate(state.aiAutoUpdate);
        throw err;
      }
    }
    exitDrawModeLocal();
    await refreshSessions();
  }, [nvRef, exitDrawModeLocal, refreshSessions, cancelPendingPrediction, saveLabelEdits]);

  const handleExitAndDeleteSession = useCallback(async (): Promise<void> => {
    if (!aiActiveSession) return;
    const ok = await requestSessionDeleteConfirmation();
    if (!ok) return;

    const id = aiActiveSession.session_id;
    try {
      const res = await fetch(`/ai/session/${encodeURIComponent(id)}`, {
        method: "DELETE",
      });
      if (!res.ok) {
        const msg = await res.text();
        console.error(`DELETE session failed (${res.status}): ${msg}`);
      }
    } catch (err) {
      console.error("DELETE session error:", err);
    }

    exitDrawModeLocal();
    await refreshSessions();
  }, [aiActiveSession, exitDrawModeLocal, refreshSessions]);

  return {
    refreshSessions,
    handleNewSession,
    handleLoadSession,
    handleRunSegmentation,
    handleExitAndSaveSession,
    handleExitAndDeleteSession,
  };
}
