import { beforeEach, describe, expect, it, vi } from "vitest";
import { act, renderHook } from "@testing-library/react";
import type { Niivue } from "@niivue/niivue";
import { useFreeBrowseStore } from "@/store";
import { useAiSession } from "@/hooks/use-ai-session";
import type { AiSessionSummary } from "@/store/ai-slice";

vi.mock("@/hooks/use-multiclass-annotation", () => ({
  useMulticlassAnnotation: () => ({ requestPrediction: vi.fn(), cancelPendingPrediction: vi.fn() }),
}));

const summary: AiSessionSummary = {
  session_id: "saved", session_name: "saved", created_at: "2026-10-09",
  volume_path: "scan.nii.gz", volume_path_root: "data", annotation_path: "annotations.nii.gz",
  result_path: "result.nii.gz", last_inference_ml_id: "fiery-flame", last_inference_at: "2026-10-09",
  model_id: "fiery-flame", annotation_mode: "multiclass", modality: "ct",
  classes: [{ id: 1, name: "Liver", color: "#00ff00" }],
};

function viewer() {
  return {
    volumes: [{ saveToUint8Array: vi.fn().mockResolvedValue(new Uint8Array([1, 2])) }],
    drawBitmap: new Uint8Array([1, 0]), opts: { drawingEnabled: true },
    loadVolumes: vi.fn().mockResolvedValue(undefined),
    niftiArray2NVImage: vi.fn().mockResolvedValue({}), loadDrawing: vi.fn().mockReturnValue(true),
    setDrawColormap: vi.fn(), setPenValue: vi.fn(), setDrawOpacity: vi.fn(),
    setDrawingEnabled: vi.fn(), closeDrawing: vi.fn(),
    drawClearAllUndoBitmaps: vi.fn(), drawAddUndoBitmap: vi.fn(), refreshDrawing: vi.fn(),
  };
}

beforeEach(() => {
  useFreeBrowseStore.setState({
    aiEnabled: false, aiModelId: "fiery-flame", aiAnnotationMode: "multiclass",
    aiActiveSession: null, aiAutoUpdate: true,
    aiEditingMode: "joint", aiActiveLabelId: null, aiBinaryState: null,
    aiLabelDrawingReady: true, aiRevision: 0,
    aiClasses: [{ id: 1, name: "Class 1", color: "#b6ff00" }],
    aiSessions: [{ ...summary, annotation_path: null, result_path: null, classes: undefined }],
  });
});

describe("new AI sessions", () => {
  function mockCreation(modelId: string, editingMode: "joint" | "per_label") {
    const saved: AiSessionSummary = {
      ...summary, session_id: "new", session_name: "new", model_id: modelId,
      annotation_path: null, result_path: null,
      classes: [{ id: 1, name: "Class 1", color: "#b6ff00" }],
      editing_mode: editingMode, active_label_id: editingMode === "per_label" ? 1 : null,
      state_version: 1,
      label_edits: editingMode === "per_label" ? { "1": { context_id: "new-label-context" } } : {},
    };
    const fetchMock = vi.fn(async (url: string, _options?: RequestInit) => {
      if (url === "/ai/session/new") return Response.json({ session_id: "new", session_name: "new" });
      if (url === "/ai/session/new/settings") return Response.json({ success: true, ...saved });
      if (url === "/ai/session/list") return Response.json([saved]);
      if (url === "/ai/session/new/set_volume") return Response.json({ success: true });
      throw new Error(`Unexpected request: ${url}`);
    });
    vi.stubGlobal("fetch", fetchMock);
    const nv = viewer();
    Object.assign(nv.volumes[0], { url: "/data/scan.nii.gz" });
    return { nv, fetchMock, saved };
  }

  it.each(["cool-blob", "open-whale", "grey-plant"])(
    "starts %s in per-label mode with foreground selected and the saved revision",
    async (modelId) => {
      useFreeBrowseStore.setState({ aiModelId: modelId, aiRevision: 12 });
      const { nv, fetchMock, saved } = mockCreation(modelId, "per_label");
      const { result } = renderHook(() => useAiSession({ current: nv as unknown as Niivue }));
      await act(async () => { await result.current.handleNewSession("new"); });
      const settings = fetchMock.mock.calls.find(([url]) => url === "/ai/session/new/settings");
      expect(settings).toBeDefined();
      expect(JSON.parse(settings?.[1]?.body as string)).toMatchObject({
        model_id: modelId, editing_mode: "per_label", active_label_id: 1, state_version: 0,
      });
      expect(useFreeBrowseStore.getState()).toMatchObject({
        aiEditingMode: "per_label", aiActiveLabelId: 1, aiLabelDrawingReady: true, aiRevision: 0,
        aiBinaryState: saved, aiActiveSession: { session_id: "new", session_name: "new" },
        drawingOptions: { penValue: 1, mode: "pen" },
      });
      expect(nv.setPenValue).toHaveBeenLastCalledWith(1, false);
      expect(nv.volumes[0].saveToUint8Array).not.toHaveBeenCalled();
    },
  );

  it.each(["fiery-flame", "nninteractive-v1"])("keeps %s on its existing editing mode", async (modelId) => {
    useFreeBrowseStore.setState({
      aiModelId: modelId, aiAnnotationMode: modelId === "fiery-flame" ? "multiclass" : "binary",
    });
    const { nv } = mockCreation(modelId, "joint");
    const { result } = renderHook(() => useAiSession({ current: nv as unknown as Niivue }));
    await act(async () => { await result.current.handleNewSession("new"); });
    expect(useFreeBrowseStore.getState()).toMatchObject({ aiEditingMode: "joint", aiActiveLabelId: null });
  });
});

describe("saved AI sessions", () => {
  it("preserves a supported model's saved joint mode", async () => {
    const saved = { ...summary, model_id: "grey-plant", editing_mode: "joint" };
    useFreeBrowseStore.setState({ aiEditingMode: "per_label", aiActiveLabelId: 1 });
    vi.stubGlobal("fetch", vi.fn(async (url: string) => url === "/ai/session/list"
      ? Response.json([saved]) : new Response(new Uint8Array([1]))));
    const nv = viewer();
    const { result } = renderHook(() => useAiSession({ current: nv as unknown as Niivue }));
    await act(async () => { await result.current.handleLoadSession("saved"); });
    expect(useFreeBrowseStore.getState()).toMatchObject({
      aiModelId: "grey-plant", aiEditingMode: "joint", aiActiveLabelId: null,
    });
    expect(nv.setPenValue).toHaveBeenLastCalledWith(255, false);
  });

  it.each([
    [[{ id: 2, name: "Kidney", color: "#00cfff" }], 255],
    [[], 255],
  ])("selects Background by default after reopening a session", async (classes, pen) => {
    const saved = { ...summary, classes };
    vi.stubGlobal("fetch", vi.fn(async (url: string) => url === "/ai/session/list"
      ? new Response(JSON.stringify([saved])) : new Response(new Uint8Array([1]))));
    const nv = viewer();
    const { result } = renderHook(() => useAiSession({ current: nv as unknown as Niivue }));
    await act(async () => { await result.current.handleLoadSession("saved"); });
    expect(nv.setPenValue).toHaveBeenLastCalledWith(pen, false);
    expect(useFreeBrowseStore.getState().drawingOptions.penValue).toBe(pen);
  });

  it("reloads authoritative annotations and class metadata instead of the stale sidebar summary", async () => {
    const fetchMock = vi.fn(async (url: string) => url === "/ai/session/list"
      ? new Response(JSON.stringify([summary])) : new Response(new Uint8Array([1])));
    vi.stubGlobal("fetch", fetchMock);
    const nv = viewer();
    const { result } = renderHook(() => useAiSession({ current: nv as unknown as Niivue }));
    await act(async () => { await result.current.handleLoadSession("saved"); });
    expect(fetchMock).toHaveBeenCalledWith("/data/ai-sessions/saved/annotations.nii.gz");
    expect(nv.loadDrawing).toHaveBeenCalledOnce();
    expect(useFreeBrowseStore.getState().aiClasses).toEqual(summary.classes);
    expect(useFreeBrowseStore.getState().aiActiveSession?.result_path).toBe("result.nii.gz");
  });

  it("retains the active drawing when saving fails", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response("Upload failed", { status: 500 })));
    useFreeBrowseStore.getState().setAiActiveSession({ session_id: "saved", session_name: "saved" });
    const nv = viewer();
    const { result } = renderHook(() => useAiSession({ current: nv as unknown as Niivue }));
    await act(async () => {
      await expect(result.current.handleExitAndSaveSession()).rejects.toThrow("annotations upload");
    });
    expect(useFreeBrowseStore.getState().aiActiveSession?.session_id).toBe("saved");
    expect(nv.closeDrawing).not.toHaveBeenCalled();
    expect(nv.setDrawingEnabled).toHaveBeenLastCalledWith(true);
  });

  it("prunes removed foreground labels when reopening a joint session", async () => {
    vi.stubGlobal("fetch", vi.fn(async (url: string) => url === "/ai/session/list"
      ? new Response(JSON.stringify([summary])) : new Response(new Uint8Array([1]))));
    const nv = viewer();
    nv.drawBitmap = new Uint8Array([1, 8, 255, 0]);
    const ref = { current: nv as unknown as Niivue };
    const { result } = renderHook(() => useAiSession(ref));
    await act(async () => { await result.current.handleLoadSession("saved"); });
    expect(nv.drawBitmap).toEqual(new Uint8Array([1, 0, 255, 0]));
    expect(nv.drawClearAllUndoBitmaps).toHaveBeenCalled();
  });

  it("refuses to overwrite saved annotations when the selected drawing has not loaded", async () => {
    useFreeBrowseStore.setState({
      aiActiveSession: { session_id: "saved", session_name: "saved" },
      aiEditingMode: "joint", aiLabelDrawingReady: false,
    });
    const fetchMock = vi.fn();
    vi.stubGlobal("fetch", fetchMock);
    const nv = viewer();
    const ref = { current: nv as unknown as Niivue };
    const { result } = renderHook(() => useAiSession(ref));
    await act(async () => {
      await expect(result.current.handleExitAndSaveSession()).rejects.toThrow("reload its drawing");
    });
    expect(fetchMock).not.toHaveBeenCalled();
    expect(nv.closeDrawing).not.toHaveBeenCalled();
  });
});
