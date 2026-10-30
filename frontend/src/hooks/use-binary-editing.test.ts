import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { act, renderHook } from "@testing-library/react";
import { NVImage, type Niivue } from "@niivue/niivue";
import { useFreeBrowseStore } from "@/store";
import { useBinaryEditing } from "@/hooks/use-binary-editing";
import { encodeAiDrawing, postAiJson, uploadAiDrawing } from "@/lib/ai-session-api";
import type { AiSessionSummary } from "@/store/ai-slice";

vi.mock("@niivue/niivue", () => ({ NVImage: { loadFromUrl: vi.fn() } }));
vi.mock("@/lib/ai-session-api", () => ({
  postAiJson: vi.fn(), encodeAiDrawing: vi.fn(), uploadAiDrawing: vi.fn(),
}));

let server: AiSessionSummary;
let drawings: Map<string, Uint8Array>;

function viewer() {
  const nv = {
    volumes: [{}], drawBitmap: new Uint8Array([3, 8, 255, 0]),
    opts: { drawingEnabled: true }, drawUndoBitmaps: [new Uint8Array([11, 12])],
    currentDrawUndoBitmap: 0, onMouseUp: vi.fn(),
    setPenValue: vi.fn(), setDrawColormap: vi.fn(), setDrawOpacity: vi.fn(),
    setDrawingEnabled: vi.fn((value: boolean) => { nv.opts.drawingEnabled = value; }),
    refreshDrawing: vi.fn(), createEmptyDrawing: vi.fn(),
    drawClearAllUndoBitmaps: vi.fn(() => { nv.drawUndoBitmaps = []; }),
    drawAddUndoBitmap: vi.fn(), niftiArray2NVImage: vi.fn(async (bytes) => ({ img: bytes })),
    loadDrawing: vi.fn((image) => { nv.drawBitmap = image.img.slice(); return true; }),
    removeVolume: vi.fn(), addVolume: vi.fn(), updateGLVolume: vi.fn(),
  };
  return nv;
}

beforeEach(() => {
  vi.useFakeTimers();
  vi.clearAllMocks();
  drawings = new Map();
  server = {
    session_id: "one", session_name: "one", created_at: "2026-10-21", state_version: 5,
    volume_path: "scan.nii.gz", volume_path_root: "data", annotation_path: "annotations.nii.gz",
    result_path: null, last_inference_at: null, last_inference_ml_id: null,
    model_id: "cool-blob", annotation_mode: "multiclass", editing_mode: "joint", modality: "ct",
    classes: [{ id: 3, name: "A", color: "#00ff00" }, { id: 8, name: "B", color: "#00ffff" }],
  };
  useFreeBrowseStore.setState({
    aiActiveSession: { session_id: "one", session_name: "one" },
    aiModelId: "cool-blob", aiAnnotationMode: "multiclass", aiEditingMode: "joint",
    aiBinaryState: null, aiActiveLabelId: null, aiLabelActions: null, aiLabelSaving: false,
    aiLabelDrawingReady: true,
    aiRevision: 0, aiAutoUpdate: false, aiBusy: false, aiError: null, aiClasses: server.classes,
  });
  vi.mocked(encodeAiDrawing).mockImplementation(async (_volume, bitmap) => JSON.stringify([...bitmap]));
  vi.mocked(uploadAiDrawing).mockResolvedValue("annotations.nii.gz");
  vi.mocked(NVImage.loadFromUrl).mockResolvedValue({
    setColormapLabel: vi.fn(), img: new Uint8Array(4),
  } as unknown as NVImage);
  vi.stubGlobal("fetch", vi.fn(async (url: string) => {
    if (url === "/ai/session/one") return new Response(JSON.stringify(server));
    const path = url.split("/").pop()!.split("?")[0];
    return new Response(drawings.get(path) ?? new Uint8Array([3, 8, 255, 0]));
  }));
  vi.mocked(postAiJson).mockImplementation(async (url, raw) => {
    const body = raw as Record<string, unknown>;
    if (url.endsWith("/set_annots")) return { success: true };
    if (body.state_version !== server.state_version) throw new Error("409: stale session");
    server = structuredClone(server);
    server.state_version!++;
    if (url.endsWith("/settings")) {
      Object.assign(server, body, { state_version: server.state_version });
      server.state_version = (body.state_version as number) + 1;
      server.label_edits ??= {};
      for (const item of server.classes ?? []) {
        if (!server.label_edits[String(item.id)]) {
          const path = `initial-${item.id}.nii.gz`;
          drawings.set(path, new Uint8Array([3, 8, 255, 0].map((value) =>
            value === item.id || value === 255 ? value : 0)));
          server.label_edits[String(item.id)] = { annotation_path: path };
        }
      }
    } else if (url.endsWith("/binary/annotations")) {
      const path = `label-${body.label_id}-${server.state_version}.nii.gz`;
      drawings.set(path, Uint8Array.from(JSON.parse(body.data as string)));
      server.label_edits![String(body.label_id)].annotation_path = path;
    } else if (url.includes("/infer/")) server.result_path = `result-${server.state_version}.nii.gz`;
    return structuredClone(server);
  });
});
afterEach(() => { vi.useRealTimers(); vi.unstubAllGlobals(); });

describe("opt-in binary editing", () => {
  it.each(["cool-blob", "open-whale", "grey-plant"])("shares manual and automatic predictions for %s", async (modelId) => {
    server.model_id = modelId;
    useFreeBrowseStore.setState({ aiModelId: modelId });
    const nv = viewer();
    const ref = { current: nv as unknown as Niivue };
    const { result, unmount } = renderHook(() => useBinaryEditing(ref));
    await act(async () => { await useFreeBrowseStore.getState().aiLabelActions!.setMode("per_label"); });
    await act(async () => { await vi.advanceTimersByTimeAsync(400); });
    const normalPost = vi.mocked(postAiJson).getMockImplementation()!;
    let finish!: () => Promise<void>;
    vi.mocked(postAiJson).mockImplementation((url, body) => url.includes("/infer/")
      ? new Promise((resolve, reject) => {
        finish = async () => { try { resolve(await normalPost(url, body)); } catch (error) { reject(error); } };
      }) : normalPost(url, body));
    act(() => {
      useFreeBrowseStore.getState().setAiAutoUpdate(true);
      useFreeBrowseStore.getState().markAiDrawingChanged();
    });
    let running!: Promise<void>;
    await act(async () => { running = result.current.requestPrediction(); await Promise.resolve(); });
    await act(async () => { await vi.advanceTimersByTimeAsync(250); });
    expect(vi.mocked(postAiJson).mock.calls.filter(([url]) => url.includes("/infer/"))).toHaveLength(1);
    expect(postAiJson).toHaveBeenCalledWith(`/ai/session/one/infer/${modelId}`, expect.anything());
    await act(async () => { await finish(); await running; });
    expect(nv.addVolume).toHaveBeenCalledOnce();
    expect(useFreeBrowseStore.getState().aiBusy).toBe(false);
    expect(useFreeBrowseStore.getState().aiError).toBeNull();
    unmount();
  });

  it("queues only the newest drawing while a prediction is running", async () => {
    const nv = viewer();
    const ref = { current: nv as unknown as Niivue };
    const { result, unmount } = renderHook(() => useBinaryEditing(ref));
    await act(async () => { await useFreeBrowseStore.getState().aiLabelActions!.setMode("per_label"); });
    await act(async () => { await vi.advanceTimersByTimeAsync(400); });
    const normalPost = vi.mocked(postAiJson).getMockImplementation()!;
    const completions: Array<() => Promise<void>> = [];
    vi.mocked(postAiJson).mockImplementation((url, body) => url.includes("/infer/")
      ? new Promise((resolve, reject) => {
        completions.push(async () => {
          try { resolve(await normalPost(url, body)); } catch (error) { reject(error); }
        });
      }) : normalPost(url, body));
    let running!: Promise<void>;
    await act(async () => { running = result.current.requestPrediction(); await Promise.resolve(); });
    act(() => { useFreeBrowseStore.getState().setAiAutoUpdate(true); });
    for (const position of [1, 2, 3]) {
      act(() => { nv.drawBitmap[position] = 3; useFreeBrowseStore.getState().markAiDrawingChanged(); });
      await act(async () => { await vi.advanceTimersByTimeAsync(250); });
    }
    expect(completions).toHaveLength(1);
    await act(async () => { await completions[0](); await Promise.resolve(); });
    expect(completions).toHaveLength(2);
    await act(async () => { await completions[1](); await running; });
    expect(nv.addVolume).toHaveBeenCalledOnce();
    expect(postAiJson).toHaveBeenCalledWith("/ai/session/one/binary/annotations",
      expect.objectContaining({ data: "[3,3,3,3]" }));
    expect(useFreeBrowseStore.getState().aiError).toBeNull();
    unmount();
  });

  it("keeps predictions visible when an unchanged name blurs or a real rename waits", async () => {
    const nv = viewer();
    const ref = { current: nv as unknown as Niivue };
    const { result, unmount } = renderHook(() => useBinaryEditing(ref));
    await act(async () => { await useFreeBrowseStore.getState().aiLabelActions!.setMode("per_label"); });
    const normalPost = vi.mocked(postAiJson).getMockImplementation()!;
    let finish!: () => Promise<void>;
    vi.mocked(postAiJson).mockImplementation((url, body) => url.includes("/infer/")
      ? new Promise((resolve, reject) => {
        finish = async () => { try { resolve(await normalPost(url, body)); } catch (error) { reject(error); } };
      }) : normalPost(url, body));
    let running!: Promise<void>;
    await act(async () => { running = result.current.requestPrediction(); await Promise.resolve(); });
    const settingsBefore = vi.mocked(postAiJson).mock.calls.filter(([url]) => url.endsWith("/settings")).length;
    await act(async () => { await useFreeBrowseStore.getState().aiLabelActions!.update(3, { name: "A" }); });
    expect(useFreeBrowseStore.getState().aiBusy).toBe(true);
    let rename!: Promise<void>;
    await act(async () => {
      rename = useFreeBrowseStore.getState().aiLabelActions!.update(3, { name: "Renamed" });
      await vi.advanceTimersByTimeAsync(400);
    });
    expect(vi.mocked(postAiJson).mock.calls.filter(([url]) => url.endsWith("/settings"))).toHaveLength(settingsBefore);
    await act(async () => { await finish(); await running; await rename; });
    expect(nv.addVolume).toHaveBeenCalledOnce();
    expect(server.classes![0].name).toBe("Renamed");
    expect(useFreeBrowseStore.getState().aiError).toBeNull();
    unmount();
  });

  it("starts with joint editing and fetches the current revision before first opt-in", async () => {
    const nv = viewer();
    const ref = { current: nv as unknown as Niivue };
    const { unmount } = renderHook(() => useBinaryEditing(ref));
    expect(useFreeBrowseStore.getState().aiEditingMode).toBe("joint");
    await act(async () => { await useFreeBrowseStore.getState().aiLabelActions!.setMode("per_label"); });
    expect(fetch).toHaveBeenCalledWith("/ai/session/one");
    expect(postAiJson).toHaveBeenCalledWith("/ai/session/one/settings", expect.objectContaining({
      editing_mode: "per_label", state_version: 5, active_label_id: 3,
    }));
    expect(nv.drawBitmap).toEqual(new Uint8Array([3, 0, 255, 0]));
    expect(useFreeBrowseStore.getState().aiError).toBeNull();
    unmount();
  });

  it("saves manual marks and restores private foreground/background and undo when switching", async () => {
    const nv = viewer();
    const ref = { current: nv as unknown as Niivue };
    const { unmount } = renderHook(() => useBinaryEditing(ref));
    await act(async () => { await useFreeBrowseStore.getState().aiLabelActions!.setMode("per_label"); });
    nv.drawBitmap.set([3, 255, 0, 3]);
    nv.drawUndoBitmaps = [new Uint8Array([21, 22]), new Uint8Array([23, 24])];
    nv.currentDrawUndoBitmap = 1;
    await act(async () => { await useFreeBrowseStore.getState().aiLabelActions!.select(8); });
    expect(nv.drawBitmap).toEqual(new Uint8Array([0, 8, 255, 0]));
    nv.drawBitmap.set([255, 8, 8, 0]);
    nv.drawUndoBitmaps = [new Uint8Array([31, 32])];
    nv.currentDrawUndoBitmap = 0;
    await act(async () => { await useFreeBrowseStore.getState().aiLabelActions!.select(3); });
    expect(nv.drawBitmap).toEqual(new Uint8Array([3, 255, 0, 3]));
    expect(nv.drawUndoBitmaps).toEqual([new Uint8Array([21, 22]), new Uint8Array([23, 24])]);
    expect(nv.currentDrawUndoBitmap).toBe(1);
    const saves = vi.mocked(postAiJson).mock.calls.filter(([url]) => url.endsWith("binary/annotations"));
    expect(saves.map(([, body]) => (body as { data: string }).data)).toEqual([
      "[3,255,0,3]", "[255,8,8,0]",
    ]);
    expect(postAiJson).not.toHaveBeenCalledWith(expect.stringContaining("/infer/"), expect.anything());
    unmount();
  });

  it("persists drawing gestures while automatic prediction is off", async () => {
    const nv = viewer();
    const ref = { current: nv as unknown as Niivue };
    const { unmount } = renderHook(() => useBinaryEditing(ref));
    await act(async () => { await useFreeBrowseStore.getState().aiLabelActions!.setMode("per_label"); });
    act(() => {
      nv.drawBitmap[3] = 3;
      nv.onMouseUp({ mouseButtonRightDown: false, mouseButtonCenterDown: false });
    });
    await act(async () => { await vi.advanceTimersByTimeAsync(250); });
    expect(postAiJson).toHaveBeenCalledWith("/ai/session/one/binary/annotations", expect.objectContaining({
      label_id: 3, data: "[3,0,255,3]",
    }));
    expect(postAiJson).not.toHaveBeenCalledWith(expect.stringContaining("/infer/"), expect.anything());
    unmount();
  });

  it("disables editing an accepted label and restores editing after explicit unlock", async () => {
    const nv = viewer();
    const ref = { current: nv as unknown as Niivue };
    const { unmount } = renderHook(() => useBinaryEditing(ref));
    await act(async () => { await useFreeBrowseStore.getState().aiLabelActions!.setMode("per_label"); });
    await act(async () => { await useFreeBrowseStore.getState().aiLabelActions!.lock(3, true); });
    expect(nv.opts.drawingEnabled).toBe(false);
    expect(useFreeBrowseStore.getState().aiClasses.find((item) => item.id === 3)?.locked).toBe(true);
    await act(async () => { await useFreeBrowseStore.getState().aiLabelActions!.lock(3, false); });
    expect(nv.opts.drawingEnabled).toBe(true);
    unmount();
  });

  it("clears busy after a manual prediction is invalidated by a newer drawing", async () => {
    const nv = viewer();
    const ref = { current: nv as unknown as Niivue };
    const { result, unmount } = renderHook(() => useBinaryEditing(ref));
    await act(async () => { await useFreeBrowseStore.getState().aiLabelActions!.setMode("per_label"); });
    let rejectPrediction!: (error: Error) => void;
    const normalPost = vi.mocked(postAiJson).getMockImplementation()!;
    vi.mocked(postAiJson).mockImplementation((url, body) => url.includes("/infer/")
      ? new Promise((_resolve, reject) => { rejectPrediction = reject; }) : normalPost(url, body));
    let running!: Promise<void>;
    await act(async () => { running = result.current.requestPrediction(); await Promise.resolve(); });
    expect(useFreeBrowseStore.getState().aiBusy).toBe(true);
    act(() => {
      nv.drawBitmap[3] = 3;
      useFreeBrowseStore.getState().markAiDrawingChanged();
    });
    await act(async () => { await vi.advanceTimersByTimeAsync(250); });
    await act(async () => { rejectPrediction(new Error("409: stale prediction")); await running; });
    expect(useFreeBrowseStore.getState().aiBusy).toBe(false);
    expect(nv.addVolume).not.toHaveBeenCalled();
    unmount();
  });

  it("restores the prior selection and drawing after a failed label load", async () => {
    const nv = viewer();
    const ref = { current: nv as unknown as Niivue };
    const { result, unmount } = renderHook(() => useBinaryEditing(ref));
    await act(async () => { await useFreeBrowseStore.getState().aiLabelActions!.setMode("per_label"); });
    const normalFetch = vi.mocked(fetch).getMockImplementation()!;
    vi.mocked(fetch).mockImplementation((url, options) => String(url).includes("initial-8")
      ? Promise.resolve(new Response("Failed drawing", { status: 500 })) : normalFetch(url, options));
    await act(async () => { await useFreeBrowseStore.getState().aiLabelActions!.select(8); });
    expect(useFreeBrowseStore.getState().aiActiveLabelId).toBe(3);
    expect(server.active_label_id).toBe(3);
    expect(useFreeBrowseStore.getState().aiLabelDrawingReady).toBe(true);
    expect(nv.drawBitmap).toEqual(new Uint8Array([3, 0, 255, 0]));
    expect(postAiJson).not.toHaveBeenCalledWith("/ai/session/one/binary/annotations",
      expect.objectContaining({ label_id: 8 }));
    vi.mocked(fetch).mockImplementation(normalFetch);
    await act(async () => { await useFreeBrowseStore.getState().aiLabelActions!.select(8); });
    expect(nv.drawBitmap).toEqual(new Uint8Array([0, 8, 255, 0]));
    expect(nv.opts.drawingEnabled).toBe(true);
    expect(useFreeBrowseStore.getState().aiLabelDrawingReady).toBe(true);
    unmount();
  });

  it("restores binary mode and its exact drawing if loading the joint drawing fails", async () => {
    const nv = viewer();
    const ref = { current: nv as unknown as Niivue };
    const { result, unmount } = renderHook(() => useBinaryEditing(ref));
    await act(async () => { await useFreeBrowseStore.getState().aiLabelActions!.setMode("per_label"); });
    nv.drawBitmap.set([3, 255, 3, 0]);
    const normalFetch = vi.mocked(fetch).getMockImplementation()!;
    vi.mocked(fetch).mockImplementation((url, options) => String(url).includes("annotations.nii.gz")
      ? Promise.resolve(new Response("Failed drawing", { status: 503 })) : normalFetch(url, options));
    await act(async () => { await useFreeBrowseStore.getState().aiLabelActions!.setMode("joint"); });
    expect(useFreeBrowseStore.getState().aiEditingMode).toBe("per_label");
    expect(server.editing_mode).toBe("per_label");
    expect(nv.drawBitmap).toEqual(new Uint8Array([3, 255, 3, 0]));
    await act(async () => { await result.current.save(); });
    const finalSave = vi.mocked(postAiJson).mock.calls.filter(([url]) =>
      url.endsWith("binary/annotations")).at(-1)![1];
    expect(finalSave).toEqual(expect.objectContaining({ label_id: 3, data: "[3,255,3,0]" }));
    unmount();
  });

  it("removes deleted labels from the restored joint drawing and its undo history", async () => {
    const nv = viewer();
    const ref = { current: nv as unknown as Niivue };
    const { unmount } = renderHook(() => useBinaryEditing(ref));
    await act(async () => { await useFreeBrowseStore.getState().aiLabelActions!.setMode("per_label"); });
    await act(async () => { await useFreeBrowseStore.getState().aiLabelActions!.remove(3); });
    await act(async () => { await useFreeBrowseStore.getState().aiLabelActions!.setMode("joint"); });
    expect(nv.drawBitmap).toEqual(new Uint8Array([0, 8, 255, 0]));
    expect(nv.drawClearAllUndoBitmaps).toHaveBeenCalled();
    expect(nv.drawAddUndoBitmap).toHaveBeenCalled();
    unmount();
  });
});
