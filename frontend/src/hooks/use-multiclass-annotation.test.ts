import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { act, renderHook } from "@testing-library/react";
import { NVImage, type Niivue } from "@niivue/niivue";
import { useFreeBrowseStore } from "@/store";
import { useMulticlassAnnotation } from "@/hooks/use-multiclass-annotation";
import { postAiJson, saveAiSettings, uploadAiDrawing } from "@/lib/ai-session-api";

vi.mock("@niivue/niivue", () => ({ NVImage: { loadFromUrl: vi.fn() } }));
vi.mock("@/lib/ai-session-api", () => ({
  postAiJson: vi.fn(), saveAiSettings: vi.fn(), uploadAiDrawing: vi.fn(),
}));

beforeEach(() => {
  vi.useFakeTimers();
  vi.clearAllMocks();
  useFreeBrowseStore.setState({
    aiActiveSession: { session_id: "one", session_name: "one" }, aiAnnotationMode: "multiclass",
    aiModelId: "fiery-flame", aiRevision: 0, aiClearRequest: 0, aiAutoUpdate: true,
    aiClasses: [{ id: 1, name: "Class 1", color: "#b6ff00" }],
  });
  vi.mocked(uploadAiDrawing).mockResolvedValue("annotations.nii.gz");
  vi.mocked(saveAiSettings).mockImplementation(async (sessionId, settings) => ({
    session_id: sessionId, session_name: sessionId, created_at: "2026-10-26",
    volume_path: "scan.nii.gz", volume_path_root: "data", annotation_path: null,
    result_path: null, last_inference_ml_id: null, last_inference_at: null,
    ...settings,
  }));
  vi.mocked(postAiJson).mockResolvedValue({ result_path: "result.nii.gz" });
});
afterEach(() => vi.useRealTimers());

function viewer() {
  return {
    volumes: [{}], drawBitmap: new Uint8Array(4), opts: { drawingEnabled: true },
    onMouseUp: vi.fn(), setDrawColormap: vi.fn(), setDrawOpacity: vi.fn(),
    removeVolume: vi.fn(), addVolume: vi.fn(), updateGLVolume: vi.fn(),
    drawAddUndoBitmap: vi.fn(), drawClearAllUndoBitmaps: vi.fn(), refreshDrawing: vi.fn(),
  };
}

describe("automatic multiclass prediction", () => {
  it("restores surviving saved predictions and clears deleted labels", async () => {
    useFreeBrowseStore.setState({ aiAutoUpdate: false, aiActiveSession: {
      session_id: "one", session_name: "one", result_path: "result.nii.gz",
      result_model_id: "fiery-flame",
    } });
    const image = {
      setColormapLabel: vi.fn(), img: new Uint8Array([1, 2, 0]),
    } as unknown as NVImage;
    vi.mocked(NVImage.loadFromUrl).mockResolvedValue(image);
    const nv = viewer();
    nv.drawBitmap[0] = 1;
    const ref = { current: nv as unknown as Niivue };
    const { unmount } = renderHook(() => useMulticlassAnnotation(ref));
    await act(async () => { await Promise.resolve(); });
    expect(NVImage.loadFromUrl).toHaveBeenCalledOnce();
    expect(image.img).toEqual(new Uint8Array([1, 0, 0]));
    expect(nv.addVolume).toHaveBeenCalledWith(image);
    unmount();
  });

  it("removes deleted marks and resets prediction context in manual mode", async () => {
    useFreeBrowseStore.setState({ aiAutoUpdate: false, aiModelId: "segmento-v1", aiClasses: [
      { id: 1, name: "Liver", color: "#b6ff00" },
      { id: 2, name: "Kidney", color: "#00cfff" },
    ] });
    const image = {
      setColormapLabel: vi.fn(), img: new Uint8Array([1, 2, 0, 2]),
    } as unknown as NVImage;
    vi.mocked(NVImage.loadFromUrl).mockResolvedValue(image);
    const nv = viewer();
    nv.drawBitmap.set([1, 2, 255, 2]);
    const ref = { current: nv as unknown as Niivue };
    const { result, unmount } = renderHook(() => useMulticlassAnnotation(ref));
    await act(async () => { await result.current.requestPrediction(); });
    nv.volumes.push(image);
    act(() => {
      useFreeBrowseStore.getState().setAiClasses([
        { id: 1, name: "Liver", color: "#b6ff00" },
      ]);
    });
    expect(nv.drawBitmap).toEqual(new Uint8Array([1, 0, 255, 0]));
    expect(nv.removeVolume).not.toHaveBeenCalled();
    expect(image.img).toEqual(new Uint8Array([1, 0, 0, 0]));
    expect(nv.updateGLVolume).toHaveBeenCalled();
    expect(nv.drawClearAllUndoBitmaps).toHaveBeenCalledOnce();
    expect(nv.drawAddUndoBitmap).toHaveBeenCalledOnce();
    await act(async () => { await result.current.requestPrediction(); });
    const requests = vi.mocked(postAiJson).mock.calls.filter(([url]) => url.includes("/infer/"));
    expect(requests[1][1]).not.toEqual(requests[0][1]);
    expect(vi.mocked(uploadAiDrawing).mock.calls[1][1]).toEqual(new Uint8Array([1, 0, 255, 0]));
    unmount();
  });

  it("discards an in-flight prediction after deletion and waits for the next interaction",
    async () => {
      useFreeBrowseStore.setState({ aiClasses: [
        { id: 1, name: "Liver", color: "#b6ff00" },
        { id: 2, name: "Kidney", color: "#00cfff" },
      ] });
      let resolveImage!: (image: NVImage) => void;
      vi.mocked(NVImage.loadFromUrl).mockReturnValueOnce(new Promise((resolve) => {
        resolveImage = resolve;
      })).mockResolvedValue({ setColormapLabel: vi.fn() } as unknown as NVImage);
      const nv = viewer();
      nv.drawBitmap.set([1, 2, 255, 0]);
      const ref = { current: nv as unknown as Niivue };
      const { result, unmount } = renderHook(() => useMulticlassAnnotation(ref));
      await act(async () => { await result.current.requestPrediction(); });
      act(() => {
        useFreeBrowseStore.getState().setAiClasses([
          { id: 1, name: "Liver", color: "#b6ff00" },
        ]);
      });
      await act(async () => { resolveImage({ setColormapLabel: vi.fn() } as unknown as NVImage); });
      expect(nv.addVolume).not.toHaveBeenCalled();
      await act(async () => { await vi.advanceTimersByTimeAsync(1000); });
      expect(uploadAiDrawing).toHaveBeenCalledOnce();
      expect(nv.addVolume).not.toHaveBeenCalled();
      act(() => {
        nv.drawBitmap[3] = 1;
        nv.onMouseUp({ mouseButtonRightDown: false, mouseButtonCenterDown: false });
      });
      await act(async () => { await vi.advanceTimersByTimeAsync(250); });
      expect(vi.mocked(uploadAiDrawing).mock.calls[1][1]).toEqual(new Uint8Array([1, 0, 255, 1]));
      expect(nv.addVolume).toHaveBeenCalledOnce();
      unmount();
    });

  it("cancels a debounced gesture when its label is deleted in the same batch", async () => {
    useFreeBrowseStore.setState({ aiClasses: [
      { id: 1, name: "Liver", color: "#b6ff00" },
      { id: 2, name: "Kidney", color: "#00cfff" },
    ] });
    const nv = viewer();
    const ref = { current: nv as unknown as Niivue };
    const { unmount } = renderHook(() => useMulticlassAnnotation(ref));
    act(() => {
      nv.drawBitmap[1] = 2;
      nv.onMouseUp({ mouseButtonRightDown: false, mouseButtonCenterDown: false });
      useFreeBrowseStore.getState().setAiClasses([
        { id: 1, name: "Liver", color: "#b6ff00" },
      ]);
    });
    await act(async () => { await vi.advanceTimersByTimeAsync(1000); });
    expect(nv.drawBitmap).toEqual(new Uint8Array(4));
    expect(uploadAiDrawing).not.toHaveBeenCalled();
    unmount();
  });

  it("keeps surviving predictions after deletion with auto-update enabled", async () => {
    useFreeBrowseStore.setState({ aiModelId: "segmento-v1", aiClasses: [
      { id: 1, name: "Liver", color: "#b6ff00" },
      { id: 2, name: "Kidney", color: "#00cfff" },
    ] });
    const image = {
      setColormapLabel: vi.fn(), img: new Uint8Array([1, 2, 0, 1]),
    } as unknown as NVImage;
    vi.mocked(NVImage.loadFromUrl).mockResolvedValue(image);
    const nv = viewer();
    nv.drawBitmap.set([1, 2, 255, 0]);
    const ref = { current: nv as unknown as Niivue };
    const { result, unmount } = renderHook(() => useMulticlassAnnotation(ref));
    await act(async () => { await vi.advanceTimersByTimeAsync(250); });
    act(() => {
      useFreeBrowseStore.getState().setAiClasses([
        { id: 1, name: "Liver", color: "#b6ff00" },
      ]);
    });
    await act(async () => { await vi.advanceTimersByTimeAsync(1000); });
    expect(uploadAiDrawing).toHaveBeenCalledOnce();
    expect(image.img).toEqual(new Uint8Array([1, 0, 0, 1]));
    expect(nv.removeVolume).not.toHaveBeenCalled();
    const before = vi.mocked(postAiJson).mock.calls.find(([url]) => url.includes("/infer/"))?.[1];
    await act(async () => { await result.current.requestPrediction(); });
    const after = vi.mocked(postAiJson).mock.calls.filter(([url]) => url.includes("/infer/"))[1][1];
    expect(after).not.toEqual(before);
    unmount();
  });

  it("keeps prediction context across marks and resets it on Clear even in manual mode", async () => {
    useFreeBrowseStore.setState({ aiAutoUpdate: false, aiModelId: "segmento-v1" });
    vi.mocked(NVImage.loadFromUrl).mockResolvedValue({ setColormapLabel: vi.fn() } as unknown as NVImage);
    const nv = viewer();
    const ref = { current: nv as unknown as Niivue };
    const { result, unmount } = renderHook(() => useMulticlassAnnotation(ref));
    nv.drawBitmap[0] = 1;
    await act(async () => { await result.current.requestPrediction(); });
    nv.drawBitmap[1] = 1;
    await act(async () => { await result.current.requestPrediction(); });
    act(() => { useFreeBrowseStore.getState().clearAiAnnotations(); });
    nv.drawBitmap[0] = 1;
    await act(async () => { await result.current.requestPrediction(); });
    const requests = vi.mocked(postAiJson).mock.calls.filter(([url]) => url.includes("/infer/"));
    const contexts = requests.map(([, body]) =>
      (body as { prediction_context_id: string }).prediction_context_id);
    expect(contexts[0]).toEqual(expect.any(String));
    expect(contexts[1]).toBe(contexts[0]);
    expect(contexts[2]).not.toBe(contexts[0]);
    unmount();
  });

  it("starts fresh context after switching away from and back to Cool Blob", async () => {
    useFreeBrowseStore.setState({ aiAutoUpdate: false, aiModelId: "segmento-v1" });
    vi.mocked(NVImage.loadFromUrl).mockResolvedValue({ setColormapLabel: vi.fn() } as unknown as NVImage);
    const nv = viewer();
    const ref = { current: nv as unknown as Niivue };
    const { result, unmount } = renderHook(() => useMulticlassAnnotation(ref));
    nv.drawBitmap[0] = 1;
    await act(async () => { await result.current.requestPrediction(); });
    act(() => { useFreeBrowseStore.getState().setAiModel("fiery-flame", "multiclass"); });
    act(() => { useFreeBrowseStore.getState().setAiModel("segmento-v1", "multiclass"); });
    await act(async () => { await result.current.requestPrediction(); });
    const requests = vi.mocked(postAiJson).mock.calls.filter(([url]) => url.includes("/infer/"));
    const contexts = requests.map(([, body]) =>
      (body as { prediction_context_id: string }).prediction_context_id);
    expect(contexts[1]).not.toBe(contexts[0]);
    unmount();
  });

  it("does not restore an old prediction after annotations were cleared and saved", async () => {
    useFreeBrowseStore.setState({ aiActiveSession: {
      session_id: "one", session_name: "one", result_path: "result.nii.gz",
      result_model_id: "fiery-flame",
    } });
    const nv = viewer();
    const { unmount } = renderHook(() => useMulticlassAnnotation({ current: nv as unknown as Niivue }));
    await act(async () => { await vi.advanceTimersByTimeAsync(500); });
    expect(NVImage.loadFromUrl).not.toHaveBeenCalled();
    expect(uploadAiDrawing).not.toHaveBeenCalled();
    unmount();
  });

  it("captures the completed gesture and suppresses a response after Clear", async () => {
    let resolveImage!: (image: NVImage) => void;
    vi.mocked(NVImage.loadFromUrl).mockReturnValue(new Promise((resolve) => { resolveImage = resolve; }));
    const nv = viewer();
    const ref = { current: nv as unknown as Niivue };
    const hook = renderHook(() => useMulticlassAnnotation(ref));
    act(() => {
      nv.drawBitmap[0] = 1;
      nv.onMouseUp({ mouseButtonRightDown: false, mouseButtonCenterDown: false });
    });
    await act(async () => { await vi.advanceTimersByTimeAsync(250); });
    expect(uploadAiDrawing).toHaveBeenCalledOnce();
    expect(vi.mocked(uploadAiDrawing).mock.calls[0][1]).toEqual(new Uint8Array([1, 0, 0, 0]));
    act(() => { useFreeBrowseStore.getState().clearAiAnnotations(); });
    expect(nv.drawBitmap).toEqual(new Uint8Array(4));
    await act(async () => { resolveImage({ setColormapLabel: vi.fn() } as unknown as NVImage); });
    expect(nv.addVolume).not.toHaveBeenCalled();
    hook.unmount();
  });

  it("does not infer after gestures in manual mode, but allows a manual request", async () => {
    useFreeBrowseStore.setState({ aiAutoUpdate: false });
    vi.mocked(NVImage.loadFromUrl).mockResolvedValue({ setColormapLabel: vi.fn() } as unknown as NVImage);
    const nv = viewer();
    const ref = { current: nv as unknown as Niivue };
    const { result, unmount } = renderHook(() => useMulticlassAnnotation(ref));
    act(() => {
      nv.drawBitmap[1] = 1;
      nv.onMouseUp({ mouseButtonRightDown: false, mouseButtonCenterDown: false });
    });
    await act(async () => { await vi.advanceTimersByTimeAsync(500); });
    expect(uploadAiDrawing).not.toHaveBeenCalled();
    await act(async () => { await result.current.requestPrediction(); });
    expect(uploadAiDrawing).toHaveBeenCalledOnce();
    expect(nv.addVolume).toHaveBeenCalledOnce();
    unmount();
  });
});
