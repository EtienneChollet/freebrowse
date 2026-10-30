import { describe, expect, it, vi } from "vitest";
import { act, renderHook } from "@testing-library/react";
import type { Niivue } from "@niivue/niivue";
import { useDrawing } from "@/hooks/use-drawing";

describe("drawing Undo", () => {
  it("does not call Niivue when the previous undo slot was cleared after deleting a label", () => {
    const nv = {
      drawUndoBitmaps: [new Uint8Array([1, 1]), new Uint8Array()],
      currentDrawUndoBitmap: 0, drawUndo: vi.fn(),
    };
    const ref = { current: nv as unknown as Niivue };
    const { result } = renderHook(() => useDrawing(ref, vi.fn()));
    act(() => { result.current.handleDrawUndo(); });
    expect(nv.drawUndo).not.toHaveBeenCalled();
    nv.drawUndoBitmaps[1] = new Uint8Array([1, 255]);
    nv.currentDrawUndoBitmap = 1;
    act(() => { result.current.handleDrawUndo(); });
    expect(nv.drawUndo).toHaveBeenCalledOnce();
  });
});
