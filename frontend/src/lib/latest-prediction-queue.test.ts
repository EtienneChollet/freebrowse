import { describe, expect, it, vi } from "vitest";
import { LatestPredictionQueue } from "@/lib/latest-prediction-queue";

function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((done) => { resolve = done; });
  return { promise, resolve };
}

describe("latest prediction queue", () => {
  it("serializes inference, skips intermediate edits, and suppresses stale results", async () => {
    const first = deferred<number>();
    const last = deferred<number>();
    const predict = vi.fn().mockReturnValueOnce(first.promise).mockReturnValueOnce(last.promise);
    const apply = vi.fn();
    const queue = new LatestPredictionQueue<number, number>(predict, apply, vi.fn(), vi.fn());
    queue.enqueue(1);
    queue.enqueue(2);
    queue.enqueue(3);
    expect(predict.mock.calls).toEqual([[1]]);
    first.resolve(10);
    await Promise.resolve();
    expect(apply).not.toHaveBeenCalled();
    expect(predict.mock.calls).toEqual([[1], [3]]);
    last.resolve(30);
    await Promise.resolve();
    expect(apply.mock.calls).toEqual([[30, 3]]);
  });

  it("clear or changing sessions prevents an old response from restoring an overlay", async () => {
    const pending = deferred<number>();
    const apply = vi.fn();
    const queue = new LatestPredictionQueue<number, number>(() => pending.promise, apply, vi.fn(), vi.fn());
    queue.enqueue(1);
    queue.enqueue(2);
    queue.invalidate();
    pending.resolve(10);
    await Promise.resolve();
    expect(apply).not.toHaveBeenCalled();
  });
});
