/** Run one prediction at a time, keep only the latest queued edit, and discard stale results. */
export class LatestPredictionQueue<Input, Output> {
  private version = 0;
  private pending: { input: Input; version: number } | null = null;
  private running = false;

  constructor(
    private predict: (input: Input) => Promise<Output>,
    private apply: (output: Output, input: Input) => void,
    private fail: (error: unknown) => void,
    private busy: (running: boolean) => void,
  ) {}

  invalidate() {
    this.version++;
    this.pending = null;
  }

  enqueue(input: Input) {
    this.pending = { input, version: ++this.version };
    void this.drain();
  }

  private async drain() {
    if (this.running) return;
    this.running = true;
    this.busy(true);
    try {
      while (this.pending) {
        const next = this.pending;
        this.pending = null;
        try {
          const output = await this.predict(next.input);
          if (next.version === this.version) this.apply(output, next.input);
        } catch (error) {
          if (next.version === this.version) this.fail(error);
        }
      }
    } finally {
      this.running = false;
      this.busy(false);
    }
  }
}
