import type { AiClass } from "@/store/ai-slice";
import type { NVImage } from "@niivue/niivue";

export const BACKGROUND_LABEL = 255;

export function supportsPerLabelEditing(modelId: string): boolean {
  return ["segmento-v1", "open-whale", "grey-plant"].includes(modelId);
}

const CLASS_COLORS = [
  "#b6ff00", "#00cfff", "#ff50ca", "#ffad28", "#a080ff", "#00efad", "#ff5858",
  "#fff040", "#55a5ff", "#ef80ff", "#72ff71", "#ff8270", "#32ffff", "#dcbeff",
  "#f3ff80", "#ff7dae", "#7ce0ff", "#ffbf78", "#a4ffcf", "#d1a0ff",
];

export function newAnnotationClass(classes: AiClass[]): AiClass {
  const id = Math.max(0, ...classes.map((entry) => entry.id)) + 1;
  const used = new Set(classes.map((entry) => entry.color.toLowerCase()));
  let color = CLASS_COLORS.find((candidate) => !used.has(candidate));
  if (!color) {
    let index = id;
    do {
      const hue = (index++ * 137.508) % 360;
      const channel = (offset: number) => {
        const k = (offset + hue / 30) % 12;
        return Math.round(255 * (0.6 - 0.4 * Math.max(-1, Math.min(k - 3, 9 - k, 1))))
          .toString(16).padStart(2, "0");
      };
      color = `#${channel(0)}${channel(8)}${channel(4)}`;
    } while (used.has(color));
  }
  return { id, name: `Class ${id}`, color };
}

export function predictionColor(color: string): string {
  const rgb = [1, 3, 5].map((offset) => parseInt(color.slice(offset, offset + 2), 16));
  const center = (Math.max(...rgb) + Math.min(...rgb)) / 2;
  return "#" + rgb.map((value) => Math.round(center + (value - center) * 0.45)
    .toString(16).padStart(2, "0")).join("");
}

/** Clear removed classes from a prediction in memory, preserving every other voxel. */
export function retainPredictionLabels(image: NVImage, classes: AiClass[]): void {
  if (!image.img) return;
  const ids = new Set([0, ...classes.map((entry) => entry.id)]);
  for (let index = 0; index < image.img.length; index++) {
    if (!ids.has(image.img[index])) image.img[index] = 0;
  }
}

export function annotationColormap(classes: AiClass[], prediction = false) {
  const entries = [{ id: 0, name: "Unmarked", color: "#000000" }, ...classes];
  if (!prediction) entries.push({ id: BACKGROUND_LABEL, name: "Background", color: "#ffffff" });
  const colors = entries.map((entry) => prediction ? predictionColor(entry.color) : entry.color);
  return {
    R: colors.map((color) => parseInt(color.slice(1, 3), 16)),
    G: colors.map((color) => parseInt(color.slice(3, 5), 16)),
    B: colors.map((color) => parseInt(color.slice(5, 7), 16)),
    A: entries.map((entry) => entry.id === 0 ? 0 : 255),
    I: entries.map((entry) => entry.id),
    labels: entries.map((entry) => entry.name),
  };
}
