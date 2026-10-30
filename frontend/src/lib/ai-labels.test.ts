import { describe, expect, it } from "vitest";
import { annotationColormap, newAnnotationClass, predictionColor } from "@/lib/ai-labels";
import type { AiClass } from "@/store/ai-slice";

describe("multiclass colors", () => {
  it("starts lime green and assigns 20 distinct colors with stable sequential IDs", () => {
    const classes: AiClass[] = [];
    for (let index = 0; index < 20; index++) classes.push(newAnnotationClass(classes));
    expect(classes[0]).toEqual({ id: 1, name: "Class 1", color: "#b6ff00" });
    expect(classes[19].name).toBe("Class 20");
    expect(new Set(classes.map((entry) => entry.color)).size).toBe(20);
  });

  it("keeps annotations opaque, distinguishes background, and softens only prediction saturation", () => {
    const classes = [{ id: 17, name: "Liver", color: "#ff0000" }];
    const prompts = annotationColormap(classes);
    const prediction = annotationColormap(classes, true);
    expect(prompts.I).toEqual([0, 17, 255]);
    expect(prompts.A).toEqual([0, 255, 255]);
    expect(prediction.I).toEqual([0, 17]);
    expect(predictionColor("#ff0000")).toBe("#b94646");
  });
});
