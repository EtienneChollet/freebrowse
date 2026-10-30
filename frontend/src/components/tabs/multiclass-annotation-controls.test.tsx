import { beforeEach, describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen, within } from "@testing-library/react";
import { useFreeBrowseStore } from "@/store";
import MulticlassAnnotationControls from "@/components/tabs/multiclass-annotation-controls";

beforeEach(() => {
  useFreeBrowseStore.setState({
    aiClasses: [{ id: 1, name: "Class 1", color: "#b6ff00" }],
    aiAnnotationMode: "multiclass", aiBusy: false, aiError: null, aiAutoUpdate: true,
    aiEditingMode: "joint", aiActiveLabelId: null, aiLabelSaving: false,
    aiLabelDrawingReady: true,
    drawingOptions: { ...useFreeBrowseStore.getState().drawingOptions, mode: "pen", penValue: 1 },
  });
});

function renderControls() {
  const select = vi.fn((penValue: number) => {
    useFreeBrowseStore.getState().setDrawingOptions((value) => ({ ...value, penValue }));
  });
  const undo = vi.fn();
  render(<MulticlassAnnotationControls onPenValueChange={select}
    onDrawModeChange={vi.fn()} onDrawUndo={undo} onRunSegmentation={vi.fn()} />);
  return { select, undo };
}

describe("multiclass controls", () => {
  it.each(["segmento-v1", "open-whale", "grey-plant"])("highlights only the drawing choice in %s binary editing", (modelId) => {
    useFreeBrowseStore.setState({ aiModelId: modelId, aiEditingMode: "per_label", aiActiveLabelId: 1 });
    renderControls();
    expect(screen.getByLabelText("One label at a time")).toBeChecked();
    const foreground = screen.getByRole("group", { name: "Class 1" });
    const background = screen.getByRole("button", { name: "Background for active label" });
    expect(foreground).toHaveClass("bg-accent");
    fireEvent.click(background);
    expect(background).toHaveAttribute("aria-pressed", "true");
    expect(foreground).not.toHaveClass("bg-accent");
    expect(screen.getByText("Active: Background for Class 1")).toBeInTheDocument();
    fireEvent.click(screen.getByLabelText("Name for class 1"));
    expect(background).toHaveAttribute("aria-pressed", "false");
    expect(foreground).toHaveClass("bg-accent");
    expect(screen.getByText("Active: Class 1")).toBeInTheDocument();
  });

  it("adds and selects classes through 20, allows names and swatches to be edited", () => {
    const { select } = renderControls();
    expect(screen.getByLabelText("Color for Class 1")).toHaveValue("#b6ff00");
    for (let index = 1; index < 20; index++) {
      fireEvent.click(screen.getByRole("button", { name: "Add new class" }));
    }
    expect(select).toHaveBeenLastCalledWith(20);
    expect(screen.getByLabelText("Name for class 20")).toHaveValue("Class 20");
    const name = screen.getByLabelText("Name for class 20");
    fireEvent.change(name, { target: { value: "Liver" } });
    fireEvent.blur(name);
    fireEvent.change(screen.getByLabelText("Color for Liver"), { target: { value: "#00ff00" } });
    expect(useFreeBrowseStore.getState().aiClasses[19]).toEqual({ id: 20, name: "Liver", color: "#00ff00" });
    fireEvent.change(screen.getByPlaceholderText("Search labels…"), { target: { value: "Liver" } });
    expect(screen.queryByLabelText("Name for class 1")).not.toBeInTheDocument();
  });

  it("keeps drawing, background and undo available while predictions are running", () => {
    useFreeBrowseStore.setState({ aiBusy: true });
    const { select, undo } = renderControls();
    expect(screen.getByRole("button", { name: "Updating prediction…" })).toBeDisabled();
    fireEvent.click(screen.getByRole("button", { name: "Background" }));
    expect(select).toHaveBeenLastCalledWith(255);
    fireEvent.click(screen.getByRole("button", { name: "Undo" }));
    expect(undo).toHaveBeenCalledOnce();
    fireEvent.click(screen.getByLabelText("Auto-update prediction"));
    expect(useFreeBrowseStore.getState().aiAutoUpdate).toBe(false);
  });

  it("keeps Background in the label list even when foreground labels are filtered", () => {
    renderControls();
    for (let index = 1; index < 6; index++) {
      fireEvent.click(screen.getByRole("button", { name: "Add new class" }));
    }
    fireEvent.change(screen.getByPlaceholderText("Search labels…"), {
      target: { value: "no matching foreground" },
    });
    const labels = screen.getByLabelText("Annotation labels");
    fireEvent.click(within(labels).getByRole("button", { name: "Background" }));
    expect(useFreeBrowseStore.getState().drawingOptions.penValue).toBe(255);
    expect(screen.queryByRole("button", { name: "Delete Background" })).not.toBeInTheDocument();
    expect(screen.getByText("Active: Background")).toBeInTheDocument();
  });

  it("deletes active and inactive classes without changing surviving IDs", () => {
    const { select } = renderControls();
    fireEvent.click(screen.getByRole("button", { name: "Add new class" }));
    fireEvent.click(screen.getByRole("button", { name: "Delete Class 1" }));
    expect(select).toHaveBeenLastCalledWith(2);
    expect(useFreeBrowseStore.getState().aiClasses.map(({ id }) => id)).toEqual([2]);
    fireEvent.click(screen.getByRole("button", { name: "Delete Class 2" }));
    expect(select).toHaveBeenLastCalledWith(255);
    expect(useFreeBrowseStore.getState().aiClasses).toEqual([]);
    fireEvent.click(screen.getByRole("button", { name: "Add new class" }));
    expect(screen.getByLabelText("Name for class 1")).toHaveValue("Class 1");
    expect(select).toHaveBeenLastCalledWith(1);
  });

  it("clears a label filter when deletion hides the search box", () => {
    renderControls();
    for (let index = 1; index < 6; index++) {
      fireEvent.click(screen.getByRole("button", { name: "Add new class" }));
    }
    fireEvent.change(screen.getByPlaceholderText("Search labels…"), {
      target: { value: "Class 6" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Delete Class 6" }));
    expect(screen.queryByPlaceholderText("Search labels…")).not.toBeInTheDocument();
    expect(screen.getByLabelText("Name for class 1")).toBeInTheDocument();
    expect(screen.getByLabelText("Name for class 5")).toBeInTheDocument();
  });
});
