import { useState } from "react";
import { Brain, Plus, Undo, Trash2, X, Lock, Unlock } from "lucide-react";
import { useFreeBrowseStore } from "@/store";
import type { AiClass } from "@/store/ai-slice";
import { newAnnotationClass, BACKGROUND_LABEL, supportsPerLabelEditing } from "@/lib/ai-labels";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Checkbox } from "@/components/ui/checkbox";
import { Select } from "@/components/ui/select";
import { LabeledSliderWithInput } from "@/components/ui/labeled-slider-with-input";

type Props = {
  onPenValueChange: (value: number) => void;
  onDrawModeChange: (mode: "none" | "pen" | "wand") => void;
  onDrawUndo: () => void;
  onRunSegmentation: () => void;
};

export default function MulticlassAnnotationControls(props: Props) {
  const classes = useFreeBrowseStore((s) => s.aiClasses);
  const setClasses = useFreeBrowseStore((s) => s.setAiClasses);
  const drawing = useFreeBrowseStore((s) => s.drawingOptions);
  const autoUpdate = useFreeBrowseStore((s) => s.aiAutoUpdate);
  const setAutoUpdate = useFreeBrowseStore((s) => s.setAiAutoUpdate);
  const visible = useFreeBrowseStore((s) => s.aiPredictionVisible);
  const setVisible = useFreeBrowseStore((s) => s.setAiPredictionVisible);
  const opacity = useFreeBrowseStore((s) => s.aiPredictionOpacity);
  const setOpacity = useFreeBrowseStore((s) => s.setAiPredictionOpacity);
  const modality = useFreeBrowseStore((s) => s.aiModality);
  const setModality = useFreeBrowseStore((s) => s.setAiModality);
  const clear = useFreeBrowseStore((s) => s.clearAiAnnotations);
  const busy = useFreeBrowseStore((s) => s.aiBusy);
  const error = useFreeBrowseStore((s) => s.aiError);
  const modelId = useFreeBrowseStore((s) => s.aiModelId);
  const editingMode = useFreeBrowseStore((s) => s.aiEditingMode);
  const activeLabelId = useFreeBrowseStore((s) => s.aiActiveLabelId);
  const actions = useFreeBrowseStore((s) => s.aiLabelActions);
  const saving = useFreeBrowseStore((s) => s.aiLabelSaving);
  const drawingReady = useFreeBrowseStore((s) => s.aiLabelDrawingReady);
  const binary = editingMode === "per_label";
  const active = classes.find((entry) => entry.id === activeLabelId);
  const locked = binary && Boolean(active?.locked);
  const hasLocks = classes.some((entry) => entry.locked);
  const [search, setSearch] = useState("");
  const selected = binary ? active : classes.find((entry) => entry.id === drawing.penValue);
  const selectedName = binary && drawing.penValue === BACKGROUND_LABEL
    ? `Background for ${active?.name ?? "active label"}` : selected?.name ?? "Background";
  const updateClass = (id: number, changes: Partial<AiClass>) => {
    const entry = classes.find((item) => item.id === id);
    if (!entry || Object.entries(changes).every(([key, value]) =>
      entry[key as keyof AiClass] === value)) return;
    if (binary) { void actions?.update(id, changes); return; }
    setClasses(classes.map((entry) => entry.id === id ? { ...entry, ...changes } : entry));
  };
  const choose = (id: number) => {
    if (saving) return;
    if (binary && id !== BACKGROUND_LABEL && (id !== activeLabelId || !drawingReady)) {
      void actions?.select(id);
      return;
    }
    if (locked || saving || !drawingReady) return;
    props.onPenValueChange(id);
    if (drawing.mode !== "pen") props.onDrawModeChange("pen");
  };
  const addClass = () => {
    const entry = newAnnotationClass(classes);
    if (entry.id > 254) return;
    if (binary) { void actions?.add(entry); setSearch(""); return; }
    setClasses([...classes, entry]);
    setSearch("");
    choose(entry.id);
    requestAnimationFrame(() => {
      document.getElementById(`ai-class-${entry.id}`)?.scrollIntoView({ block: "nearest" });
    });
  };
  const removeClass = (id: number) => {
    if (binary) { void actions?.remove(id); return; }
    const remaining = classes.filter((entry) => entry.id !== id);
    setClasses(remaining);
    if (remaining.length <= 5) setSearch("");
    if (drawing.penValue === id) choose(remaining[0]?.id ?? BACKGROUND_LABEL);
  };

  return <>
    <div className="px-4 pb-3 space-y-3 shrink-0">
      {supportsPerLabelEditing(modelId) && <>
        <div className="flex items-center gap-2">
          <Checkbox id="ai-binary-editing" checked={binary} disabled={saving || hasLocks || !actions}
            onCheckedChange={(checked) => { void actions?.setMode(checked ? "per_label" : "joint"); }} />
          <Label htmlFor="ai-binary-editing">One label at a time</Label>
        </div>
        {binary && <p className="text-xs text-muted-foreground">
          Locked labels remain unchanged.
          {hasLocks && " Unlock labels to turn this option off."}
        </p>}
      </>}
      <div className="flex gap-2 items-center">
        <Label htmlFor="ai-modality">Scan</Label>
        <Select id="ai-modality" value={modality} disabled={binary}
          onChange={(event) => setModality(event.target.value as "ct" | "other")}>
          <option value="ct">CT (Hounsfield units)</option>
          <option value="other">Other modality</option>
        </Select>
      </div>
      <Select aria-label="Annotation tool" value={locked ? "none" : drawing.mode}
        disabled={locked || saving || !drawingReady}
        onChange={(event) => props.onDrawModeChange(event.target.value as "none" | "pen")}>
        <option value="pen">Annotate</option><option value="none">Navigate</option>
      </Select>
      <p className="text-xs text-muted-foreground">Click for a dot; drag for a scribble.</p>
    </div>
    <div className={"flex flex-col flex-1 px-4 gap-2 " + (binary ? "min-h-64" : "min-h-24")}>
      <Label htmlFor="ai-label-search">Annotation label</Label>
      <p className="text-xs text-muted-foreground">Active: {selectedName}
        {locked && " · Locked"}</p>
      {classes.length > 5 && <Input id="ai-label-search" type="search" placeholder="Search labels…"
        value={search} onChange={(event) => setSearch(event.target.value)} />}
      <div className="flex flex-1 min-h-16 flex-col gap-1"
        aria-label="Annotation labels">
        <button type="button" aria-pressed={drawing.penValue === BACKGROUND_LABEL}
          disabled={locked || saving || !drawingReady}
          className={"flex w-full shrink-0 items-center gap-2 rounded-md border " +
            "px-2 py-1 text-sm " + (drawing.penValue === BACKGROUND_LABEL
              ? "border-foreground/60 bg-accent" : "border-transparent bg-background")}
          onClick={() => choose(BACKGROUND_LABEL)}>
          <span className="flex h-6 w-6 shrink-0 items-center justify-center" aria-hidden="true">
            <span className="h-4 w-4 border border-muted-foreground bg-white" />
          </span>
          {binary ? "Background for active label" : "Background"}
        </button>
        <div className={"flex-1 overflow-y-auto space-y-1 pr-1 " + (binary ? "min-h-32" : "min-h-0")}>
        {classes.filter((entry) => entry.name.toLowerCase().includes(search.toLowerCase()))
          .map((entry) =>
          <div key={entry.id} id={`ai-class-${entry.id}`} role="group" aria-label={entry.name}
            className={"flex items-center gap-2 rounded-md border px-2 py-1 " +
              (drawing.penValue === entry.id
                ? "border-foreground/60 bg-accent" : "border-transparent")}
            onClick={() => choose(entry.id)}>
            <input type="color" value={entry.color} aria-label={`Color for ${entry.name}`}
              className="h-6 w-6 shrink-0 cursor-pointer border-0 bg-transparent p-0"
              disabled={saving} onFocus={() => choose(entry.id)}
              onChange={(event) => updateClass(entry.id, { color: event.target.value })} />
            <input aria-label={`Name for class ${entry.id}`}
              defaultValue={entry.name} maxLength={100}
              className="min-w-0 w-full bg-transparent text-sm outline-none focus:underline"
              disabled={saving}
              onFocus={() => choose(entry.id)} onBlur={(event) => {
                const name = event.target.value.trim() || `Class ${entry.id}`;
                event.target.value = name;
                updateClass(entry.id, { name });
              }} onKeyDown={(event) => {
                if (event.key === "Enter") event.currentTarget.blur();
              }} />
            {binary && <Button type="button" variant="ghost" size="icon"
              className="h-6 w-6 shrink-0" disabled={saving}
              aria-label={`${entry.locked ? "Unlock" : "Accept and lock"} ${entry.name}`}
              title={entry.locked ? "Unlock to edit" : "Accept and lock this structure"}
              onClick={(event) => {
                event.stopPropagation(); void actions?.lock(entry.id, !entry.locked);
              }}>
              {entry.locked ? <Lock className="h-4 w-4" /> : <Unlock className="h-4 w-4" />}
            </Button>}
            <Button type="button" variant="ghost" size="icon"
              disabled={saving || (binary && (entry.locked || classes.length === 1))}
              className="h-6 w-6 shrink-0" aria-label={`Delete ${entry.name}`}
              title={`Delete ${entry.name} and its annotations`}
              onClick={(event) => { event.stopPropagation(); removeClass(entry.id); }}>
              <X className="h-4 w-4" />
            </Button>
          </div>)}
        <Button variant="ghost" className="w-full justify-start" onClick={addClass}
          disabled={classes.length >= 254 || saving}>
          <Plus className="mr-2 h-4 w-4" />Add new class
        </Button>
        </div>
      </div>
    </div>
    <div className="p-4 border-t shrink-0 space-y-3">
      <div className="flex gap-2">
        <Button variant="outline" className="flex-1" onClick={props.onDrawUndo}
          disabled={locked || saving || !drawingReady}>
          <Undo className="h-4 w-4 mr-2" />Undo
        </Button>
        <Button variant="outline" className="flex-1"
          disabled={locked || saving || !drawingReady}
          onClick={() => { if (binary) void actions?.clear(); else clear(); }}>
          <Trash2 className="h-4 w-4 mr-2" />{binary ? "Clear label" : "Clear"}
        </Button>
      </div>
      <div className="flex items-center gap-2">
        <Checkbox id="ai-auto-update" checked={autoUpdate} onCheckedChange={setAutoUpdate} />
        <Label htmlFor="ai-auto-update">Auto-update prediction</Label>
      </div>
      <div className="flex items-center gap-2">
        <Checkbox id="ai-show-prediction" checked={visible} onCheckedChange={setVisible} />
        <Label htmlFor="ai-show-prediction">Show prediction</Label>
      </div>
      <LabeledSliderWithInput label="Prediction opacity" value={opacity} onValueChange={setOpacity}
        min={0} max={1} step={0.05} />
      <Button className="w-full" onClick={props.onRunSegmentation}
        disabled={busy || saving || locked || !drawingReady}>
        <Brain className="h-4 w-4 mr-2" />{busy ? "Updating prediction…" : "Run Segmentation"}
      </Button>
      {error && <p role="alert" className="text-xs text-destructive break-words">{error}</p>}
    </div>
  </>;
}
