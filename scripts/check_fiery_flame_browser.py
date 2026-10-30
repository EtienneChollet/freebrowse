"""
Exercise multiclass inference and saved overlays in the real FreeBrowse browser UI.
"""

import argparse
import json
import os
from pathlib import Path
from typing import List, Sequence

import nibabel as nib
from playwright.sync_api import ConsoleMessage, Page, sync_playwright


def check_annotation_labels(page: Page, output: Path) -> None:
    """
    Check label deletion, background gestures, and Undo using the real viewer.
    """
    page.get_by_label("Auto-update prediction", exact=True).uncheck()
    for _ in range(19):
        page.get_by_role("button", name="Add new class", exact=True).click()
    labels = page.get_by_label("Annotation labels", exact=True)
    assert labels.get_by_role("button", name="Background", exact=True).count() == 1
    assert page.get_by_role("button", name="Delete Background", exact=True).count() == 0
    page.get_by_placeholder("Search labels…").fill("Class 20")
    background = labels.get_by_role("button", name="Background", exact=True)
    background.click()
    assert background.get_attribute("aria-pressed") == "true"
    position = page.evaluate("""() => {
      const nv = window.__browserCheckNv;
      const frac = [0.5, 0.5, 0.5];
      nv.scene.crosshairPos = frac;
      nv.drawScene();
      const pos = nv.frac2canvasPos(frac);
      const rect = nv.canvas.getBoundingClientRect();
      return [rect.x + pos[0] / devicePixelRatio, rect.y + pos[1] / devicePixelRatio];
    }""")
    page.mouse.move(position[0] - 12, position[1])
    page.mouse.down()
    page.mouse.move(position[0] + 12, position[1], steps=12)
    page.mouse.up()
    assert page.evaluate("window.__browserCheckNv.drawBitmap.some(value => value === 255)")
    page.get_by_label("Name for class 20", exact=True).click()
    page.mouse.click(position[0], position[1] + 30)
    assert page.evaluate("window.__browserCheckNv.drawBitmap.some(value => value === 20)")
    page.get_by_placeholder("Search labels…").fill("")
    page.screenshot(path=str(output / "twenty-labels.png"))
    page.get_by_role("button", name="Delete Class 20", exact=True).click()
    page.wait_for_function("!window.__browserCheckNv.drawBitmap.some(value => value === 20)")
    page.get_by_role("button", name="Undo", exact=True).click()
    assert not page.evaluate("window.__browserCheckNv.drawBitmap.some(value => value === 20)")
    assert page.evaluate("window.__browserCheckNv.drawBitmap.some(value => value === 255)")
    print("deleted marked class; Undo kept background without restoring the class", flush=True)
    for label in range(19, 0, -1):
        page.get_by_role("button", name=f"Delete Class {label}", exact=True).click()
    assert background.get_attribute("aria-pressed") == "true"
    page.get_by_role("button", name="Add new class", exact=True).click()
    assert page.get_by_label("Name for class 1", exact=True).input_value() == "Class 1"
    page.get_by_role("button", name="Add new class", exact=True).click()
    page.get_by_role("button", name="Delete Class 1", exact=True).click()
    page.screenshot(path=str(output / "deleted-labels.png"))
    print("label controls passed: background scribble, deletion, Undo, and empty list", flush=True)


def check_prediction_deletion(page: Page, output: Path) -> None:
    """
    Delete a predicted class and verify other masks and marks survive without an inference request.
    """
    requests: List[str] = []
    page.on("request", lambda request: requests.append(request.url))
    has_labels = page.evaluate("""() => {
      const nv = window.__browserCheckNv;
      window.__predictionBeforeDeletion = nv.volumes[1].img.slice();
      window.__marksBeforeDeletion = nv.drawBitmap.slice();
      return [nv.volumes[1].img.includes(1), nv.volumes[1].img.includes(2)];
    }""")
    assert all(has_labels), has_labels
    page.get_by_role("button", name="Delete Right kidney", exact=True).click()
    page.wait_for_function("""() => {
      const nv = window.__browserCheckNv;
      return nv.volumes.length === 2 && !nv.volumes[1].img.includes(2)
        && !nv.drawBitmap.includes(2);
    }""")
    page.wait_for_timeout(1000)
    assert not any("/infer/" in url for url in requests), requests
    is_preserved = page.evaluate("""() => {
      const nv = window.__browserCheckNv;
      const prediction = window.__predictionBeforeDeletion;
      const marks = window.__marksBeforeDeletion;
      for (let index = 0; index < prediction.length; index++) {
        const expected = prediction[index] === 2 ? 0 : prediction[index];
        if (nv.volumes[1].img[index] !== expected) return false;
      }
      for (let index = 0; index < marks.length; index++) {
        const expected = marks[index] === 2 ? 0 : marks[index];
        if (nv.drawBitmap[index] !== expected) return false;
      }
      return true;
    }""")
    assert is_preserved
    page.screenshot(path=str(output / "prediction-after-deletion.png"))
    print("deletion preserved other predictions and marks; no inference requested", flush=True)


def main() -> None:
    """
    Capture a real scan session, optionally testing clicks, scribbles, correction, and reload.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8102")
    parser.add_argument("--model", default="fiery-flame", choices=["fiery-flame", "cool-blob"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hardware-gpu", action="store_true")
    parser.add_argument("--exercise", action="store_true")
    parser.add_argument("--check-labels", action="store_true")
    parser.add_argument("--check-deletion", action="store_true")
    parser.add_argument("--drop-file", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    errors = []
    console = []
    requests = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            executable_path="/usr/bin/google-chrome",
            headless=True,
            chromium_sandbox=True,
            ignore_default_args=True,
            args=["--headless=new", "--no-first-run", "--force-gpu-mem-available-mb=4096",
                  "--remote-debugging-pipe",
                  f"--user-data-dir={args.output.resolve() / 'chrome-profile'}"]
            + (["--enable-gpu", "--use-angle=vulkan"] if args.hardware_gpu else []),
            timeout=30000,
            env={key: value for key, value in os.environ.items() if key != "LD_LIBRARY_PATH"},
        )
        page = browser.new_page(viewport={"width": 1440, "height": 1000})
        page.add_init_script("""(() => {
          const getExtension = WebGL2RenderingContext.prototype.getExtension;
          WebGL2RenderingContext.prototype.getExtension = function(name) {
            const extension = getExtension.call(this, name);
            if (name === 'WEBGL_lose_context' && extension && !extension.traced) {
              const lose = extension.loseContext.bind(extension);
              extension.loseContext = () => {
                console.error('Explicit context loss: ' + new Error().stack); lose();
              };
              extension.traced = true;
            }
            return extension;
          };
        })();""")
        def capture_console(message: ConsoleMessage) -> None:
            """
            Record browser messages and expose the viewer for geometry-aware interaction checks.
            """
            console.append({"type": message.type, "text": message.text[:400]})
            if message.text.startswith("Niivue attached to canvas"):
                message.args[1].evaluate("nv => { window.__browserCheckNv = nv; }")

        page.on("pageerror", lambda error: errors.append(str(error)))
        page.on("console", capture_console)
        page.on("response", lambda response: requests.append({
            "status": response.status, "url": response.url,
        }))
        page.on("requestfailed", lambda request: print(
            "request_failed=" + json.dumps({"url": request.url, "error": request.failure}),
            flush=True,
        ))
        query = f"/?model={args.model}&tab=aiAnnotation"
        if not args.drop_file:
            query += "&vol=/data/amos_0223.nii.gz"
        page.goto(args.url + query, timeout=120000)
        if args.drop_file:
            page.evaluate("""() => {
              const input = document.createElement('input');
              input.type = 'file'; input.id = 'browser-drop-source'; input.hidden = true;
              document.body.appendChild(input);
            }""")
            page.locator("#browser-drop-source").set_input_files(args.drop_file)
            transfer = page.evaluate_handle("""() => {
              const transfer = new DataTransfer();
              transfer.items.add(document.querySelector('#browser-drop-source').files[0]);
              return transfer;
            }""")
            page.locator("div.border-dashed").dispatch_event("drop", {"dataTransfer": transfer})
        page.get_by_role("button", name="New Session", exact=True).wait_for(timeout=120000)
        page.wait_for_function("window.__browserCheckNv?.volumes.length > 0", timeout=120000)
        page.wait_for_timeout(2000)
        before_session = page.evaluate("""() => ({
          lost: window.__browserCheckNv.gl.isContextLost(),
          volumes: window.__browserCheckNv.volumes.length
        })""")
        print("before_session=" + json.dumps(before_session), flush=True)
        page.screenshot(path=str(args.output / "before-session.png"))
        with page.expect_response("**/ai/session/new") as created:
            page.get_by_role("button", name="New Session", exact=True).click(timeout=120000)
            if args.drop_file:
                page.get_by_role("button", name="Upload", exact=True).click(timeout=30000)
        session = created.value.json()
        print("created_session=" + json.dumps(session), flush=True)
        page.get_by_role("button", name="Add new class", exact=True).wait_for(timeout=120000)
        background = page.get_by_role("button", name="Background", exact=True)
        assert background.get_attribute("aria-pressed") == "true"
        page.get_by_text("Axial", exact=True).click()
        page.wait_for_timeout(5000)
        page.screenshot(path=str(args.output / "initial.png"))
        if args.check_labels:
            check_annotation_labels(page, args.output)
            page.get_by_role("button", name="Save & Exit", exact=True).click()
            saved_option = page.locator(f'option[value="{session["session_id"]}"]')
            page.locator("select").filter(has=saved_option).select_option(session["session_id"])
            page.get_by_role("button", name="Load Session", exact=True).click()
            page.get_by_label("Name for class 2", exact=True).wait_for(timeout=120000)
            assert page.get_by_label("Name for class 1", exact=True).count() == 0
            assert page.get_by_text("Active: Background", exact=True).count() == 1
            assert not errors, errors
            assert not page.evaluate("window.__browserCheckNv.gl.isContextLost()")
            page.screenshot(path=str(args.output / "reloaded-labels.png"))
        if args.exercise or args.check_deletion:
            if page.evaluate("window.__browserCheckNv.gl.isContextLost()"):
                raise RuntimeError("The browser lost its graphics context")
            source = nib.load(
                "/autofs/cluster/dalcalab1/data/raw_data/AMOS/original_unzipped/"
                "retrieved_2024_05_26/amos22/imagesVa/amos_0223.nii.gz"
            )

            def position(voxel: Sequence[int]) -> List[float]:
                """
                Locate a native scan voxel on the displayed slice for a real mouse gesture.
                """
                world = nib.affines.apply_affine(source.affine, voxel).tolist()
                return page.evaluate("""world => {
                  const nv = window.__browserCheckNv;
                  const frac = nv.mm2frac(world);
                  nv.scene.crosshairPos = frac;
                  nv.drawScene();
                  const pos = nv.frac2canvasPos(frac);
                  const rect = nv.canvas.getBoundingClientRect();
                  return [rect.x + pos[0] / devicePixelRatio, rect.y + pos[1] / devicePixelRatio];
                }""", world)

            def wait_prediction() -> None:
                """
                Require an actual prediction overlay and report visible inference errors.
                """
                page.get_by_role("button", name="Run Segmentation", exact=True).wait_for(
                    timeout=180000,
                )
                page.wait_for_function(
                    "window.__browserCheckNv.volumes.length === 2", timeout=180000,
                )
                if page.get_by_role("alert").count():
                    raise RuntimeError(page.get_by_role("alert").inner_text())
                print("prediction rendered", flush=True)

            page.get_by_label("Auto-update prediction", exact=True).uncheck()
            page.get_by_role("button", name="Background", exact=True).click()
            page.mouse.click(*position([20, 20, 182]))
            page.get_by_label("Name for class 1", exact=True).fill("Liver")
            page.get_by_label("Name for class 1", exact=True).press("Enter")
            page.mouse.click(*position([157, 343, 182]))
            with page.expect_response(f"**/infer/{args.model}", timeout=180000) as inferred:
                page.get_by_label("Auto-update prediction", exact=True).check()
            if inferred.value.status != 200:
                raise RuntimeError(inferred.value.text())
            wait_prediction()
            page.screenshot(path=str(args.output / "one-class-prediction.png"))
            with page.expect_response(f"**/infer/{args.model}", timeout=180000) as inferred:
                page.mouse.click(*position([167, 343, 182]))
            if inferred.value.status != 200:
                raise RuntimeError(inferred.value.text())
            wait_prediction()
            page.screenshot(path=str(args.output / "corrected-prediction.png"))
            page.get_by_role("button", name="Add new class", exact=True).click()
            page.get_by_label("Name for class 2", exact=True).fill("Right kidney")
            page.get_by_label("Name for class 2", exact=True).press("Enter")
            x, y = position([139, 200, 138])
            with page.expect_response(f"**/infer/{args.model}", timeout=180000) as inferred:
                page.mouse.move(x - 6, y)
                page.mouse.down()
                page.mouse.move(x + 6, y + 3, steps=12)
                page.mouse.up()
            if inferred.value.status != 200:
                raise RuntimeError(inferred.value.text())
            wait_prediction()
            page.screenshot(path=str(args.output / "two-class-prediction.png"))
            if args.check_deletion:
                check_prediction_deletion(page, args.output)
            page.get_by_label("Show prediction", exact=True).uncheck()
            if page.evaluate("window.__browserCheckNv.volumes[1].opacity") != 0:
                raise RuntimeError("Prediction visibility control did not hide the overlay")
            page.get_by_label("Show prediction", exact=True).check()
            page.get_by_role("button", name="Save & Exit", exact=True).click()
            page.get_by_role("button", name="New Session", exact=True).wait_for(timeout=30000)
            saved_option = page.locator(f'option[value="{session["session_id"]}"]')
            page.locator("select").filter(has=saved_option).select_option(session["session_id"])
            page.get_by_role("button", name="Load Session", exact=True).click()
            saved_label = "Name for class 1" if args.check_deletion else "Name for class 2"
            page.get_by_label(saved_label, exact=True).wait_for(timeout=120000)
            wait_prediction()
            if page.get_by_label("Name for class 1", exact=True).input_value() != "Liver":
                raise RuntimeError("Saved class name did not survive reload")
            page.screenshot(path=str(args.output / "saved-session.png"))
            print("saved session=" + json.dumps(session), flush=True)
        (args.output / "page.txt").write_text(page.locator("body").inner_text())
        report = {
            "errors": errors, "console": console, "requests": requests,
            "session": session, "canvas": page.locator("canvas").bounding_box(),
        }
        (args.output / "browser.json").write_text(json.dumps(report, indent=2))
        assert not errors, errors
        print(json.dumps({"errors": errors, "console": console[:15]}), flush=True)
        browser.close()


if __name__ == "__main__":
    main()
