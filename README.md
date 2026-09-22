# grip_ocr

A local GUI app that reads the 7-segment LCD of a grip dynamometer (GD DYNO-200) through a webcam.
It uses no OCR models, LLMs or cloud services: each segment point is simply judged on or off.

## Build and run

Requires Rust 1.92 or later (eframe 0.35 MSRV). eframe 0.36 needs Rust 1.95, so it is pinned at 0.35. Verified with 1.94.1.

```sh
cargo build --release
./target/release/grip_ocr                      # camera 0, 1280x720 @30
./target/release/grip_ocr --camera 1 --size 1920x1080 --fps 30 --profile my.json
./target/release/grip_ocr --make-demo demo     # generate a synthetic recording + profile.json
./target/release/grip_ocr --video demo         # replay without a camera (same pipeline)
cargo test --no-default-features               # decoder/session tests only, no GUI deps
```

Profile lookup order: `--profile`, then `<video>/profile.json`, then the parent folder of `--video`, then `./profile.json`.

### Linux (V4L2)
- Build dependencies: `sudo apt install libclang-dev libv4l-dev libxkbcommon-dev libxcb-render0-dev libxcb-shape0-dev libxcb-xfixes0-dev`
- The camera controls panel shows every control the V4L2 driver reports, including `auto_exposure` and `exposure_time_absolute`.
- **Only verified on Windows so far.** The Linux build and run have not been tested.

### Windows (MSMF)
- No extra dependencies.
- **Limitation:** through nokhwa's MSMF backend, only the value of exposure/gain can be changed. The Auto/Manual flag is kept as it is, so auto exposure cannot be turned off from the app. Turn it off once in the Windows Camera app or the vendor tool and the setting persists.

## Usage

1. Fix the camera in place so the LCD fills a large part of the frame.
2. **ROI:** drag on an empty area to add a digit box, drag inside a box to move it, drag its bottom-right corner to resize it. Select a box and press Delete to remove it. Boxes are ordered left to right automatically.
3. Check the dots: green = segment on, gray = off, orange = segment disagrees with the matched digit. If the digits are italic, adjust `shear` so the dots sit on the segments.
   Draw each box snug around its digit and centered on it: a box a few px off to one side is right at the edge of what the decoder tolerates, and any tracking error then turns the digit into ERR.
4. Set `decimals` to the number of digits after the fixed decimal point (DYNO-200: 1). `Binarized view` shows the thresholding result.
5. Save the profile JSON.
6. Enter the subject, hand and memo, then **Start session**. Trials are split automatically:
   - A trial starts at ≥2 kg.
   - It ends after 1.5 s below that value or blank.
   - `Peak-hold device`: if the displayed value stays unchanged for 1.5 s, that value is taken as the final one.
   - Manual start/stop is also available. `Delete` removes the selected trial (or the last one if none is selected), for retries.
7. **Box follow:** the area around the boxes (bezel, label, window edges) is stored in the profile as an anchor. Each frame the boxes are moved/scaled to where it is found, so a moved camera or device is followed. Lost → full-frame search every ~0.5 s; boxes hold their last place meanwhile. Redrawing boxes retakes the anchor, so redraw after moving the camera to a new place. The anchor covers only the LCD window and the casing right around it: a wider area pulls in the mount / bracket, which moves relative to the display.
8. Selecting a row in the table shows that trial's curve. Deselect it to go back to the live view.
9. Diagnosis: run with `GRIP_DUMP=<dir>` to save every raw frame that decoded as ERR as JPEG.

### Session folder `sessions/YYYYmmdd-HHMMSS/`
| File | Contents |
|---|---|
| `raw.csv` | every frame: timestamp, value, bits (hex mask per digit, bit0=a…bit6=g), status (ok/blank/err), stable |
| `trials.csv` | wall_start, subject, hand, peak_kg, duration_s, memo |
| `trial_curves.csv` | trial, t, value (for re-plotting) |
| `profile.json` | saved at session start, then overwritten at stop with the final tuned profile |
| `video/` | only when `Record raw video` is on: `NNNNNN.jpg` + `index.csv`, which `--video` can replay |

Only frames that pass the stabilization filter (N identical frames in a row) and are not ERR are fed into trials and peaks. Transition frames are never recorded as values.

## Recognition method
- Sampling: per digit, the 25th percentile of an elongated patch on each of the 7 segments (robust to a few px misalignment), with `shear` correction.
- Background: the two "8" holes of the same digit. Upper segments are scaled against the upper hole and the darkest upper segment, lower against lower, so bezel shade or uneven light on one half or one digit doesn't flip the rest.
- Match: soft nearest pattern (top segment `a` half weight, it's the first to hide under the bezel). No close pattern, or runner-up within `margin` → digit is ERR (half-lit transition frame).
- Blank: digit contrast below half the strongest digit's, or below `min_contrast`.
- A blank leading digit is allowed. A blank or unknown pattern in the middle is ERR.
- Decoding takes about 0.16 ms per frame at 720p (release build).

## Dependencies and why
| crate | why |
|---|---|
| `eframe` / `egui_plot` | immediate-mode GUI + real-time plot, single binary, no OS GUI toolkit needed |
| `nokhwa` (`input-native`) | V4L2/MSMF/AVFoundation camera access and controls. Default `decoding` is turned off because mozjpeg needs nasm |
| `image` (jpeg only) | MJPEG frame decoding + recording encoding, pure Rust |
| `csv` | CSV quoting (commas and quotes in memos) |
| `serde` / `serde_json` | profile JSON |
| `chrono` (clock) | local time for session folder names and the table |

The library part (`decoder`, `session`) builds with only csv/serde (`--no-default-features`).

## Camera setup tips
- **Fix the camera.** Use a tripod or clamp so the ROIs stay put. If the device moves, the ROIs have to be set again.
- **Diffuse lighting.** Reflective LCDs show glare easily. Use indirect or diffused light instead of a bare bulb.
- **Avoid glare.** If the lamp is reflected on the LCD, tilt the camera or the device slightly (10–20°) off the direct angle. Use a polarizing filter if you have one.
- **Turn off auto exposure/gain** and set them manually. Otherwise the brightness changes as a hand gets closer, and the threshold wobbles with it.
- Aim for each digit box to be at least ~40 px tall on screen. Refocus if it looks blurred.
- If ERR persists for more than 3 s, a warning appears in the status bar. Check the ROI, glare and exposure.

## Out of scope
- Phone mp4 input: needs an ffmpeg dependency. For now, record with this app (JPEG folder).
- Recording into a video container: a JPEG sequence is enough for replay. Can be switched to MJPEG-AVI if VLC playback is needed.
- File dialogs: paths are typed as text (to avoid adding a dependency).
