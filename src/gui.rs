//! egui front end. Owns the Profile (pushes it to the capture thread on change),
//! the TrialDetector and the SessionWriter. Never blocks: frames arrive by channel.

use crate::capture::{Capture, Cmd, FrameMsg, Source, list_cameras};
use eframe::egui::{self, Align2, Color32, FontId, Pos2, Rect, RichText, Sense, Stroke, StrokeKind, TextureHandle, Vec2};
use egui_plot::{Line, Plot, PlotPoints};
use grip_ocr::decoder::{Profile, Reading, Roi, Status, Threshold};
use grip_ocr::session::{Obs, SessionWriter, Trial, TrialConfig, TrialDetector, summarize};
use nokhwa::utils::{CameraControl, ControlValueDescription, ControlValueSetter};
use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Instant;

const PLOT_WINDOW_S: f64 = 60.0;
const ERR_WARN_S: f64 = 3.0;
const HANDLE_PX: f32 = 10.0;

enum Drag {
    New(Pos2, Pos2),
    Move(usize, Vec2),
    Resize(usize),
}

pub struct App {
    ctx: egui::Context,
    cap: Option<Capture>,
    cams: Vec<(u32, String)>,
    cam: (u32, u32, u32, u32), // index, w, h, fps
    video_path: String,
    profile: Profile,
    shared_profile: Arc<Mutex<Profile>>,
    profile_path: String,
    controls: Vec<CameraControl>,

    tex: Option<TextureHandle>,
    tex_seq: u64,
    binarized: bool,
    img_size: [usize; 2],
    reading: Option<Reading>,
    /// Box-follow score of the latest frame.
    track: Option<f32>,
    sel_roi: Option<usize>,
    drag: Option<Drag>,

    det: TrialDetector,
    subject: String,
    hand: String,
    memo: String,
    trials: Vec<Trial>,
    sel_trial: Option<usize>,

    t0: Option<f64>,
    last_t: f64,
    series: VecDeque<[f64; 2]>,
    current: Option<f64>,
    session_peak: f64,
    writer: Option<SessionWriter>,
    record: bool,

    recog: VecDeque<bool>,
    frame_times: VecDeque<Instant>,
    decode_ms: f32,
    err_since: Option<f64>,
    msg: String,
}

impl App {
    pub fn new(cc: &eframe::CreationContext, src: Source, profile: Profile, profile_path: PathBuf) -> Self {
        let (cam, video_path) = match &src {
            Source::Camera { index, width, height, fps } => ((*index, *width, *height, *fps), String::new()),
            Source::Video(p) => ((0, 1280, 720, 30), p.display().to_string()),
        };
        let mut app = Self {
            ctx: cc.egui_ctx.clone(),
            cap: None,
            cams: list_cameras(),
            cam,
            video_path,
            shared_profile: Arc::new(Mutex::new(profile.clone())),
            profile,
            profile_path: profile_path.display().to_string(),
            controls: Vec::new(),
            tex: None,
            tex_seq: 0,
            binarized: false,
            img_size: [0, 0],
            reading: None,
            track: None,
            sel_roi: None,
            drag: None,
            det: TrialDetector::new(TrialConfig::default()),
            subject: String::new(),
            hand: "R".into(),
            memo: String::new(),
            trials: Vec::new(),
            sel_trial: None,
            t0: None,
            last_t: 0.0,
            series: VecDeque::new(),
            current: None,
            session_peak: 0.0,
            writer: None,
            record: false,
            recog: VecDeque::new(),
            frame_times: VecDeque::new(),
            decode_ms: 0.0,
            err_since: None,
            msg: String::new(),
        };
        app.open(src);
        app
    }

    fn open(&mut self, src: Source) {
        self.cap = None; // drop joins the old thread first
        let ctx = self.ctx.clone();
        let cap = Capture::start(src, self.shared_profile.clone(), move || ctx.request_repaint());
        if let Some(w) = &self.writer
            && self.record
        {
            cap.send(Cmd::Record(Some(w.dir.join("video"))));
        }
        self.cap = Some(cap);
        self.t0 = None;
        self.series.clear();
        self.det = TrialDetector::new(self.det.cfg.clone());
    }

    fn on_frame(&mut self, m: FrameMsg) {
        let now = Instant::now();
        self.frame_times.push_back(now);
        while self.frame_times.front().is_some_and(|t| now - *t > std::time::Duration::from_secs(1)) {
            self.frame_times.pop_front();
        }
        self.recog.push_back(m.reading.status != Status::Err);
        if self.recog.len() > 100 {
            self.recog.pop_front();
        }
        self.decode_ms = m.decode_ms;
        self.last_t = m.t;
        let t0 = *self.t0.get_or_insert(m.t);

        if let Some(w) = &mut self.writer
            && let Err(e) = w.log_frame(m.t, &m.reading, m.stable)
        {
            self.msg = format!("raw.csv write failed: {e}");
        }
        if m.reading.status == Status::Err {
            self.err_since.get_or_insert(m.t);
        } else {
            self.err_since = None;
        }
        let obs = match (m.stable, m.reading.status) {
            (true, Status::Ok) => Obs::Value(m.reading.value.unwrap_or(0.0)),
            (true, Status::Blank) => Obs::Blank,
            _ => Obs::Unknown,
        };
        match obs {
            Obs::Value(v) => {
                self.current = Some(v);
                self.session_peak = self.session_peak.max(v);
                self.series.push_back([m.t - t0, v]);
            }
            Obs::Blank => {
                self.current = None;
                self.series.push_back([m.t - t0, 0.0]);
            }
            Obs::Unknown => {}
        }
        while self.series.front().is_some_and(|p| m.t - t0 - p[0] > PLOT_WINDOW_S) {
            self.series.pop_front();
        }
        if let Some(tr) = self.det.push(m.t, obs) {
            self.finish_trial(tr);
        }
    }

    fn finish_trial(&mut self, mut tr: Trial) {
        tr.wall_start = tr.start_t; // frame timestamps are epoch seconds (recordings keep theirs)
        tr.subject = self.subject.clone();
        tr.hand = self.hand.clone();
        tr.memo = self.memo.clone();
        self.trials.push(tr);
        self.save_trials();
    }

    fn save_trials(&mut self) {
        if let Some(w) = &self.writer
            && let Err(e) = w.write_trials(&self.trials)
        {
            self.msg = format!("trials.csv write failed: {e}");
        }
    }

    fn toggle_session(&mut self) {
        if let Some(w) = self.writer.take() {
            self.send(Cmd::Record(None));
            let _ = std::fs::write(w.dir.join("profile.json"), self.profile.to_json()); // final tuned profile
            self.msg = "session stopped".into();
            return;
        }
        let dir = PathBuf::from("sessions").join(chrono::Local::now().format("%Y%m%d-%H%M%S").to_string());
        match SessionWriter::create(&dir, &self.profile) {
            Ok(w) => {
                self.trials.clear();
                self.sel_trial = None;
                self.session_peak = 0.0;
                if self.record {
                    self.send(Cmd::Record(Some(dir.join("video"))));
                }
                self.msg = format!("session → {}", std::fs::canonicalize(&dir).unwrap_or(dir).display());
                self.writer = Some(w);
            }
            Err(e) => self.msg = format!("session create failed: {e}"),
        }
    }

    fn send(&self, c: Cmd) {
        if let Some(cap) = &self.cap {
            cap.send(c);
        }
    }

    // ---------- panels ----------

    fn left(&mut self, ui: &mut egui::Ui) {
        egui::ScrollArea::vertical().show(ui, |ui| {
            ui.heading("Source");
            ui.horizontal(|ui| {
                let name = self.cams.iter().find(|c| c.0 == self.cam.0).map_or(format!("#{}", self.cam.0), |c| format!("{}: {}", c.0, c.1));
                egui::ComboBox::from_id_salt("cam").selected_text(name).width(170.0).show_ui(ui, |ui| {
                    for (i, n) in &self.cams {
                        ui.selectable_value(&mut self.cam.0, *i, format!("{i}: {n}"));
                    }
                });
                if ui.button("⟳").on_hover_text("rescan").clicked() {
                    self.cams = list_cameras();
                }
            });
            ui.horizontal(|ui| {
                ui.add(egui::DragValue::new(&mut self.cam.1).range(160..=3840));
                ui.label("×");
                ui.add(egui::DragValue::new(&mut self.cam.2).range(120..=2160));
                ui.add(egui::DragValue::new(&mut self.cam.3).range(1..=120).suffix(" fps"));
            });
            if ui.button("Open camera").clicked() {
                let (index, width, height, fps) = self.cam;
                self.open(Source::Camera { index, width, height, fps });
            }
            ui.horizontal(|ui| {
                ui.add(egui::TextEdit::singleline(&mut self.video_path).hint_text("recording folder").desired_width(170.0));
                if ui.button("Replay").clicked() && !self.video_path.is_empty() {
                    self.open(Source::Video(PathBuf::from(&self.video_path)));
                }
            });

            if !self.controls.is_empty() {
                ui.separator();
                ui.heading("Camera controls");
                self.camera_controls(ui);
            }

            ui.separator();
            ui.heading("Recognition");
            self.profile_ui(ui);

            ui.separator();
            ui.heading("Trial");
            ui.horizontal(|ui| {
                ui.label("Subject");
                ui.text_edit_singleline(&mut self.subject);
            });
            ui.horizontal(|ui| {
                ui.label("Hand");
                ui.radio_value(&mut self.hand, "L".to_string(), "L");
                ui.radio_value(&mut self.hand, "R".to_string(), "R");
            });
            ui.horizontal(|ui| {
                ui.label("Memo");
                ui.text_edit_singleline(&mut self.memo);
            });
            let cfg = &mut self.det.cfg;
            ui.add(egui::DragValue::new(&mut cfg.start_kg).range(0.1..=50.0).speed(0.1).prefix("start ≥ ").suffix(" kg"));
            ui.add(egui::DragValue::new(&mut cfg.end_hold_s).range(0.2..=10.0).speed(0.05).prefix("end hold ").suffix(" s"));
            ui.checkbox(&mut cfg.peak_hold_mode, "Peak-hold device (held value = final)");
            ui.horizontal(|ui| {
                if self.det.is_active() {
                    if ui.button("■ Stop trial").clicked()
                        && let Some(tr) = self.det.force_stop(self.last_t)
                    {
                        self.finish_trial(tr);
                    }
                } else if ui.button("▶ Start trial").clicked() {
                    self.det.force_start(self.last_t);
                }
                let label = if self.sel_trial.is_some() { "Delete selected" } else { "Delete last (retry)" };
                if ui.add_enabled(!self.trials.is_empty(), egui::Button::new(label)).clicked() {
                    let i = self.sel_trial.take().unwrap_or(self.trials.len() - 1);
                    self.trials.remove(i);
                    self.save_trials();
                }
            });

            ui.separator();
            ui.heading("Session");
            ui.add_enabled(self.writer.is_none(), egui::Checkbox::new(&mut self.record, "Record raw video"));
            let label = if self.writer.is_some() { "■ Stop session" } else { "● Start session" };
            if ui.button(label).clicked() {
                self.toggle_session();
            }
            if let Some(w) = &self.writer {
                ui.label(format!("→ {}", w.dir.display()));
            }
        });
    }

    fn camera_controls(&mut self, ui: &mut egui::Ui) {
        let mut sets = Vec::new();
        for c in &self.controls {
            let id = c.control();
            match c.description() {
                ControlValueDescription::IntegerRange { min, max, value, step, .. } => {
                    let mut v = *value;
                    let r = ui.add(egui::Slider::new(&mut v, *min..=*max).step_by((*step).max(1) as f64).text(c.name()));
                    if r.changed() {
                        sets.push((id, ControlValueSetter::Integer(v)));
                    }
                }
                ControlValueDescription::Integer { value, step, .. } => {
                    let mut v = *value;
                    ui.horizontal(|ui| {
                        if ui.add(egui::DragValue::new(&mut v).speed((*step).max(1) as f64)).changed() {
                            sets.push((id, ControlValueSetter::Integer(v)));
                        }
                        ui.label(c.name());
                    });
                }
                ControlValueDescription::Boolean { value, .. } => {
                    let mut v = *value;
                    if ui.checkbox(&mut v, c.name()).changed() {
                        sets.push((id, ControlValueSetter::Boolean(v)));
                    }
                }
                ControlValueDescription::Enum { value, possible, .. } => {
                    let mut v = *value;
                    egui::ComboBox::from_label(c.name()).selected_text(v.to_string()).show_ui(ui, |ui| {
                        for p in possible {
                            ui.selectable_value(&mut v, *p, p.to_string());
                        }
                    });
                    if v != *value {
                        sets.push((id, ControlValueSetter::EnumValue(v)));
                    }
                }
                _ => {} // ponytail: float/string/rgb controls don't matter for exposure/gain
            }
        }
        for (id, v) in sets {
            self.send(Cmd::Control(id, v));
        }
    }

    fn profile_ui(&mut self, ui: &mut egui::Ui) {
        let p = &mut self.profile;
        ui.label(format!("{} digit boxes — drag on image to add, drag inside to move, corner to resize", p.rois.len()));
        ui.horizontal(|ui| {
            if ui.add_enabled(self.sel_roi.is_some(), egui::Button::new("Delete box")).clicked()
                && let Some(i) = self.sel_roi.take()
            {
                if let Some(t) = self.reading.as_ref().map(|r| &r.rois).filter(|t| t.len() == p.rois.len()) {
                    p.rois = t.clone();
                }
                p.rois.remove(i);
                self.drag = None;
            }
            if ui.button("Clear").clicked() {
                p.rois.clear();
                self.sel_roi = None;
                self.drag = None;
            }
        });
        ui.horizontal(|ui| {
            ui.label("rotate");
            for d in [0, 90, 180, 270] {
                ui.radio_value(&mut p.rotate, d, format!("{d}°"));
            }
        });
        ui.add(egui::Slider::new(&mut p.decimals, 0..=3).text("decimals"));
        ui.add(egui::Slider::new(&mut p.shear, -0.4..=0.4).text("shear (italic)"));
        ui.horizontal(|ui| {
            ui.radio_value(&mut p.threshold, Threshold::Auto, "Auto");
            ui.radio_value(&mut p.threshold, Threshold::Manual, "Manual");
        });
        if p.threshold == Threshold::Manual {
            ui.add(egui::Slider::new(&mut p.manual_thresh, 0..=255).text("threshold"));
        }
        ui.checkbox(&mut p.lit_is_dark, "Segments darker than background");
        ui.add(egui::Slider::new(&mut p.stable_frames, 1..=15).text("stable frames"));
        ui.add(egui::Slider::new(&mut p.min_contrast, 0..=150).text("min contrast"));
        ui.add(egui::Slider::new(&mut p.margin, 0.0..=0.15).text("ambiguity margin"));
        ui.checkbox(&mut self.binarized, "Binarized view");
        ui.horizontal(|ui| {
            ui.add(egui::TextEdit::singleline(&mut self.profile_path).desired_width(140.0));
            if ui.button("Save").clicked() {
                self.msg = match std::fs::write(&self.profile_path, self.profile.to_json()) {
                    Ok(()) => format!("profile saved: {}", self.profile_path),
                    Err(e) => format!("profile save failed: {e}"),
                };
            }
            if ui.button("Load").clicked() {
                match std::fs::read_to_string(&self.profile_path)
                    .map_err(|e| e.to_string())
                    .and_then(|s| Profile::from_json(&s).map_err(|e| e.to_string()))
                {
                    Ok(p) => {
                        self.profile = p;
                        self.sel_roi = None;
                        self.msg = format!("profile loaded: {}", self.profile_path);
                    }
                    Err(e) => self.msg = format!("profile load failed: {e}"),
                }
            }
        });
    }

    fn right(&mut self, ui: &mut egui::Ui) {
        let (txt, col) = match (&self.reading, self.current) {
            (Some(r), _) if r.status == Status::Err => ("ERR".to_string(), Color32::RED),
            (_, Some(v)) => (format!("{v:.*}", self.profile.decimals as usize), ui.visuals().strong_text_color()),
            _ => ("—".to_string(), Color32::GRAY),
        };
        ui.label(RichText::new(txt).size(64.0).color(col).monospace());
        ui.label(RichText::new(format!("session peak {:.*} kg", self.profile.decimals as usize, self.session_peak)).size(22.0));
        if let Some(p) = self.det.current_peak() {
            ui.label(RichText::new(format!("● trial running, peak {p:.1}")).color(Color32::from_rgb(230, 140, 0)));
        }
        ui.separator();
        ui.heading("Trials");
        egui::ScrollArea::vertical().max_height(ui.available_height() * 0.6).show(ui, |ui| {
            egui::Grid::new("trials").striped(true).num_columns(6).show(ui, |ui| {
                for h in ["time", "subject", "hand", "peak", "dur", "memo"] {
                    ui.strong(h);
                }
                ui.end_row();
                for (i, t) in self.trials.iter().enumerate() {
                    let sel = self.sel_trial == Some(i);
                    if ui.selectable_label(sel, hms(t.wall_start)).clicked() {
                        self.sel_trial = if sel { None } else { Some(i) };
                    }
                    ui.label(&t.subject);
                    ui.label(&t.hand);
                    ui.label(format!("{:.1}", t.peak));
                    ui.label(format!("{:.1}s", t.duration()));
                    ui.label(&t.memo);
                    ui.end_row();
                }
            });
        });
        ui.separator();
        ui.heading("Summary");
        egui::Grid::new("summary").striped(true).show(ui, |ui| {
            for h in ["subject", "hand", "n", "mean", "max"] {
                ui.strong(h);
            }
            ui.end_row();
            for s in summarize(&self.trials) {
                ui.label(s.subject);
                ui.label(s.hand);
                ui.label(s.n.to_string());
                ui.label(format!("{:.1}", s.mean_peak));
                ui.label(format!("{:.1}", s.max_peak));
                ui.end_row();
            }
        });
    }

    fn status_bar(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            let rate = if self.recog.is_empty() {
                0.0
            } else {
                self.recog.iter().filter(|&&b| b).count() as f64 / self.recog.len() as f64 * 100.0
            };
            ui.label(format!("recog {rate:.0}%  |  {} fps  |  decode {:.2} ms", self.frame_times.len(), self.decode_ms));
            match self.track {
                Some(t) if t < grip_ocr::track::MIN_SCORE => {
                    ui.label(
                        RichText::new(format!("⚠ box follow lost ({t:.2}) — boxes held, redraw if the display moved a lot"))
                            .color(Color32::YELLOW),
                    );
                }
                Some(t) => {
                    ui.label(format!("follow {t:.2}"));
                }
                None => {}
            }
            if let Some(since) = self.err_since
                && self.last_t - since > ERR_WARN_S
            {
                ui.label(
                    RichText::new(format!("⚠ ERR for {:.0}s — check ROI / glare / exposure", self.last_t - since))
                        .color(Color32::RED)
                        .strong(),
                );
            }
            if let Some(cap) = &self.cap {
                let s = cap.shared.lock().unwrap();
                ui.separator();
                ui.label(&s.status);
                if s.rec_dropped > 0 {
                    ui.label(RichText::new(format!("rec dropped {}", s.rec_dropped)).color(Color32::YELLOW));
                }
            }
            if !self.msg.is_empty() {
                ui.separator();
                ui.label(&self.msg);
            }
        });
    }

    fn central(&mut self, ui: &mut egui::Ui) {
        let plot_h = 200.0;
        let avail = ui.available_size() - Vec2::new(0.0, plot_h + 8.0);
        let [w, h] = self.img_size;
        if let (Some(tex), true) = (&self.tex, w > 0) {
            let s = (avail.x / w as f32).min(avail.y / h as f32).max(0.01);
            let (resp, painter) = ui.allocate_painter(Vec2::new(w as f32 * s, h as f32 * s), Sense::click_and_drag());
            let r = resp.rect;
            painter.image(tex.id(), r, Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)), Color32::WHITE);
            let to_screen = |x: f32, y: f32| r.min + Vec2::new(x, y) * s;
            let to_img = |p: Pos2| ((p - r.min) / s).to_pos2();
            self.edit_rois(&resp, to_img, s);
            self.overlay(&painter, to_screen);
        } else {
            ui.allocate_space(avail);
        }
        self.plot(ui, plot_h);
    }

    fn edit_rois(&mut self, resp: &egui::Response, to_img: impl Fn(Pos2) -> Pos2, s: f32) {
        // edits start from where the boxes are drawn (followed), not where they were first set
        if (resp.drag_started() || resp.clicked() || resp.ctx.input(|i| i.key_pressed(egui::Key::Delete)))
            && let Some(t) = self.reading.as_ref().map(|r| &r.rois).filter(|t| t.len() == self.profile.rois.len())
        {
            self.profile.rois = t.clone();
        }
        let rois = &mut self.profile.rois;
        let hit = |p: Pos2, rois: &[Roi]| {
            rois.iter().position(|r| p.x >= r.x as f32 && p.x <= (r.x + r.w) as f32 && p.y >= r.y as f32 && p.y <= (r.y + r.h) as f32)
        };
        if let Some(p) = resp.interact_pointer_pos().map(&to_img) {
            if resp.drag_started() {
                let corner = rois.iter().position(|r| (Pos2::new((r.x + r.w) as f32, (r.y + r.h) as f32) - p).length() * s < HANDLE_PX);
                self.drag = Some(match (corner, hit(p, rois)) {
                    (Some(i), _) => Drag::Resize(i),
                    (None, Some(i)) => Drag::Move(i, p - Pos2::new(rois[i].x as f32, rois[i].y as f32)),
                    (None, None) => Drag::New(p, p),
                });
                if let Some(Drag::Resize(i) | Drag::Move(i, _)) = self.drag {
                    self.sel_roi = Some(i);
                }
            }
            match &mut self.drag {
                Some(Drag::New(_, cur)) => *cur = p,
                Some(Drag::Move(i, off)) => {
                    if let Some(r) = rois.get_mut(*i) {
                        r.x = (p.x - off.x).round() as i32;
                        r.y = (p.y - off.y).round() as i32;
                    }
                }
                Some(Drag::Resize(i)) => {
                    if let Some(r) = rois.get_mut(*i) {
                        r.w = ((p.x - r.x as f32).round() as i32).max(5);
                        r.h = ((p.y - r.y as f32).round() as i32).max(5);
                    }
                }
                None => {}
            }
            if resp.clicked() {
                self.sel_roi = hit(p, rois);
            }
        }
        if resp.drag_stopped() {
            let moved = match self.drag.take() {
                Some(Drag::New(a, b)) => {
                    let r = Rect::from_two_pos(a, b);
                    (r.width() >= 5.0 && r.height() >= 5.0).then(|| {
                        rois.push(Roi {
                            x: r.min.x.round() as i32,
                            y: r.min.y.round() as i32,
                            w: r.width().round() as i32,
                            h: r.height().round() as i32,
                        });
                        rois.len() - 1
                    })
                }
                Some(Drag::Move(i, _) | Drag::Resize(i)) => Some(i),
                None => None,
            };
            // keep left → right order (digit order); follow the edited box
            let key = moved.and_then(|i| rois.get(i).copied());
            rois.sort_by_key(|r| r.x);
            self.sel_roi = key.and_then(|k| rois.iter().position(|r| *r == k));
        }
        if self.sel_roi.is_some() && resp.ctx.memory(|m| m.focused().is_none()) && resp.ctx.input(|i| i.key_pressed(egui::Key::Delete)) {
            rois.remove(self.sel_roi.take().unwrap());
            self.drag = None;
        }
    }

    /// Boxes as last decoded (followed), or the profile's while editing.
    fn boxes(&self) -> Vec<Roi> {
        match &self.reading {
            Some(r) if self.drag.is_none() && r.rois.len() == self.profile.rois.len() => r.rois.clone(),
            _ => self.profile.rois.clone(),
        }
    }

    fn overlay(&self, painter: &egui::Painter, to_screen: impl Fn(f32, f32) -> Pos2) {
        for (i, r) in self.boxes().iter().enumerate() {
            let rect = Rect::from_min_max(to_screen(r.x as f32, r.y as f32), to_screen((r.x + r.w) as f32, (r.y + r.h) as f32));
            let col = if self.sel_roi == Some(i) { Color32::YELLOW } else { Color32::from_rgb(60, 160, 255) };
            painter.rect_stroke(rect, 0.0, Stroke::new(2.0, col), StrokeKind::Outside);
            painter.rect_filled(Rect::from_center_size(rect.max, Vec2::splat(8.0)), 0.0, col);
            painter.text(rect.left_top(), Align2::LEFT_BOTTOM, i.to_string(), FontId::proportional(14.0), col);
        }
        if let Some(Drag::New(a, b)) = &self.drag {
            painter.rect_stroke(
                Rect::from_two_pos(to_screen(a.x, a.y), to_screen(b.x, b.y)),
                0.0,
                Stroke::new(1.5, Color32::YELLOW),
                StrokeKind::Outside,
            );
        }
        let Some(rd) = &self.reading else { return };
        for smp in &rd.samples {
            let col = if smp.ambiguous {
                Color32::from_rgb(255, 150, 0)
            } else if smp.on {
                Color32::from_rgb(0, 220, 0)
            } else {
                Color32::from_gray(140)
            };
            painter.circle_filled(to_screen(smp.x, smp.y), 4.0, col);
        }
        let boxes = self.boxes();
        if let Some(r0) = boxes.first() {
            let (txt, col) = match (rd.status, rd.value) {
                (Status::Ok, Some(v)) => (format!("{v:.*}", self.profile.decimals as usize), Color32::from_rgb(0, 220, 0)),
                (Status::Blank, _) => ("blank".into(), Color32::GRAY),
                _ => ("ERR".into(), Color32::RED),
            };
            let min_y = boxes.iter().map(|r| r.y).min().unwrap_or(r0.y);
            painter.text(
                to_screen(r0.x as f32, min_y as f32) - Vec2::new(0.0, 18.0),
                Align2::LEFT_BOTTOM,
                txt,
                FontId::proportional(28.0),
                col,
            );
        }
    }

    fn plot(&self, ui: &mut egui::Ui, h: f32) {
        let (name, pts): (&str, Vec<[f64; 2]>) = match self.sel_trial.and_then(|i| self.trials.get(i)) {
            Some(t) => ("selected trial", t.curve.iter().map(|&(ts, v)| [ts - t.start_t, v]).collect()),
            None => ("live", self.series.iter().copied().collect()),
        };
        Plot::new("kg").height(h).include_y(0.0).x_axis_label("s").y_axis_label("kg").legend(egui_plot::Legend::default()).show(ui, |p| {
            p.line(Line::new(name, PlotPoints::from(pts)));
        });
    }

    fn update_texture(&mut self) {
        let Some(cap) = &self.cap else { return };
        let mut s = cap.shared.lock().unwrap();
        if let Some(a) = s.new_anchor.take() {
            self.profile.anchor = Some(a);
        }
        let Some(l) = &s.latest else { return };
        if l.seq == self.tex_seq {
            return;
        }
        self.tex_seq = l.seq;
        self.img_size = [l.width, l.height];
        self.reading = Some(l.reading.clone());
        self.track = l.track;

        let img = if self.binarized {
            let t = l.reading.thresh;
            let dark = self.profile.lit_is_dark;
            // lit segments drawn black
            let bin: Vec<u8> = l.gray.iter().map(|&v| if (v <= t) == dark { 0 } else { 255 }).collect();
            egui::ColorImage::from_gray([l.width, l.height], &bin)
        } else {
            egui::ColorImage::from_gray([l.width, l.height], &l.gray)
        };
        if s.controls.len() != self.controls.len() || !self.ctx.input(|i| i.pointer.any_down()) {
            self.controls = s.controls.clone(); // not while dragging a slider, or it snaps back
        }
        drop(s);
        match &mut self.tex {
            Some(t) => t.set(img, egui::TextureOptions::LINEAR),
            None => self.tex = Some(self.ctx.load_texture("frame", img, egui::TextureOptions::LINEAR)),
        }
    }
}

fn hms(t: f64) -> String {
    chrono::DateTime::from_timestamp(t as i64, 0)
        .map(|d| d.with_timezone(&chrono::Local).format("%H:%M:%S").to_string())
        .unwrap_or_default()
}

impl eframe::App for App {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let msgs: Vec<FrameMsg> = self.cap.as_ref().map(|c| c.rx.try_iter().collect()).unwrap_or_default();
        for m in msgs {
            self.on_frame(m);
        }
        self.update_texture();

        egui::Panel::bottom("status").show(ui, |ui| self.status_bar(ui));
        egui::Panel::left("left").resizable(true).default_size(300.0).show(ui, |ui| self.left(ui));
        egui::Panel::right("right").resizable(true).default_size(340.0).show(ui, |ui| self.right(ui));
        egui::CentralPanel::default().show(ui, |ui| self.central(ui));

        let mut sp = self.shared_profile.lock().unwrap();
        if *sp != self.profile {
            if sp.rois != self.profile.rois
                && let Some(r) = &mut self.reading
            {
                r.rois = self.profile.rois.clone(); // edited: draw the new boxes until the next frame
            }
            *sp = self.profile.clone();
        }
    }
}
