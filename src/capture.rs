//! Capture thread: camera (nokhwa) or recorded folder → gray → decode → stabilize.
//! Readings go through an unbounded channel (never dropped, they're the data);
//! the display frame is a latest-only slot (dropping is fine).

use crate::video::{RecFrame, Recorder, decode_jpeg_gray, read_index};
use grip_ocr::decoder::{Gray, Profile, Reading, Stabilizer, decode, rotate};
use grip_ocr::track::{Anchor, Tracker};
use nokhwa::Camera;
use nokhwa::pixel_format::LumaFormat;
use nokhwa::utils::{
    ApiBackend, CameraControl, CameraFormat, CameraIndex, ControlValueSetter, FrameFormat, KnownCameraControl, RequestedFormat,
    RequestedFormatType, Resolution,
};
use std::path::PathBuf;
use std::sync::mpsc::{Receiver, Sender, TryRecvError, channel};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

#[derive(Clone, Debug)]
pub enum Source {
    Camera { index: u32, width: u32, height: u32, fps: u32 },
    Video(PathBuf),
}

pub struct FrameMsg {
    pub t: f64,
    pub reading: Reading,
    pub stable: bool,
    pub decode_ms: f32,
}

pub struct Latest {
    pub gray: Vec<u8>,
    pub width: usize,
    pub height: usize,
    pub reading: Reading,
    pub seq: u64,
    /// Box-follow match score, None when not tracking.
    pub track: Option<f32>,
}

pub enum Cmd {
    Control(KnownCameraControl, ControlValueSetter),
    Record(Option<PathBuf>),
    Stop,
}

#[derive(Default)]
pub struct Shared {
    pub latest: Option<Latest>,
    pub controls: Vec<CameraControl>,
    pub status: String,
    pub finished: bool,
    pub rec_dropped: u64,
    /// Fresh anchor for the GUI to put into the profile (so it gets saved).
    pub new_anchor: Option<Anchor>,
}

pub struct Capture {
    pub rx: Receiver<FrameMsg>,
    pub shared: Arc<Mutex<Shared>>,
    cmd: Sender<Cmd>,
    handle: Option<JoinHandle<()>>,
}

impl Capture {
    pub fn start(src: Source, profile: Arc<Mutex<Profile>>, repaint: impl Fn() + Send + 'static) -> Self {
        let (tx, rx) = channel();
        let (cmd, cmd_rx) = channel();
        let shared = Arc::new(Mutex::new(Shared::default()));
        let mut w = Worker {
            tx,
            cmd_rx,
            shared: shared.clone(),
            profile,
            stab: Stabilizer::default(),
            rec: None,
            trk: None,
            seq: 0,
            repaint: Box::new(repaint),
            stop: false,
        };
        let handle = thread::spawn(move || match src {
            Source::Video(dir) => w.run_video(dir),
            Source::Camera { index, width, height, fps } => w.run_camera(index, width, height, fps),
        });
        Self { rx, shared, cmd, handle: Some(handle) }
    }

    pub fn send(&self, c: Cmd) {
        let _ = self.cmd.send(c);
    }
}

impl Drop for Capture {
    fn drop(&mut self) {
        self.send(Cmd::Stop);
        // Bounded wait: frame() can hang on a stalled USB camera; never freeze the GUI for it.
        // ponytail: after 1 s the thread is detached and exits once frame() returns.
        if let Some(h) = self.handle.take() {
            let end = Instant::now() + Duration::from_secs(1);
            while !h.is_finished() && Instant::now() < end {
                thread::sleep(Duration::from_millis(10));
            }
            if h.is_finished() {
                let _ = h.join();
            }
        }
    }
}

pub fn list_cameras() -> Vec<(u32, String)> {
    nokhwa::query(ApiBackend::Auto)
        .unwrap_or_default()
        .into_iter()
        .filter_map(|c| match c.index() {
            CameraIndex::Index(i) => Some((*i, c.human_name())),
            _ => None,
        })
        .collect()
}

fn now() -> f64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs_f64()).unwrap_or(0.0)
}

struct Worker {
    tx: Sender<FrameMsg>,
    cmd_rx: Receiver<Cmd>,
    shared: Arc<Mutex<Shared>>,
    profile: Arc<Mutex<Profile>>,
    stab: Stabilizer,
    rec: Option<Recorder>,
    trk: Option<Tracker>,
    seq: u64,
    repaint: Box<dyn Fn() + Send>,
    stop: bool,
}

impl Worker {
    fn status(&self, s: impl Into<String>) {
        self.shared.lock().unwrap().status = s.into();
        (self.repaint)();
    }

    /// Drain commands; camera-only ones are handed back.
    fn poll_cmds(&mut self, cam: Option<&mut Camera>) {
        let mut cam = cam;
        loop {
            match self.cmd_rx.try_recv() {
                Ok(Cmd::Stop) | Err(TryRecvError::Disconnected) => {
                    self.stop = true;
                    return;
                }
                Ok(Cmd::Record(dir)) => {
                    self.rec = None; // drop flushes the old one
                    if let Some(d) = dir {
                        match Recorder::start(&d) {
                            Ok(r) => self.rec = Some(r),
                            Err(e) => self.status(format!("record failed: {e}")),
                        }
                    }
                }
                Ok(Cmd::Control(id, v)) => {
                    if let Some(c) = cam.as_deref_mut() {
                        if let Err(e) = c.set_camera_control(id, v) {
                            self.status(format!("set {id} failed: {e}"));
                        }
                        self.shared.lock().unwrap().controls = c.camera_controls().unwrap_or_default();
                    }
                }
                Err(TryRecvError::Empty) => return,
            }
        }
    }

    fn process(&mut self, t: f64, gray: Vec<u8>, width: usize, height: usize, jpeg: Option<&[u8]>) {
        let p = self.profile.lock().unwrap().clone();
        // record the raw frame; rotation is re-applied from profile.json on replay
        if let Some(r) = &mut self.rec {
            r.push(
                t,
                match jpeg {
                    Some(j) => RecFrame::Jpeg(j.to_vec()),
                    None => RecFrame::Gray(gray.clone(), width as u32, height as u32),
                },
            );
        }
        let (gray, width, height) =
            if p.rotate == 0 { (gray, width, height) } else { rotate(&Gray { data: &gray, width, height }, p.rotate) };
        let img = Gray { data: &gray, width, height };
        let t0 = Instant::now();
        let mut p = p;
        let track = self.follow(&img, &mut p);
        let reading = decode(&img, &p);
        let decode_ms = t0.elapsed().as_secs_f32() * 1000.0;
        // GRIP_DUMP=<dir>: keep raw frames that decoded as ERR, for offline diagnosis
        if reading.status == grip_ocr::decoder::Status::Err
            && let (Some(dir), Some(j)) = (std::env::var_os("GRIP_DUMP"), jpeg)
        {
            let _ = std::fs::create_dir_all(&dir);
            let _ = std::fs::write(PathBuf::from(dir).join(format!("{:06}.jpg", self.seq)), j);
        }
        let stable = self.stab.push(&reading, p.stable_frames);
        let _ = self.tx.send(FrameMsg { t, reading: reading.clone(), stable, decode_ms });
        self.seq += 1;
        let mut s = self.shared.lock().unwrap();
        s.rec_dropped = self.rec.as_ref().map_or(0, |r| r.dropped);
        s.latest = Some(Latest { gray, width, height, reading, seq: self.seq, track });
        drop(s);
        (self.repaint)();
    }

    /// Move `p.rois` to where the display is now. Anchor missing / stale (boxes or rotation
    /// edited) → take a new one from this frame, where the boxes were just drawn.
    fn follow(&mut self, img: &Gray, p: &mut Profile) -> Option<f32> {
        if p.rois.is_empty() {
            self.trk = None;
            return None;
        }
        let fresh = |a: &Anchor| a.rois == p.rois && a.rotate == p.rotate;
        if !self.trk.as_ref().is_some_and(|t| fresh(&t.anchor)) {
            let a = match p.anchor.clone().filter(fresh) {
                Some(a) => a,
                None => {
                    let a = Anchor::capture(img, &p.rois, p.rotate)?;
                    self.shared.lock().unwrap().new_anchor = Some(a.clone());
                    a
                }
            };
            self.trk = Some(Tracker::new(a));
        }
        let t = self.trk.as_mut()?;
        // ponytail: search every 3rd frame (~14 ms release), full-frame at most every 15th (~0.5 s)
        p.rois = if self.seq.is_multiple_of(3) { t.update(img, self.seq.is_multiple_of(15)) } else { t.rois() };
        Some(t.score)
    }

    fn run_video(&mut self, dir: PathBuf) {
        let frames = match read_index(&dir) {
            Ok(f) if !f.is_empty() => f,
            Ok(_) => return self.finish("no frames in video folder"),
            Err(e) => return self.finish(format!("cannot open video {}: {e}", dir.display())),
        };
        self.status(format!("playing {} ({} frames)", dir.display(), frames.len()));
        let (wall0, t_first) = (Instant::now(), frames[0].0);
        for (t, path) in frames {
            self.poll_cmds(None);
            if self.stop {
                return;
            }
            // pace to recorded timestamps
            let due = Duration::from_secs_f64((t - t_first).max(0.0));
            if let Some(wait) = due.checked_sub(wall0.elapsed()) {
                thread::sleep(wait);
            }
            let Some((g, w, h)) = std::fs::read(&path).ok().and_then(|b| decode_jpeg_gray(&b)) else {
                continue;
            };
            self.process(t, g, w as usize, h as usize, None);
        }
        self.finish("playback finished");
    }

    fn finish(&mut self, msg: impl Into<String>) {
        self.rec = None;
        self.shared.lock().unwrap().finished = true;
        self.status(msg);
    }

    fn open(index: u32, width: u32, height: u32, fps: u32) -> Result<Camera, nokhwa::NokhwaError> {
        let want = CameraFormat::new(Resolution::new(width, height), FrameFormat::MJPEG, fps);
        let mut cam = Camera::new(CameraIndex::Index(index), RequestedFormat::new::<LumaFormat>(RequestedFormatType::Closest(want)))?;
        cam.open_stream()?;
        Ok(cam)
    }

    fn run_camera(&mut self, index: u32, width: u32, height: u32, fps: u32) {
        while !self.stop {
            let mut cam = match Self::open(index, width, height, fps) {
                Ok(c) => c,
                Err(e) => {
                    self.status(format!("camera {index} open failed, retrying: {e}"));
                    self.sleep_polling(Duration::from_secs(1));
                    continue;
                }
            };
            let f = cam.camera_format();
            self.shared.lock().unwrap().controls = cam.camera_controls().unwrap_or_default();
            self.status(format!("camera {index}: {}x{} {} {}fps", f.width(), f.height(), f.format(), f.frame_rate()));
            while !self.stop {
                self.poll_cmds(Some(&mut cam));
                match cam.frame() {
                    Ok(buf) => {
                        let t = now();
                        let res = buf.resolution();
                        let (w, h) = (res.width() as usize, res.height() as usize);
                        let raw = buf.buffer();
                        match buf.source_frame_format() {
                            FrameFormat::MJPEG => {
                                if let Some((g, w, h)) = decode_jpeg_gray(raw) {
                                    self.process(t, g, w as usize, h as usize, Some(raw));
                                }
                            }
                            fmt => {
                                if let Some(g) = to_gray(fmt, raw, w, h) {
                                    self.process(t, g, w, h, None);
                                }
                            }
                        }
                    }
                    Err(e) => {
                        self.status(format!("frame failed, reconnecting: {e}"));
                        break; // drop camera, reopen
                    }
                }
            }
            let _ = cam.stop_stream();
        }
    }

    fn sleep_polling(&mut self, d: Duration) {
        let end = Instant::now() + d;
        while Instant::now() < end && !self.stop {
            self.poll_cmds(None);
            thread::sleep(Duration::from_millis(50));
        }
    }
}

/// Uncompressed formats → luma. Y is all the decoder needs.
fn to_gray(fmt: FrameFormat, raw: &[u8], w: usize, h: usize) -> Option<Vec<u8>> {
    let n = w * h;
    match fmt {
        FrameFormat::GRAY => raw.get(..n).map(<[u8]>::to_vec),
        FrameFormat::NV12 => raw.get(..n).map(<[u8]>::to_vec),
        FrameFormat::YUYV => (raw.len() >= n * 2).then(|| raw.iter().step_by(2).take(n).copied().collect()),
        FrameFormat::RAWRGB | FrameFormat::RAWBGR => (raw.len() >= n * 3).then(|| {
            // channel order barely matters for luma of a gray LCD; plain mean
            raw.chunks_exact(3).take(n).map(|p| ((p[0] as u16 + p[1] as u16 + p[2] as u16) / 3) as u8).collect()
        }),
        FrameFormat::MJPEG => None,
    }
}
