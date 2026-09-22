//! Recording format: a folder of JPEG frames + index.csv (t, file).
//! ponytail: no container/codec — pure Rust, no ffmpeg. MJPEG-AVI if VLC playback is ever needed.

use grip_ocr::synth::{Lcg, SynthStyle, display_text, layout, levels_for, render_levels};
use std::fs::{self, File};
use std::io::{self, BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::sync::mpsc::{SyncSender, sync_channel};
use std::thread::{self, JoinHandle};

pub enum RecFrame {
    /// Camera already delivered JPEG — stored as is, no re-encode.
    Jpeg(Vec<u8>),
    Gray(Vec<u8>, u32, u32),
}

/// Writer thread so JPEG encoding never slows the capture loop.
pub struct Recorder {
    tx: Option<SyncSender<(f64, RecFrame)>>,
    handle: Option<JoinHandle<()>>,
    pub dropped: u64,
}

impl Recorder {
    pub fn start(dir: &Path) -> io::Result<Self> {
        fs::create_dir_all(dir)?;
        let mut index = File::create(dir.join("index.csv"))?;
        writeln!(index, "t,file")?;
        let dir = dir.to_path_buf();
        let (tx, rx) = sync_channel::<(f64, RecFrame)>(64);
        let handle = thread::spawn(move || {
            for (n, (t, f)) in rx.into_iter().enumerate() {
                let name = format!("{n:06}.jpg");
                let bytes = match f {
                    RecFrame::Jpeg(b) => b,
                    RecFrame::Gray(g, w, h) => match encode_gray(&g, w, h) {
                        Some(b) => b,
                        None => continue,
                    },
                };
                if fs::write(dir.join(&name), bytes).is_ok() {
                    let _ = writeln!(index, "{t:.4},{name}");
                }
            }
        });
        Ok(Self { tx: Some(tx), handle: Some(handle), dropped: 0 })
    }

    /// Blocking — offline tools only.
    fn push_wait(&self, t: f64, f: RecFrame) {
        if let Some(tx) = &self.tx {
            let _ = tx.send((t, f));
        }
    }

    pub fn push(&mut self, t: f64, f: RecFrame) {
        if let Some(tx) = &self.tx
            && tx.try_send((t, f)).is_err()
        {
            self.dropped += 1; // disk too slow; never block capture
        }
    }
}

impl Drop for Recorder {
    fn drop(&mut self) {
        self.tx.take();
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

pub fn encode_gray(g: &[u8], w: u32, h: u32) -> Option<Vec<u8>> {
    let mut out = Vec::new();
    image::codecs::jpeg::JpegEncoder::new_with_quality(&mut out, 90).encode(g, w, h, image::ExtendedColorType::L8).ok()?;
    Some(out)
}

/// (t, jpeg path) list of a recording folder.
pub fn read_index(dir: &Path) -> io::Result<Vec<(f64, PathBuf)>> {
    let f = BufReader::new(File::open(dir.join("index.csv"))?);
    let mut out = Vec::new();
    for line in f.lines().skip(1) {
        let line = line?;
        if let Some((t, name)) = line.split_once(',')
            && let Ok(t) = t.parse()
        {
            out.push((t, dir.join(name.trim())));
        }
    }
    Ok(out)
}

pub fn decode_jpeg_gray(bytes: &[u8]) -> Option<(Vec<u8>, u32, u32)> {
    let img = image::load_from_memory_with_format(bytes, image::ImageFormat::Jpeg).ok()?.into_luma8();
    let (w, h) = img.dimensions();
    Some((img.into_raw(), w, h))
}

/// Synthetic recording: 0 → peak → 0 twice, with half-lit transition frames, plus a
/// matching profile.json. Lets the whole app run without a camera.
pub fn make_demo(dir: &Path) -> io::Result<()> {
    let (w, h, fps) = (1280usize, 720usize, 30.0);
    let rois = layout(3, w, h);
    let style = SynthStyle { shear: 0.12, noise: 10.0, gradient: 30.0, ..Default::default() };
    let mut rng = Lcg(9);
    let rec = Recorder::start(dir)?;
    let mut n = 0usize;
    let mut prev: Option<Vec<[f32; 7]>> = None;
    let t0 = 1_700_000_000.0;
    let mut seq: Vec<(f64, usize)> = vec![(0.0, 45)];
    for peak in [28.4, 36.9] {
        seq.extend((1..=12).map(|i| (peak * i as f64 / 12.0, 4)));
        seq.push((peak, 15));
        seq.extend((0..12).rev().map(|i| (peak * i as f64 / 12.0, 3)));
        seq.push((0.0, 75));
    }
    for (v, hold) in seq {
        let lv = levels_for(&display_text(v, 3, 1));
        let mut frames = Vec::new();
        if let Some(p) = &prev {
            let mid: Vec<[f32; 7]> = p.iter().zip(&lv).map(|(a, b)| std::array::from_fn(|i| (a[i] + b[i]) / 2.0)).collect();
            if mid != lv {
                frames.push(mid);
            }
        }
        frames.extend(std::iter::repeat_n(lv.clone(), hold));
        for f in frames {
            let img = render_levels(w, h, &rois, &f, &style, &mut rng);
            rec.push_wait(t0 + n as f64 / fps, RecFrame::Gray(img, w as u32, h as u32));
            n += 1;
        }
        prev = Some(lv);
    }
    let prof = grip_ocr::decoder::Profile { rois, shear: style.shear, ..Default::default() };
    fs::write(dir.join("profile.json"), prof.to_json())?;
    println!("demo: {n} frames → {}", dir.display());
    Ok(())
}
