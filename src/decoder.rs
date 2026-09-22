//! 7-segment LCD decoder. Pure functions, no GUI deps.
//!
//! Per digit ROI: 7 segment patches vs the digit's own background → nearest
//! pattern. A digit that fits no pattern, or two patterns about equally
//! (half-lit transition frame), is ERR, so it never gets logged as a value.

use serde::{Deserialize, Serialize};

/// Borrowed 8-bit grayscale frame, row-major, stride == width.
#[derive(Clone, Copy)]
pub struct Gray<'a> {
    pub data: &'a [u8],
    pub width: usize,
    pub height: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Roi {
    pub x: i32,
    pub y: i32,
    pub w: i32,
    pub h: i32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Threshold {
    /// Per-digit, from the digit's own background (holes) and darkest segment.
    #[serde(alias = "otsu")]
    Auto,
    Manual,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Profile {
    /// Digit boxes, left → right.
    pub rois: Vec<Roi>,
    /// Fixed decimal places (the LCD decimal point never moves).
    pub decimals: u32,
    /// Italic correction: sample x += shear * (0.5 - ny) * h. Positive = top leans right.
    pub shear: f32,
    pub threshold: Threshold,
    pub manual_thresh: u8,
    /// Typical reflective LCD: segments darker than background.
    pub lit_is_dark: bool,
    /// Same reading N frames in a row before it counts.
    pub stable_frames: u32,
    /// Digit contrast (background − darkest segment) below this → blank (display off).
    pub min_contrast: u8,
    /// Best digit must beat the runner-up pattern by this much (0..1 cost) or the digit is ERR.
    pub margin: f32,
    /// Clockwise frame rotation before everything else: 0, 90, 180, 270.
    pub rotate: u16,
    /// Snapshot around the boxes for auto-follow (taken automatically when boxes change).
    pub anchor: Option<crate::track::Anchor>,
}

impl Default for Profile {
    fn default() -> Self {
        Self {
            rois: Vec::new(),
            decimals: 1,
            shear: 0.0,
            threshold: Threshold::Auto,
            manual_thresh: 128,
            lit_is_dark: true,
            stable_frames: 3,
            min_contrast: 30,
            margin: 0.03,
            rotate: 0,
            anchor: None,
        }
    }
}

/// Rotate a gray frame clockwise by 90/180/270 degrees (anything else = copy). Returns (data, w, h).
pub fn rotate(g: &Gray, deg: u16) -> (Vec<u8>, usize, usize) {
    let (w, h) = (g.width, g.height);
    let at = |x: usize, y: usize| g.data[y * w + x];
    match deg {
        90 => ((0..w).flat_map(|y| (0..h).map(move |x| (x, y))).map(|(x, y)| at(y, h - 1 - x)).collect(), h, w),
        180 => (g.data[..w * h].iter().rev().copied().collect(), w, h),
        270 => ((0..w).flat_map(|y| (0..h).map(move |x| (x, y))).map(|(x, y)| at(w - 1 - y, x)).collect(), h, w),
        _ => (g.data[..w * h].to_vec(), w, h),
    }
}

impl Profile {
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).expect("profile serializes")
    }
    pub fn from_json(s: &str) -> serde_json::Result<Self> {
        serde_json::from_str(s)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Status {
    Ok,
    Blank,
    Err,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sample {
    pub x: f32,
    pub y: f32,
    pub on: bool,
    pub ambiguous: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Reading {
    pub value: Option<f64>,
    /// Per digit segment mask, bit0 = a … bit6 = g.
    pub bits: Vec<u8>,
    pub status: Status,
    /// 7 per ROI, for the overlay.
    pub samples: Vec<Sample>,
    /// Mean per-digit threshold, for the binarized view.
    pub thresh: u8,
    /// Boxes actually decoded (after tracking), for the overlay.
    pub rois: Vec<Roi>,
}

/// Segment sample points a..g in normalized digit-box coords.
pub const SEG_POINTS: [(f32, f32); 7] = [
    (0.50, 0.07), // a
    (0.85, 0.28), // b
    (0.85, 0.72), // c
    (0.50, 0.93), // d
    (0.15, 0.72), // e
    (0.15, 0.28), // f
    (0.50, 0.50), // g
];

/// Centers of the two loops of an "8": never lit.
const HOLE_POINTS: [(f32, f32); 2] = [(0.50, 0.28), (0.50, 0.72)];

/// Masks → digit. Includes common 6/7/9 variants (with/without tail).
const PATTERNS: [(u8, u8); 13] = [
    (0x3F, 0),
    (0x06, 1),
    (0x5B, 2),
    (0x4F, 3),
    (0x66, 4),
    (0x6D, 5),
    (0x7D, 6),
    (0x7C, 6),
    (0x07, 7),
    (0x27, 7),
    (0x7F, 8),
    (0x6F, 9),
    (0x67, 9),
];

pub fn segment_digit(bits: u8) -> Option<u8> {
    PATTERNS.iter().find(|(m, _)| *m == bits).map(|(_, d)| *d)
}

pub fn digit_bits(d: u8) -> u8 {
    PATTERNS.iter().find(|(_, x)| *x == d).map(|(m, _)| *m).unwrap_or(0)
}

/// Segment patch half-sizes as (fraction of box w, fraction of box h).
const H_PATCH: (f32, f32) = (0.15, 0.05); // a, d, g
const V_PATCH: (f32, f32) = (0.08, 0.10); // b, c, e, f
const HOLE_PATCH: (f32, f32) = (0.08, 0.05);
/// 'a' is the first thing the bezel hides when the camera looks up at the display → half weight.
const SEG_WEIGHT: [f32; 7] = [0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
/// Digit contrast below this fraction of the strongest digit's → blank (unused leading digit).
const BLANK_FRAC: f32 = 0.5;
/// Best pattern farther than this from the observed segments → nothing matches → Err.
const MAX_COST: f32 = 0.35;

/// q-quantile of a patch in "darkness" space (lit is always low; `invert` flips bright-lit LCDs).
/// Low quantile of an elongated patch = "does a stroke cross it", robust to a few px misalignment.
fn patch_q(img: &Gray, cx: f32, cy: f32, (hw, hh): (f32, f32), q: f32, invert: bool) -> f32 {
    let x0 = ((cx - hw).round() as i32).max(0);
    let x1 = ((cx + hw).round() as i32).min(img.width as i32 - 1);
    let y0 = ((cy - hh).round() as i32).max(0);
    let y1 = ((cy + hh).round() as i32).min(img.height as i32 - 1);
    let mut v: Vec<u8> = (y0..=y1)
        .flat_map(|y| (x0..=x1).map(move |x| (x, y)))
        .map(|(x, y)| img.data[y as usize * img.width + x as usize])
        .map(|p| if invert { 255 - p } else { p })
        .collect();
    if v.is_empty() {
        return f32::NAN;
    }
    let k = ((v.len() - 1) as f32 * q).round() as usize;
    *v.select_nth_unstable(k).1 as f32
}

fn to_px(r: &Roi, shear: f32, (nx, ny): (f32, f32)) -> (f32, f32) {
    (r.x as f32 + nx * r.w as f32 + shear * (0.5 - ny) * r.h as f32, r.y as f32 + ny * r.h as f32)
}

pub fn sample_points(r: &Roi, shear: f32) -> [(f32, f32); 7] {
    SEG_POINTS.map(|p| to_px(r, shear, p))
}

/// Per digit: 7 segment values, 2 hole (background) values, 7 sample centers.
type DigitSamples = ([f32; 7], [f32; 2], [(f32, f32); 7]);

/// Nearest pattern to per-segment "litness" z (0..1): (mask, digit, cost, cost of best other digit).
fn best_pattern(z: &[f32; 7]) -> (u8, u8, f32, f32) {
    let wsum: f32 = SEG_WEIGHT.iter().sum();
    let cost = |m: u8| (0..7).map(|k| SEG_WEIGHT[k] * (z[k] - ((m >> k) & 1) as f32).abs()).sum::<f32>() / wsum;
    let &(m, d) = PATTERNS.iter().min_by(|a, b| cost(a.0).total_cmp(&cost(b.0))).expect("patterns");
    let other = PATTERNS.iter().filter(|p| p.1 != d).map(|p| cost(p.0)).fold(f32::MAX, f32::min);
    (m, d, cost(m), other)
}

/// Per digit: segments = 25th percentile of elongated patches, background = the "8" holes,
/// litness z = (bg - seg) / contrast → nearest pattern. Per-digit contrast (not one global
/// threshold) so a bezel edge or uneven light in one box can't flip the others; soft
/// matching so one blurred / half-hidden segment doesn't kill the digit.
pub fn decode(img: &Gray, p: &Profile) -> Reading {
    let inv = !p.lit_is_dark;
    let per: Vec<DigitSamples> = p
        .rois
        .iter()
        .map(|r| {
            let (w, h) = (r.w as f32, r.h as f32);
            let pts = sample_points(r, p.shear);
            let seg = std::array::from_fn(|k| {
                let (fx, fy) = if matches!(k, 0 | 3 | 6) { H_PATCH } else { V_PATCH };
                patch_q(img, pts[k].0, pts[k].1, (fx * w, fy * h), 0.25, inv)
            });
            let holes = HOLE_POINTS.map(|q| {
                let (x, y) = to_px(r, p.shear, q);
                patch_q(img, x, y, (HOLE_PATCH.0 * w, HOLE_PATCH.1 * h), 0.5, inv)
            });
            (seg, holes, pts)
        })
        .collect();
    let darkest = |seg: &[f32; 7]| seg.iter().copied().fold(f32::MAX, f32::min);
    let range = |seg: &[f32; 7], h: &[f32; 2]| h[0].min(h[1]) - darkest(seg);
    let top = per.iter().map(|(s, h, _)| range(s, h)).filter(|v| v.is_finite()).fold(0.0, f32::max);
    let manual = if inv { 255.0 - p.manual_thresh as f32 } else { p.manual_thresh as f32 };

    let mut bits = Vec::with_capacity(per.len());
    let mut samples = Vec::with_capacity(per.len() * 7);
    let mut digits: Vec<Option<Option<u8>>> = Vec::new(); // None=err, Some(None)=blank
    let mut mids = Vec::new();
    for (seg, holes, pts) in &per {
        let rg = range(seg, holes);
        let lo = darkest(seg);
        let (z, blank): ([f32; 7], bool) = match p.threshold {
            Threshold::Auto => {
                let blank = rg.is_nan() || rg < (BLANK_FRAC * top).max(p.min_contrast as f32);
                // upper segments vs upper hole + upper darkest, lower vs lower: light and the
                // bezel shade the halves differently. Every digit lights >=1 segment per half;
                // the floor keeps an unlit half's noise from being stretched to "on".
                let half_lo = |ks: [usize; 3]| ks.iter().map(|&k| seg[k]).fold(seg[6], f32::min);
                let (top_lo, bot_lo) = (half_lo([0, 1, 5]), half_lo([2, 3, 4]));
                let z = std::array::from_fn(|k| {
                    let (bg, lo) = match k {
                        0 | 1 | 5 => (holes[0], top_lo),
                        2..=4 => (holes[1], bot_lo),
                        _ => (holes[0].min(holes[1]), lo),
                    };
                    ((bg - seg[k]) / (bg - lo).max(0.5 * rg).max(1.0)).clamp(0.0, 1.0)
                });
                (z, blank)
            }
            Threshold::Manual => {
                let z = seg.map(|v| if v <= manual { 1.0 } else { 0.0 });
                (z, z.iter().all(|&v| v == 0.0))
            }
        };
        if blank {
            bits.push(0);
            digits.push(Some(None));
            samples.extend(pts.map(|(x, y)| Sample { x, y, on: false, ambiguous: false }));
            continue;
        }
        mids.push(lo + rg / 2.0);
        let (m, d, c0, c1) = best_pattern(&z);
        bits.push(m);
        digits.push((c0 <= MAX_COST && c1 - c0 >= p.margin).then_some(Some(d)));
        // orange = segment disagrees with the digit it was matched to
        samples.extend((0..7).map(|k| {
            let on = (m >> k) & 1 == 1;
            Sample { x: pts[k].0, y: pts[k].1, on, ambiguous: (z[k] - on as u8 as f32).abs() > 0.35 }
        }));
    }

    let (value, status) = assemble(&digits, p.decimals);
    let t = if mids.is_empty() { 128.0 } else { mids.iter().sum::<f32>() / mids.len() as f32 };
    let t = if inv { 255.0 - t } else { t };
    Reading { value, bits, status, samples, thresh: t.round().clamp(0.0, 255.0) as u8, rois: p.rois.clone() }
}

/// Leading blanks allowed; blank after the first digit or any err → Err.
fn assemble(digits: &[Option<Option<u8>>], decimals: u32) -> (Option<f64>, Status) {
    let mut n: Option<u64> = None;
    for d in digits {
        match (d, n) {
            (None, _) => return (None, Status::Err),
            (Some(None), None) => {}
            (Some(None), Some(_)) => return (None, Status::Err),
            (Some(Some(d)), _) => n = Some(n.unwrap_or(0) * 10 + *d as u64),
        }
    }
    match n {
        None => (None, Status::Blank),
        Some(n) => (Some(n as f64 / 10f64.powi(decimals as i32)), Status::Ok),
    }
}

/// Passes a reading only after N identical consecutive frames (status + bits).
#[derive(Default)]
pub struct Stabilizer {
    last: Option<(Status, Vec<u8>)>,
    run: u32,
}

impl Stabilizer {
    /// True when `r` is stable (never for Err).
    pub fn push(&mut self, r: &Reading, n: u32) -> bool {
        let key = (r.status, r.bits.clone());
        if self.last.as_ref() == Some(&key) {
            self.run = self.run.saturating_add(1);
        } else {
            self.last = Some(key);
            self.run = 1;
        }
        r.status != Status::Err && self.run >= n.max(1)
    }
}
