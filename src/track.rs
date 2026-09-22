//! Keeps the digit boxes on the display when the camera or the device moves.
//!
//! Anchor = coarse snapshot (CELL px cells) of the area around the boxes with the
//! boxes masked out: digits change, the bezel / label / window edges don't.
//! Each frame: masked NCC search for shift + scale near the last pose. Lost →
//! coarse full-frame search. Too weak a match (hand in the way) → keep the last pose.

use crate::decoder::{Gray, Roi};
use serde::{Deserialize, Serialize};

pub const CELL: usize = 4;
/// NCC below this = not the anchor.
pub const MIN_SCORE: f32 = 0.6;
const LOCAL: i32 = 3; // cells
const LOCAL_SCALE: f32 = 0.03;
const GLOBAL_SCALES: [f32; 10] = [0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2, 1.25];

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Anchor {
    /// Boxes + rotation the snapshot was taken with; profile differs → retake.
    pub rois: Vec<Roi>,
    pub rotate: u16,
    /// Top-left of the snapshot, full-res px (multiple of CELL).
    pub x: i32,
    pub y: i32,
    pub cols: usize,
    pub rows: usize,
    /// cols*rows cell means, hex.
    pub cells: String,
}

impl Anchor {
    /// Snapshot around `rois`: their bounding box grown 15% sideways, 25% up/down — the LCD
    /// window and the casing right around it. Wider pulls in the mount / bracket, which moves
    /// relative to the display and drags the boxes to a wrong (scaled) pose.
    pub fn capture(img: &Gray, rois: &[Roi], rotate: u16) -> Option<Self> {
        let x0 = rois.iter().map(|r| r.x).min()?;
        let y0 = rois.iter().map(|r| r.y).min()?;
        let x1 = rois.iter().map(|r| r.x + r.w).max()?;
        let y1 = rois.iter().map(|r| r.y + r.h).max()?;
        let (gw, gh) = ((x1 - x0) * 3 / 20, (y1 - y0) / 4);
        let c = CELL as i32;
        let x = ((x0 - gw).max(0) / c) * c;
        let y = ((y0 - gh).max(0) / c) * c;
        let cols = ((x1 + gw).min(img.width as i32) - x) as usize / CELL;
        let rows = ((y1 + gh).min(img.height as i32) - y) as usize / CELL;
        if cols < 4 || rows < 4 {
            return None;
        }
        let (lvl, lc, _) = downsample(img, CELL);
        let (cx, cy) = (x as usize / CELL, y as usize / CELL);
        let cells = (0..rows)
            .flat_map(|i| (0..cols).map(move |j| (i, j)))
            .map(|(i, j)| format!("{:02x}", lvl[(cy + i) * lc + cx + j].round() as u8))
            .collect();
        Some(Self { rois: rois.to_vec(), rotate, x, y, cols, rows, cells })
    }
}

/// Anchor-frame → current-frame: p' = origin + (dx, dy) + s * (p - origin).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Pose {
    pub dx: f32,
    pub dy: f32,
    pub s: f32,
}

impl Pose {
    pub const IDENTITY: Pose = Pose { dx: 0.0, dy: 0.0, s: 1.0 };
}

/// Template cells: (row, col, normalized value); masked cells left out.
type Tmpl = Vec<(f32, f32, f32)>;

pub struct Tracker {
    pub anchor: Anchor,
    pub pose: Pose,
    pub score: f32,
    fine: Tmpl,
    coarse: Tmpl,
}

impl Tracker {
    pub fn new(anchor: Anchor) -> Self {
        let (c, r) = (anchor.cols, anchor.rows);
        let v: Vec<f32> =
            (0..c * r).map(|k| u8::from_str_radix(anchor.cells.get(2 * k..2 * k + 2).unwrap_or("0"), 16).unwrap_or(0) as f32).collect();
        // mask digit boxes (+1 cell): their content changes
        let masked = |i: usize, j: usize| {
            let (px, py) = ((anchor.x + (j * CELL) as i32) as f32, (anchor.y + (i * CELL) as i32) as f32);
            let m = CELL as f32;
            anchor
                .rois
                .iter()
                .any(|q| px + m > q.x as f32 - m && px < (q.x + q.w) as f32 + m && py + m > q.y as f32 - m && py < (q.y + q.h) as f32 + m)
        };
        let fine: Vec<_> = (0..r)
            .flat_map(|i| (0..c).map(move |j| (i, j)))
            .filter(|&(i, j)| !masked(i, j))
            .map(|(i, j)| (i as f32, j as f32, v[i * c + j]))
            .collect();
        // 4x4 blocks for the full-frame search; keep blocks mostly unmasked
        let mut coarse = Vec::new();
        for bi in 0..r / 4 {
            for bj in 0..c / 4 {
                let vals: Vec<f32> =
                    (0..16).map(|k| (bi * 4 + k / 4, bj * 4 + k % 4)).filter(|&(i, j)| !masked(i, j)).map(|(i, j)| v[i * c + j]).collect();
                if vals.len() >= 12 {
                    coarse.push((bi as f32, bj as f32, vals.iter().sum::<f32>() / vals.len() as f32));
                }
            }
        }
        Self { anchor, pose: Pose::IDENTITY, score: 1.0, fine: normalize(fine), coarse: normalize(coarse) }
    }

    /// Track in `img` (same rotation as the anchor). `global` allows the slow full-frame search when lost.
    pub fn update(&mut self, img: &Gray, global: bool) -> Vec<Roi> {
        let (lvl, lc, lr) = downsample(img, CELL);
        // poses are whole cells: dx, dy multiples of CELL
        let c = CELL as i32;
        let cell = |a: i32, d: f32| ((a as f32 + d) / c as f32).round() as i32;
        let (ox, oy) = (cell(self.anchor.x, self.pose.dx), cell(self.anchor.y, self.pose.dy));
        let pose_of = |ox: i32, oy: i32, s: f32| (ox, oy, s);
        let current = ncc(&lvl, lc, lr, &place(&self.fine, self.pose.s), ox, oy);
        let (mut best, mut bp) = (current, (ox, oy, self.pose.s));
        for s in [self.pose.s - LOCAL_SCALE, self.pose.s, self.pose.s + LOCAL_SCALE] {
            let t = place(&self.fine, s);
            for di in -LOCAL..=LOCAL {
                for dj in -LOCAL..=LOCAL {
                    let sc = ncc(&lvl, lc, lr, &t, ox + dj, oy + di);
                    // hysteresis: a 1-cell jitter shouldn't move the boxes
                    if sc > best + 0.01 {
                        (best, bp) = (sc, pose_of(ox + dj, oy + di, s));
                    }
                }
            }
        }
        if best < MIN_SCORE && global {
            let (g, gc, gr) = downsample_f(&lvl, lc, lr, 4);
            let (mut gb, mut gp) = (f32::MIN, (0, 0, 1.0));
            for s in GLOBAL_SCALES {
                let t = place(&self.coarse, s);
                for oy in -(gr as i32) / 4..gr as i32 {
                    for ox in -(gc as i32) / 4..gc as i32 {
                        let sc = ncc(&g, gc, gr, &t, ox, oy);
                        if sc > gb {
                            (gb, gp) = (sc, (ox * 4, oy * 4, s));
                        }
                    }
                }
            }
            // refine at fine level around the coarse hit
            // coarse scales are 0.05 apart: refine across the whole gap, finely
            for s in (-4..=4).map(|k| gp.2 + k as f32 * 0.0125) {
                let t = place(&self.fine, s);
                for di in -4..=4 {
                    for dj in -4..=4 {
                        let sc = ncc(&lvl, lc, lr, &t, gp.0 + dj, gp.1 + di);
                        if sc > best {
                            (best, bp) = (sc, pose_of(gp.0 + dj, gp.1 + di, s));
                        }
                    }
                }
            }
        }
        self.score = best;
        if best >= MIN_SCORE {
            // sub-cell: parabola through the NCC at the neighbouring cells. A whole-cell pose is
            // up to 2 px off, which a segment patch on a small digit can't afford.
            let (bx, by, s) = bp;
            let t = place(&self.fine, s);
            let sub = |l: f32, r: f32| {
                let k = l - 2.0 * best + r;
                if k < -1e-6 { (0.5 * (l - r) / k).clamp(-0.5, 0.5) } else { 0.0 }
            };
            let n = |dx: i32, dy: i32| ncc(&lvl, lc, lr, &t, bx + dx, by + dy);
            let (fx, fy) = (sub(n(-1, 0), n(1, 0)), sub(n(0, -1), n(0, 1)));
            let cf = c as f32;
            self.pose = Pose { dx: (bx as f32 + fx) * cf - self.anchor.x as f32, dy: (by as f32 + fy) * cf - self.anchor.y as f32, s };
        }
        self.rois()
    }

    pub fn rois(&self) -> Vec<Roi> {
        let (ax, ay, p) = (self.anchor.x as f32, self.anchor.y as f32, self.pose);
        self.anchor
            .rois
            .iter()
            .map(|r| Roi {
                x: (ax + p.dx + p.s * (r.x as f32 - ax)).round() as i32,
                y: (ay + p.dy + p.s * (r.y as f32 - ay)).round() as i32,
                w: (p.s * r.w as f32).round() as i32,
                h: (p.s * r.h as f32).round() as i32,
            })
            .collect()
    }

    pub fn lost(&self) -> bool {
        self.score < MIN_SCORE
    }
}

/// Zero-mean, unit-variance template values.
fn normalize(t: Vec<(f32, f32, f32)>) -> Tmpl {
    let n = t.len().max(1) as f32;
    let m = t.iter().map(|c| c.2).sum::<f32>() / n;
    let sd = (t.iter().map(|c| (c.2 - m).powi(2)).sum::<f32>() / n).sqrt().max(1e-3);
    t.into_iter().map(|(i, j, v)| (i, j, (v - m) / sd)).collect()
}

/// Template cell offsets at scale s (nearest sampling), computed once per scale.
fn place(t: &Tmpl, s: f32) -> Vec<(i32, i32, f32)> {
    t.iter().map(|&(i, j, v)| ((s * j).round() as i32, (s * i).round() as i32, v)).collect()
}

/// Masked NCC of a placed template with its origin at cell (ox, oy).
/// Cells off-image are skipped; under 70% on-image → -1.
fn ncc(img: &[f32], cols: usize, rows: usize, t: &[(i32, i32, f32)], ox: i32, oy: i32) -> f32 {
    let (mut n, mut st, mut stt, mut sf, mut sff, mut stf) = (0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32);
    for &(dx, dy, tv) in t {
        let (x, y) = (ox + dx, oy + dy);
        if x < 0 || y < 0 || x >= cols as i32 || y >= rows as i32 {
            continue;
        }
        let f = img[y as usize * cols + x as usize];
        n += 1.0;
        st += tv;
        stt += tv * tv;
        sf += f;
        sff += f * f;
        stf += tv * f;
    }
    if n < 0.7 * t.len() as f32 || n < 8.0 {
        return -1.0;
    }
    let cov = stf - st * sf / n;
    let vt = stt - st * st / n;
    let vf = sff - sf * sf / n;
    if vt <= 1e-6 || vf <= 1e-6 { -1.0 } else { cov / (vt * vf).sqrt() }
}

/// k×k box mean → (cells, cols, rows).
pub fn downsample(img: &Gray, k: usize) -> (Vec<f32>, usize, usize) {
    let (c, r) = (img.width / k, img.height / k);
    let mut out = vec![0.0f32; c * r];
    for y in 0..r * k {
        let row = &img.data[y * img.width..y * img.width + c * k];
        let o = &mut out[(y / k) * c..(y / k + 1) * c];
        for (x, &v) in row.iter().enumerate() {
            o[x / k] += v as f32;
        }
    }
    let n = (k * k) as f32;
    out.iter_mut().for_each(|v| *v /= n);
    (out, c, r)
}

fn downsample_f(img: &[f32], cols: usize, rows: usize, k: usize) -> (Vec<f32>, usize, usize) {
    let (c, r) = (cols / k, rows / k);
    let mut out = vec![0.0f32; c * r];
    for y in 0..r * k {
        for x in 0..c * k {
            out[(y / k) * c + x / k] += img[y * cols + x];
        }
    }
    let n = (k * k) as f32;
    out.iter_mut().for_each(|v| *v /= n);
    (out, c, r)
}
