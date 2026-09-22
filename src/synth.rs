//! Synthetic 7-segment renderer. Used by tests and `--synth` demo input.

use crate::decoder::{Roi, digit_bits};

pub struct SynthStyle {
    pub bg: f32,
    pub fg: f32,
    pub shear: f32,
    pub noise: f32,
    /// Brightness gradient added across x (bg+fg), e.g. uneven lighting.
    pub gradient: f32,
}

impl Default for SynthStyle {
    fn default() -> Self {
        Self { bg: 190.0, fg: 40.0, shear: 0.0, noise: 0.0, gradient: 0.0 }
    }
}

/// Segment rects (x0,y0,x1,y1) in normalized digit coords, a..g.
const SEG_RECTS: [(f32, f32, f32, f32); 7] = [
    (0.20, 0.01, 0.80, 0.13), // a
    (0.75, 0.10, 0.95, 0.46), // b
    (0.75, 0.54, 0.95, 0.90), // c
    (0.20, 0.87, 0.80, 0.99), // d
    (0.05, 0.54, 0.25, 0.90), // e
    (0.05, 0.10, 0.25, 0.46), // f
    (0.20, 0.44, 0.80, 0.56), // g
];

pub struct Lcg(pub u64);
impl Lcg {
    pub fn next_f32(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 40) as f32 / (1u64 << 24) as f32
    }
}

/// Canonical digit box layout for `n` digits in a w×h frame.
pub fn layout(n: usize, width: usize, height: usize) -> Vec<Roi> {
    let dw = (width as f32 * 0.12) as i32;
    let dh = (dw as f32 * 1.8) as i32;
    let gap = dw / 3;
    let total = n as i32 * dw + (n as i32 - 1) * gap;
    let x0 = (width as i32 - total) / 2;
    let y0 = (height as i32 - dh) / 2;
    (0..n as i32).map(|i| Roi { x: x0 + i * (dw + gap), y: y0, w: dw, h: dh }).collect()
}

/// Render per-digit segment levels (0 = off, 1 = on, 0.5 = half-lit) into a gray frame.
pub fn render_levels(width: usize, height: usize, rois: &[Roi], levels: &[[f32; 7]], s: &SynthStyle, rng: &mut Lcg) -> Vec<u8> {
    let mut img = vec![0f32; width * height];
    for y in 0..height {
        for x in 0..width {
            img[y * width + x] = s.bg + s.gradient * (x as f32 / width as f32 - 0.5);
        }
    }
    for (r, lv) in rois.iter().zip(levels) {
        let pad = (s.shear.abs() * r.h as f32) as i32 + 2;
        for y in r.y.max(0)..(r.y + r.h).min(height as i32) {
            let ny = (y - r.y) as f32 / r.h as f32;
            for x in (r.x - pad).max(0)..(r.x + r.w + pad).min(width as i32) {
                let nx = (x as f32 - r.x as f32 - s.shear * (0.5 - ny) * r.h as f32) / r.w as f32;
                for (i, &(x0, y0, x1, y1)) in SEG_RECTS.iter().enumerate() {
                    if lv[i] > 0.0 && nx >= x0 && nx < x1 && ny >= y0 && ny < y1 {
                        let p = &mut img[y as usize * width + x as usize];
                        *p += (s.fg - s.bg) * lv[i];
                    }
                }
            }
        }
    }
    img.iter()
        .map(|&v| {
            let n = (rng.next_f32() + rng.next_f32() + rng.next_f32() - 1.5) * 2.0 * s.noise;
            (v + n).clamp(0.0, 255.0) as u8
        })
        .collect()
}

/// Levels for a display string: one char per digit, ' ' = blank. '.' ignored (fixed decimals).
pub fn levels_for(text: &str) -> Vec<[f32; 7]> {
    text.chars()
        .filter(|c| *c != '.')
        .map(|c| {
            let m = c.to_digit(10).map(|d| digit_bits(d as u8)).unwrap_or(0);
            std::array::from_fn(|i| ((m >> i) & 1) as f32)
        })
        .collect()
}

/// Format kg into `n` digit chars with `decimals` places, leading blanks.
pub fn display_text(v: f64, n: usize, decimals: u32) -> String {
    let s = format!("{:.*}", decimals as usize, v).replace('.', "");
    format!("{s:>n$}")
}
