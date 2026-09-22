//! Real DYNO-200 frames from a C922 session (raw, unrotated; display tops partly under the bezel).
#![cfg(feature = "app")]

use grip_ocr::decoder::*;
use grip_ocr::track::{Anchor, Tracker};

fn frame(name: &str) -> (Vec<u8>, usize, usize) {
    let img = image::open(format!("{}/tests/frames/{name}.jpg", env!("CARGO_MANIFEST_DIR"))).unwrap().to_luma8();
    let (w, h) = (img.width() as usize, img.height() as usize);
    rotate(&Gray { data: img.as_raw(), width: w, height: h }, 270)
}

fn profile() -> Profile {
    Profile {
        rois: [(196, 741, 60, 155), (274, 748, 60, 158), (348, 761, 80, 162), (441, 776, 86, 170)]
            .map(|(x, y, w, h)| Roi { x, y, w, h })
            .to_vec(),
        decimals: 2,
        shear: 0.12,
        ..Default::default()
    }
}

#[test]
fn decodes_recorded_frames() {
    let p = profile();
    for (name, want) in [("000000", None), ("000190", Some(25.20)), ("000300", Some(39.90)), ("000470", Some(16.35))] {
        let (g, w, h) = frame(name);
        let r = decode(&Gray { data: &g, width: w, height: h }, &p);
        assert_eq!(r.value, want, "{name} {:?} {:02x?}", r.status, r.bits);
        if want.is_none() {
            assert_eq!(r.status, Status::Blank, "{name}");
        }
    }
}

/// Nearest-neighbour resample: content moved by (dx, dy) and scaled by s about (cx, cy).
fn warp(g: &[u8], w: usize, h: usize, dx: f32, dy: f32, s: f32, (cx, cy): (f32, f32)) -> Vec<u8> {
    (0..w * h)
        .map(|k| {
            let (x, y) = ((k % w) as f32, (k / w) as f32);
            let (sx, sy) = (cx + (x - cx - dx) / s, cy + (y - cy - dy) / s);
            if sx < 0.0 || sy < 0.0 || sx >= w as f32 || sy >= h as f32 { 128 } else { g[sy as usize * w + sx as usize] }
        })
        .collect()
}

#[test]
fn follows_moved_and_scaled_display() {
    let p = profile();
    let (g, w, h) = frame("000300");
    let a = Anchor::capture(&Gray { data: &g, width: w, height: h }, &p.rois, 270).unwrap();
    for (dx, dy, s) in [(0.0, 0.0, 1.0), (10.0, -7.0, 1.0), (-60.0, 90.0, 1.0), (25.0, 40.0, 1.12), (-30.0, -20.0, 0.9)] {
        let c = (a.x as f32, a.y as f32);
        let moved = warp(&g, w, h, dx, dy, s, c);
        let img = Gray { data: &moved, width: w, height: h };
        let mut t = Tracker::new(a.clone());
        // local search first (may miss big jumps), then full-frame when lost — like the capture loop
        let mut rois = t.update(&img, false);
        if t.lost() {
            rois = t.update(&img, true);
        }
        assert!(!t.lost(), "({dx},{dy},{s}) score {}", t.score);
        for (r, o) in rois.iter().zip(&p.rois) {
            let (ex, ey) = (c.0 + dx + s * (o.x as f32 - c.0), c.1 + dy + s * (o.y as f32 - c.1));
            assert!((r.x as f32 - ex).abs() <= 6.0 && (r.y as f32 - ey).abs() <= 6.0, "({dx},{dy},{s}) {r:?} want ({ex},{ey})");
        }
        let r = decode(&img, &Profile { rois, ..p.clone() });
        assert_eq!(r.value, Some(39.90), "({dx},{dy},{s}) {:02x?}", r.bits);
    }
}
