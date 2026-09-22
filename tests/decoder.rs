use grip_ocr::decoder::*;
use grip_ocr::synth::*;
use std::time::Instant;

const W: usize = 1280;
const H: usize = 720;

fn run(text: &str, style: &SynthStyle, prof: &Profile) -> Reading {
    let rois = layout(text.chars().filter(|c| *c != '.').count(), W, H);
    let img = render_levels(W, H, &rois, &levels_for(text), style, &mut Lcg(7));
    let p = Profile { rois, shear: style.shear, ..prof.clone() };
    decode(&Gray { data: &img, width: W, height: H }, &p)
}

fn styles() -> Vec<(&'static str, SynthStyle)> {
    vec![
        ("plain", SynthStyle::default()),
        ("shear", SynthStyle { shear: 0.18, ..Default::default() }),
        ("noise", SynthStyle { noise: 18.0, ..Default::default() }),
        ("dim", SynthStyle { bg: 90.0, fg: 35.0, noise: 6.0, ..Default::default() }),
        ("gradient", SynthStyle { gradient: 60.0, noise: 8.0, ..Default::default() }),
        ("shear+noise", SynthStyle { shear: 0.18, noise: 12.0, gradient: 30.0, ..Default::default() }),
    ]
}

#[test]
fn every_digit_every_position_every_style() {
    let prof = Profile::default();
    for (name, st) in styles() {
        for d in 0..=9u32 {
            // d in each of 3 positions, digit 8 elsewhere so no leading blanks trip it
            for pos in 0..3 {
                let mut chars = ['8'; 3];
                chars[pos] = char::from_digit(d, 10).unwrap();
                let text: String = [chars[0], chars[1], '.', chars[2]].iter().collect();
                let r = run(&text, &st, &prof);
                let want: f64 = text.parse().unwrap();
                assert_eq!(r.status, Status::Ok, "{name} {text} {:?}", r.bits);
                assert!((r.value.unwrap() - want).abs() < 1e-9, "{name} {text} got {:?}", r.value);
            }
        }
    }
}

#[test]
fn leading_blanks_and_decimals() {
    for (name, st) in styles() {
        let r = run("  0.0", &st, &Profile::default());
        assert_eq!((r.status, r.value), (Status::Ok, Some(0.0)), "{name}");
        let r = run(" 42.7", &st, &Profile::default());
        assert_eq!(r.value, Some(42.7), "{name}");
        let r = run("1234", &st, &Profile { decimals: 2, ..Default::default() });
        assert_eq!(r.value, Some(12.34), "{name}");
        let r = run("1234", &st, &Profile { decimals: 0, ..Default::default() });
        assert_eq!(r.value, Some(1234.0), "{name}");
    }
}

#[test]
fn blank_display() {
    for (name, st) in styles() {
        let r = run("    ", &st, &Profile::default());
        assert_eq!((r.status, r.value), (Status::Blank, None), "{name}");
    }
}

#[test]
fn inner_blank_is_err() {
    let r = run("1 2", &SynthStyle::default(), &Profile::default());
    assert_eq!(r.status, Status::Err);
}

#[test]
fn half_lit_transition_is_err() {
    // 3 → 8: segments e,f half on
    let rois = layout(1, W, H);
    let mut lv = levels_for("3");
    lv[0][4] = 0.5;
    lv[0][5] = 0.5;
    let img = render_levels(W, H, &rois, &lv, &SynthStyle::default(), &mut Lcg(1));
    let r = decode(&Gray { data: &img, width: W, height: H }, &Profile { rois, ..Default::default() });
    assert_eq!(r.status, Status::Err, "{:?}", r.samples);
}

#[test]
fn bright_segments_and_manual_threshold() {
    let st = SynthStyle { bg: 30.0, fg: 220.0, ..Default::default() };
    let r = run(" 17.5", &st, &Profile { lit_is_dark: false, ..Default::default() });
    assert_eq!(r.value, Some(17.5));
    let r = run(" 17.5", &SynthStyle::default(), &Profile { threshold: Threshold::Manual, manual_thresh: 115, ..Default::default() });
    assert_eq!(r.value, Some(17.5));
}

#[test]
fn stabilizer_needs_n_identical() {
    let st = SynthStyle::default();
    let a = run(" 10.0", &st, &Profile::default());
    let b = run(" 10.5", &st, &Profile::default());
    let mut s = Stabilizer::default();
    assert!(!s.push(&a, 3));
    assert!(!s.push(&a, 3));
    assert!(s.push(&a, 3));
    assert!(!s.push(&b, 3));
    let err = Reading { status: Status::Err, ..a.clone() };
    let mut s = Stabilizer::default();
    for _ in 0..5 {
        assert!(!s.push(&err, 3));
    }
}

#[test]
fn profile_json_roundtrip() {
    let p = Profile { rois: layout(4, W, H), shear: 0.1, ..Default::default() };
    assert_eq!(Profile::from_json(&p.to_json()).unwrap(), p);
    // missing fields fall back to defaults
    assert_eq!(Profile::from_json("{}").unwrap(), Profile::default());
}

#[test]
fn decode_is_fast_720p() {
    let rois = layout(4, W, H);
    let img = render_levels(W, H, &rois, &levels_for("12.34"), &SynthStyle { noise: 8.0, ..Default::default() }, &mut Lcg(3));
    let p = Profile { rois, ..Default::default() };
    let g = Gray { data: &img, width: W, height: H };
    let n = 200;
    let t0 = Instant::now();
    for _ in 0..n {
        std::hint::black_box(decode(&g, &p));
    }
    let per = t0.elapsed().as_secs_f64() * 1000.0 / n as f64;
    println!("decode: {per:.3} ms/frame");
    assert!(per < 5.0, "{per} ms");
}

#[test]
fn rotate_clockwise() {
    // 3x2:  a b c / d e f
    let g = Gray { data: &[1, 2, 3, 4, 5, 6], width: 3, height: 2 };
    assert_eq!(rotate(&g, 90), (vec![4, 1, 5, 2, 6, 3], 2, 3));
    assert_eq!(rotate(&g, 180), (vec![6, 5, 4, 3, 2, 1], 3, 2));
    assert_eq!(rotate(&g, 270), (vec![3, 6, 2, 5, 1, 4], 2, 3));
    let r = rotate(&g, 90);
    let back = rotate(&Gray { data: &r.0, width: r.1, height: r.2 }, 270);
    assert_eq!(back, (g.data.to_vec(), 3, 2));
}
