use grip_ocr::decoder::*;
use grip_ocr::session::*;
use grip_ocr::synth::*;

const W: usize = 640;
const H: usize = 360;
const FPS: f64 = 30.0;

/// Full pipeline over a synthetic video: each value shown for `hold` frames,
/// with one half-lit transition frame between changes. Returns finished trials.
fn pipeline(values: &[(f64, usize)], cfg: TrialConfig) -> (Vec<Trial>, usize) {
    let rois = layout(3, W, H);
    let prof = Profile { rois: rois.clone(), ..Default::default() };
    let style = SynthStyle { noise: 8.0, ..Default::default() };
    let mut rng = Lcg(42);
    let mut stab = Stabilizer::default();
    let mut det = TrialDetector::new(cfg);
    let (mut trials, mut logged_transitions, mut frame) = (Vec::new(), 0, 0usize);
    let mut prev: Option<Vec<[f32; 7]>> = None;
    let mut emit = |lv: &[[f32; 7]], transition: bool, frame: &mut usize, trials: &mut Vec<Trial>| {
        let img = render_levels(W, H, &rois, lv, &style, &mut rng);
        let r = decode(&Gray { data: &img, width: W, height: H }, &prof);
        let t = *frame as f64 / FPS;
        *frame += 1;
        let stable = stab.push(&r, prof.stable_frames);
        if transition && stable {
            logged_transitions += 1;
        }
        let obs = match (stable, r.status) {
            (true, Status::Ok) => Obs::Value(r.value.unwrap()),
            (true, Status::Blank) => Obs::Blank,
            _ => Obs::Unknown,
        };
        trials.extend(det.push(t, obs));
    };
    for &(v, hold) in values {
        let lv = if v < 0.0 { levels_for("   ") } else { levels_for(&display_text(v, 3, 1)) };
        if let Some(p) = &prev {
            // half-lit: average of old and new segment states
            let mid: Vec<[f32; 7]> = p.iter().zip(&lv).map(|(a, b)| std::array::from_fn(|i| (a[i] + b[i]) / 2.0)).collect();
            if mid != lv {
                emit(&mid, true, &mut frame, &mut trials);
            }
        }
        for _ in 0..hold {
            emit(&lv, false, &mut frame, &mut trials);
        }
        prev = Some(lv);
    }
    (trials, logged_transitions)
}

fn ramp(peak: f64) -> Vec<(f64, usize)> {
    let mut v = vec![(0.0, 30)];
    for i in 1..=10 {
        v.push((peak * i as f64 / 10.0, 5));
    }
    for i in (0..10).rev() {
        v.push((peak * i as f64 / 10.0, 5));
    }
    v.push((0.0, 90));
    v
}

#[test]
fn single_trial_peak() {
    let (trials, logged) = pipeline(&ramp(35.0), TrialConfig::default());
    assert_eq!(logged, 0, "transition frame leaked");
    assert_eq!(trials.len(), 1);
    assert!((trials[0].peak - 35.0).abs() < 1e-9, "{}", trials[0].peak);
    // ramp: 19 steps of 5 frames >= 2 kg at 30 fps ≈ 3.2 s
    assert!((trials[0].duration() - 3.2).abs() < 0.5, "{}", trials[0].duration());
}

#[test]
fn two_trials_separated() {
    let mut seq = ramp(20.0);
    seq.extend(ramp(41.3));
    let (trials, _) = pipeline(&seq, TrialConfig::default());
    let peaks: Vec<f64> = trials.iter().map(|t| t.peak).collect();
    assert_eq!(peaks.len(), 2, "{peaks:?}");
    assert!((peaks[0] - 20.0).abs() < 1e-9 && (peaks[1] - 41.3).abs() < 1e-9, "{peaks:?}");
}

#[test]
fn blank_ends_trial() {
    let seq = vec![(0.0, 30), (10.0, 10), (30.0, 10), (-1.0, 90)];
    let (trials, _) = pipeline(&seq, TrialConfig::default());
    assert_eq!(trials.len(), 1);
    assert_eq!(trials[0].peak, 30.0);
}

#[test]
fn peak_hold_mode_finalizes_held_value() {
    // rises then the device freezes on the peak; no drop at all
    let seq = vec![(0.0, 30), (10.0, 5), (25.0, 5), (38.2, 120)];
    let (trials, _) = pipeline(&seq, TrialConfig { peak_hold_mode: true, ..Default::default() });
    assert_eq!(trials.len(), 1);
    assert_eq!(trials[0].peak, 38.2);
    // without the mode nothing ends yet
    let (trials, _) = pipeline(&seq, TrialConfig::default());
    assert!(trials.is_empty());
}

#[test]
fn peak_hold_needs_reset_before_next_trial() {
    let mut det = TrialDetector::new(TrialConfig { peak_hold_mode: true, ..Default::default() });
    let mut n = 0;
    let mut t = 0.0;
    for v in [5.0, 30.0] {
        for _ in 0..60 {
            n += det.push(t, Obs::Value(v)).is_some() as usize;
            t += 0.033;
        }
    }
    for _ in 0..100 {
        n += det.push(t, Obs::Value(30.0)).is_some() as usize; // still held
        t += 0.033;
    }
    assert_eq!(n, 1);
    det.push(t, Obs::Value(0.0));
    det.push(t + 0.1, Obs::Value(12.0));
    assert!(det.is_active());
}

#[test]
fn manual_start_stop() {
    let mut det = TrialDetector::new(TrialConfig::default());
    det.force_start(0.0);
    for i in 0..200 {
        assert!(det.push(i as f64 * 0.033, Obs::Value(1.0)).is_none()); // below threshold, manual ignores
    }
    let tr = det.force_stop(7.0).unwrap();
    assert_eq!(tr.peak, 1.0);
    assert!(!det.is_active());
}

#[test]
fn summary_groups() {
    let mk = |s: &str, h: &str, p: f64| Trial { subject: s.into(), hand: h.into(), peak: p, ..Default::default() };
    let s = summarize(&[mk("A", "L", 10.0), mk("A", "L", 20.0), mk("A", "R", 30.0)]);
    assert_eq!(s.len(), 2);
    assert_eq!((s[0].n, s[0].mean_peak, s[0].max_peak), (2, 15.0, 20.0));
}

#[test]
fn writer_files() {
    let dir = std::env::temp_dir().join(format!("grip_ocr_test_{}", std::process::id()));
    let mut w = SessionWriter::create(&dir, &Profile::default()).unwrap();
    let r = Reading { value: Some(1.5), bits: vec![0x06, 0x6D], status: Status::Ok, samples: vec![], thresh: 100, rois: vec![] };
    w.log_frame(0.1, &r, true).unwrap();
    w.write_trials(&[Trial { peak: 3.0, memo: "a,b".into(), curve: vec![(0.0, 3.0)], ..Default::default() }]).unwrap();
    let raw = std::fs::read_to_string(dir.join("raw.csv")).unwrap();
    assert!(raw.contains("0.1000,1.5,06 6d,ok,1"), "{raw}");
    let tr = std::fs::read_to_string(dir.join("trials.csv")).unwrap();
    assert!(tr.contains("\"a,b\""), "{tr}");
    assert!(dir.join("profile.json").exists());
    std::fs::remove_dir_all(dir).ok();
}
