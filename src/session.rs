//! Trial splitting, peak detection, session files. No GUI deps.

use crate::decoder::{Profile, Reading, Status};
use std::fs::{self, File};
use std::io::{self, Write};
use std::path::{Path, PathBuf};

/// One stabilized observation fed to the detector.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Obs {
    Value(f64),
    Blank,
    /// Err / not yet stable. Keeps state, only lets timers run.
    Unknown,
}

#[derive(Clone, Debug)]
pub struct TrialConfig {
    pub start_kg: f64,
    /// Below start_kg (or blank) this long → trial ends. Also the peak-hold settle time.
    pub end_hold_s: f64,
    /// Device holds the peak on screen: unchanged value for end_hold_s → final.
    pub peak_hold_mode: bool,
}

impl Default for TrialConfig {
    fn default() -> Self {
        Self { start_kg: 2.0, end_hold_s: 1.5, peak_hold_mode: false }
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Trial {
    /// Wall-clock start, seconds since UNIX epoch (for the table).
    pub wall_start: f64,
    pub start_t: f64,
    pub end_t: f64,
    pub peak: f64,
    pub curve: Vec<(f64, f64)>,
    pub subject: String,
    pub hand: String,
    pub memo: String,
}

impl Trial {
    pub fn duration(&self) -> f64 {
        self.end_t - self.start_t
    }
}

struct Active {
    trial: Trial,
    manual: bool,
    below_since: Option<f64>,
    last_val: f64,
    last_change_t: f64,
}

pub struct TrialDetector {
    pub cfg: TrialConfig,
    active: Option<Active>,
    /// Must see a value below start_kg (or blank) before a new auto trial.
    armed: bool,
}

impl TrialDetector {
    pub fn new(cfg: TrialConfig) -> Self {
        Self { cfg, active: None, armed: true }
    }

    pub fn is_active(&self) -> bool {
        self.active.is_some()
    }

    /// Live peak of the running trial.
    pub fn current_peak(&self) -> Option<f64> {
        self.active.as_ref().map(|a| a.trial.peak)
    }

    fn begin(&mut self, t: f64, manual: bool) {
        self.active = Some(Active {
            trial: Trial { start_t: t, end_t: t, ..Default::default() },
            manual,
            below_since: None,
            last_val: f64::NAN,
            last_change_t: t,
        });
    }

    pub fn force_start(&mut self, t: f64) {
        self.begin(t, true);
    }

    pub fn force_stop(&mut self, t: f64) -> Option<Trial> {
        self.active.take().map(|mut a| {
            if a.trial.curve.is_empty() {
                a.trial.end_t = t;
            }
            a.trial
        })
    }

    /// Returns a trial when it ends.
    pub fn push(&mut self, t: f64, obs: Obs) -> Option<Trial> {
        let start = self.cfg.start_kg;
        let Some(a) = self.active.as_mut() else {
            match obs {
                Obs::Value(v) if v >= start && self.armed => {
                    self.begin(t, false);
                    return self.push(t, obs);
                }
                Obs::Value(v) if v < start => self.armed = true,
                Obs::Blank => self.armed = true,
                _ => {}
            }
            return None;
        };

        match obs {
            Obs::Value(v) => {
                a.trial.curve.push((t, v));
                a.trial.peak = a.trial.peak.max(v);
                if v >= start {
                    a.below_since = None;
                    a.trial.end_t = t;
                } else {
                    a.below_since.get_or_insert(t);
                }
                if v != a.last_val {
                    a.last_val = v;
                    a.last_change_t = t;
                }
            }
            Obs::Blank => {
                a.below_since.get_or_insert(t);
            }
            Obs::Unknown => {}
        }
        if a.manual {
            return None;
        }

        let hold = self.cfg.end_hold_s;
        let dropped = a.below_since.is_some_and(|s| t - s >= hold);
        let held = self.cfg.peak_hold_mode && a.last_val >= start && a.below_since.is_none() && t - a.last_change_t >= hold;
        if dropped || held {
            // held: display still shows the peak → wait for a reset (drop) before re-arming.
            self.armed = !held;
            return self.active.take().map(|a| a.trial);
        }
        None
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Summary {
    pub subject: String,
    pub hand: String,
    pub n: usize,
    pub mean_peak: f64,
    pub max_peak: f64,
}

/// Per (subject, hand) mean/max of trial peaks, in first-seen order.
pub fn summarize(trials: &[Trial]) -> Vec<Summary> {
    let mut out: Vec<Summary> = Vec::new();
    for t in trials {
        match out.iter_mut().find(|s| s.subject == t.subject && s.hand == t.hand) {
            Some(s) => {
                s.mean_peak = (s.mean_peak * s.n as f64 + t.peak) / (s.n + 1) as f64;
                s.n += 1;
                s.max_peak = s.max_peak.max(t.peak);
            }
            None => out.push(Summary { subject: t.subject.clone(), hand: t.hand.clone(), n: 1, mean_peak: t.peak, max_peak: t.peak }),
        }
    }
    out
}

pub fn bits_str(bits: &[u8]) -> String {
    bits.iter().map(|b| format!("{b:02x}")).collect::<Vec<_>>().join(" ")
}

/// Session folder: raw.csv (append), trials.csv (rewritten on change), profile.json.
pub struct SessionWriter {
    pub dir: PathBuf,
    raw: csv::Writer<File>,
}

impl SessionWriter {
    pub fn create(dir: &Path, profile: &Profile) -> io::Result<Self> {
        fs::create_dir_all(dir)?;
        fs::write(dir.join("profile.json"), profile.to_json())?;
        let mut raw = csv::Writer::from_path(dir.join("raw.csv"))?;
        raw.write_record(["timestamp", "value", "bits", "status", "stable"])?;
        let s = Self { dir: dir.to_path_buf(), raw };
        s.write_trials(&[])?;
        Ok(s)
    }

    pub fn log_frame(&mut self, t: f64, r: &Reading, stable: bool) -> io::Result<()> {
        let v = r.value.map(|v| v.to_string()).unwrap_or_default();
        let st = match r.status {
            Status::Ok => "ok",
            Status::Blank => "blank",
            Status::Err => "err",
        };
        self.raw.write_record([format!("{t:.4}"), v, bits_str(&r.bits), st.into(), (stable as u8).to_string()])?;
        self.raw.flush()
    }

    pub fn write_trials(&self, trials: &[Trial]) -> io::Result<()> {
        let mut w = csv::Writer::from_path(self.dir.join("trials.csv"))?;
        w.write_record(["wall_start", "subject", "hand", "peak_kg", "duration_s", "memo"])?;
        for t in trials {
            w.write_record([
                format!("{:.3}", t.wall_start),
                t.subject.clone(),
                t.hand.clone(),
                format!("{:.2}", t.peak),
                format!("{:.3}", t.duration()),
                t.memo.clone(),
            ])?;
        }
        w.flush()?;
        // Curves too, so a trial can be re-plotted later.
        let mut c = File::create(self.dir.join("trial_curves.csv"))?;
        writeln!(c, "trial,t,value")?;
        for (i, t) in trials.iter().enumerate() {
            for (ts, v) in &t.curve {
                writeln!(c, "{i},{ts:.4},{v}")?;
            }
        }
        Ok(())
    }
}
