//! grip_ocr — webcam 7-segment reader for a grip dynamometer.
//!
//!   grip_ocr [--camera N] [--size WxH] [--fps F] [--profile FILE]
//!   grip_ocr --video DIR [--profile FILE]   replay a recording, no camera
//!   grip_ocr --make-demo DIR                write a synthetic recording + profile

mod capture;
mod gui;
mod video;

use capture::Source;
use grip_ocr::decoder::Profile;
use std::path::{Path, PathBuf};

fn usage() -> ! {
    eprintln!("usage: grip_ocr [--camera N] [--size WxH] [--fps F] [--profile FILE] [--video DIR] [--make-demo DIR]");
    std::process::exit(2)
}

fn main() -> eframe::Result {
    let mut args = std::env::args().skip(1);
    let (mut cam, mut size, mut fps) = (0u32, (1280u32, 720u32), 30u32);
    let (mut video, mut profile_path): (Option<PathBuf>, Option<PathBuf>) = (None, None);
    while let Some(a) = args.next() {
        let mut val = || args.next().unwrap_or_else(|| usage());
        match a.as_str() {
            "--camera" => cam = val().parse().unwrap_or_else(|_| usage()),
            "--fps" => fps = val().parse().unwrap_or_else(|_| usage()),
            "--size" => {
                let v = val();
                let (w, h) = v.split_once('x').unwrap_or_else(|| usage());
                size = (w.parse().unwrap_or_else(|_| usage()), h.parse().unwrap_or_else(|_| usage()));
            }
            "--list-cameras" => {
                for (i, n) in capture::list_cameras() {
                    println!("{i}: {n}");
                }
                return Ok(());
            }
            "--video" => video = Some(val().into()),
            "--profile" => profile_path = Some(val().into()),
            "--make-demo" => {
                let dir = PathBuf::from(val());
                if let Err(e) = video::make_demo(&dir) {
                    eprintln!("make-demo failed: {e}");
                    std::process::exit(1);
                }
                return Ok(());
            }
            _ => usage(),
        }
    }

    // explicit --profile, else one next to the recording (DIR or its parent), else ./profile.json
    let profile_path = profile_path.unwrap_or_else(|| {
        video
            .as_deref()
            .into_iter()
            .flat_map(|d| [d.join("profile.json"), d.parent().unwrap_or(Path::new(".")).join("profile.json")])
            .find(|p| p.exists())
            .unwrap_or_else(|| "profile.json".into())
    });
    let profile = match std::fs::read_to_string(&profile_path) {
        Ok(s) => Profile::from_json(&s).unwrap_or_else(|e| {
            eprintln!("bad profile {}: {e}", profile_path.display());
            Profile::default()
        }),
        Err(_) => Profile::default(),
    };
    let src = match video {
        Some(d) => Source::Video(d),
        None => Source::Camera { index: cam, width: size.0, height: size.1, fps },
    };

    let opts = eframe::NativeOptions {
        viewport: eframe::egui::ViewportBuilder::default().with_inner_size([1500.0, 900.0]).with_title("grip_ocr"),
        ..Default::default()
    };
    eframe::run_native("grip_ocr", opts, Box::new(move |cc| Ok(Box::new(gui::App::new(cc, src, profile, profile_path)))))
}
