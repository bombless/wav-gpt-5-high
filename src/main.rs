use eframe::egui;
use egui_plot::{HLine, Legend, Line, Plot, PlotPoint, PlotPoints, Text as PlotText};
use ecolor::Color32;
use egui::RichText;
use hound::{SampleFormat, WavReader};
use rodio::{buffer::SamplesBuffer, OutputStream, OutputStreamHandle, Sink};
use rustfft::{num_complex::Complex, FftPlanner};
use std::{env, error::Error, f32::consts::PI, path::Path};
use eframe::egui::Align;
use eframe::emath::{Align2, Vec2b};
use egui_chinese_font::setup_chinese_fonts;

fn main() -> Result<(), Box<dyn Error>> {
    // 命令行参数：wav_path [win_size] [hop_size]
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!(
            "用法: {} <input.wav> [win_size=2048] [hop_size=512]",
            args.get(0).map(|s| s.as_str()).unwrap_or("prog")
        );
        std::process::exit(1);
    }
    let wav_path = &args[1];
    let win_size: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(2048);
    let hop_size: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(512);

    let (mono, sample_rate) = read_wav_mono_f32(wav_path)?;
    let sr_in = sample_rate as f32;
    let duration = mono.len() as f32 / sr_in;

    let (track, global_peak) = dominant_frequency_track(&mono, sr_in, win_size, hop_size)?;
    let fmax = sr_in / 2.0;

    // 合成 44.1kHz 正弦
    let sr_out = 44_100u32;
    let synth = synth_sine_from_track(&track, sr_out, duration, 0.25);

    // 准备 App 状态
    let file_name = Path::new(wav_path)
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("input.wav")
        .to_string();

    let app = App::new(
        file_name,
        duration as f64,
        fmax as f64,
        track
            .iter()
            .map(|(t, f)| [*t as f64, *f as f64])
            .collect(),
        equal_temperament_marks(20.0, fmax as f32)
            .into_iter()
            .map(|(f, name, midi)| (f as f64, name, midi))
            .collect(),
        global_peak.map(|(t, f, m)| (t as f64, f as f64, m)),
        synth,
        sr_out,
    );

    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1280.0, 800.0])
            .with_title("主频轨迹浏览器（支持滚轮缩放/平移）"),
        ..Default::default()
    };

    eframe::run_native(
        "主频轨迹浏览器",
        native_options,
        Box::new(|cc| {
            setup_chinese_fonts(&cc.egui_ctx).expect("Failed to load Chinese fonts");
            Box::new(app)
        }),
    )?;

    Ok(())
}

// ========================== 数据处理 ==========================

fn read_wav_mono_f32(path: &str) -> Result<(Vec<f32>, u32), Box<dyn Error>> {
    let mut reader = WavReader::open(path)?;
    let spec = reader.spec();
    let sr = spec.sample_rate;
    let ch = spec.channels as usize;

    let mono: Vec<f32> = match (spec.sample_format, spec.bits_per_sample) {
        (SampleFormat::Int, 16) => {
            let samples: Result<Vec<i16>, _> = reader.samples::<i16>().collect();
            mixdown_to_mono_i16(&samples?, ch)
        }
        (SampleFormat::Float, 32) => {
            let samples: Result<Vec<f32>, _> = reader.samples::<f32>().collect();
            mixdown_to_mono_f32(&samples?, ch)
        }
        _ => return Err("仅支持 16-bit PCM 或 32-bit float 的 WAV".into()),
    };

    Ok((mono, sr))
}

fn mixdown_to_mono_i16(samples: &[i16], channels: usize) -> Vec<f32> {
    if channels == 1 {
        samples.iter().map(|&s| s as f32 / 32768.0).collect()
    } else {
        let mut mono = Vec::with_capacity(samples.len() / channels);
        for frame in samples.chunks_exact(channels) {
            let mut acc = 0.0f32;
            for &s in frame {
                acc += s as f32 / 32768.0;
            }
            mono.push(acc / channels as f32);
        }
        mono
    }
}

fn mixdown_to_mono_f32(samples: &[f32], channels: usize) -> Vec<f32> {
    if channels == 1 {
        samples.to_vec()
    } else {
        let mut mono = Vec::with_capacity(samples.len() / channels);
        for frame in samples.chunks_exact(channels) {
            let mut acc = 0.0f32;
            for &s in frame {
                acc += s;
            }
            mono.push(acc / channels as f32);
        }
        mono
    }
}

fn dominant_frequency_track(
    mono: &[f32],
    sr: f32,
    win_size: usize,
    hop_size: usize,
) -> Result<(Vec<(f32, f32)>, Option<(f32, f32, f32)>), Box<dyn Error>> {
    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(win_size);
    let hann: Vec<f32> = (0..win_size)
        .map(|n| 0.5 * (1.0 - (2.0 * PI * n as f32 / (win_size as f32 - 1.0)).cos()))
        .collect();

    let mut track = Vec::<(f32, f32)>::new();
    let mut global_peak: Option<(f32, f32, f32)> = None;

    let nyquist = sr / 2.0;
    let mut start = 0usize;

    while start + win_size <= mono.len() {
        let mut buf: Vec<Complex<f32>> = (0..win_size)
            .map(|i| Complex {
                re: mono[start + i] * hann[i],
                im: 0.0,
            })
            .collect();

        fft.process(&mut buf);

        let half = win_size / 2;
        let mut max_idx = 0usize;
        let mut max_mag2 = 0.0f32;
        for k in 0..half {
            let c = buf[k];
            let mag2 = c.re * c.re + c.im * c.im;
            if mag2 > max_mag2 {
                max_mag2 = mag2;
                max_idx = k;
            }
        }

        let t = start as f32 / sr;
        let f = (max_idx as f32 * sr / win_size as f32).clamp(0.0, nyquist);
        track.push((t, f));

        if max_mag2 > global_peak.map(|(_, _, m)| m).unwrap_or(-1.0) {
            global_peak = Some((t, f, max_mag2));
        }

        start += hop_size;
    }

    Ok((track, global_peak))
}

// 生成 [fmin, fmax] 内的十二平均律标注（返回：频率、名、MIDI）
fn equal_temperament_marks(fmin: f32, fmax: f32) -> Vec<(f32, String, i32)> {
    let names = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"];
    let mut v = Vec::new();
    for midi in 0..=127 {
        let f = 440.0 * 2f32.powf((midi as f32 - 69.0) / 12.0);
        if f >= fmin && f <= fmax {
            let pc = (midi % 12) as usize;
            let octave = (midi / 12) - 1; // MIDI 60 -> C4
            let name = format!("{}{}", names[pc], octave);
            v.push((f, name, midi));
        }
    }
    v
}

// ========================== 合成与播放 ==========================

fn synth_sine_from_track(
    track: &[(f32, f32)],
    sr_out: u32,
    duration: f32,
    amp: f32,
) -> Vec<f32> {
    if track.is_empty() || duration <= 0.0 {
        return vec![];
    }

    // let mut t0 = track[0].0;
    let local_track = track.to_vec();
    // if t0 > 0.0 {
    //     local_track.insert(0, (0.0, track[0].1));
    //     t0 = 0.0;
    // }

    let n = (duration * sr_out as f32).round() as usize;
    let sr_out_f = sr_out as f32;
    let nyq_out = sr_out_f / 2.0;

    let mut y = Vec::with_capacity(n);
    let mut phase = 0.0f32;
    let mut k = 0usize;

    for i in 0..n {
        let t = i as f32 / sr_out_f;
        while k + 1 < local_track.len() && t > local_track[k + 1].0 {
            k += 1;
        }

        let f_inst = if k + 1 < local_track.len() {
            let (t0, f0) = local_track[k];
            let (t1, f1) = local_track[k + 1];
            if t1 > t0 {
                let a = (t - t0) / (t1 - t0);
                f0 + a * (f1 - f0)
            } else {
                local_track[k].1
            }
        } else {
            local_track.last().map(|(_, f)| *f).unwrap_or(0.0)
        }
            .clamp(0.0, nyq_out);

        phase += 2.0 * PI * f_inst / sr_out_f;
        y.push(amp * phase.sin());
    }

    // 20ms 淡入淡出
    let fade = (0.02 * sr_out_f) as usize;
    for i in 0..fade.min(y.len()) {
        let g = i as f32 / fade as f32;
        y[i] *= g;
        let j = y.len() - 1 - i;
        y[j] *= g;
    }

    y
}

// ========================== GUI 应用 ==========================

struct App {
    file_name: String,
    duration: f64,
    fmax: f64,
    track: Vec<[f64; 2]>,                 // (t, f)
    global_peak: Option<(f64, f64, f32)>, // (t, f, mag2)
    note_marks: Vec<(f64, String, i32)>,  // (freq, name, midi)

    // 交互选项
    show_note_lines: bool,
    dense_threshold: usize, // > 阈值时仅标 Cn/A4/中央C
    time_bounds: (f64, f64), // 初始视图
    freq_bounds: (f64, f64),

    // 音频播放
    synth: Vec<f32>,
    sr_out: u32,
    stream: Option<OutputStream>,
    handle: Option<OutputStreamHandle>,
    sink: Option<Sink>,
    playing: bool,
}

impl App {
    fn new(
        file_name: String,
        duration: f64,
        fmax: f64,
        track: Vec<[f64; 2]>,
        note_marks: Vec<(f64, String, i32)>,
        global_peak: Option<(f64, f64, f32)>,
        synth: Vec<f32>,
        sr_out: u32,
    ) -> Self {
        Self {
            file_name,
            duration,
            fmax,
            time_bounds: (0.0, duration.max(1e-6)),
            freq_bounds: (0.0, fmax.max(1.0)),
            track,
            global_peak,
            note_marks,
            show_note_lines: true,
            dense_threshold: 36,
            synth,
            sr_out,
            stream: None,
            handle: None,
            sink: None,
            playing: false,
        }
    }

    fn start_play(&mut self) {
        if self.synth.is_empty() || self.playing {
            return;
        }
        if self.stream.is_none() {
            if let Ok((stream, handle)) = OutputStream::try_default() {
                self.handle = Some(handle);
                self.stream = Some(stream);
            }
        }
        if let (Some(handle), None) = (&self.handle, &self.sink) {
            if let Ok(sink) = Sink::try_new(handle) {
                let buf = SamplesBuffer::new(1, self.sr_out, self.synth.clone());
                sink.append(buf);
                sink.set_volume(0.9);
                self.sink = Some(sink);
                self.playing = true;
            }
        }
        if let Some(sink) = &self.sink {
            sink.play();
            self.playing = true;
        }
    }

    fn stop_play(&mut self) {
        if let Some(sink) = &self.sink {
            sink.stop();
        }
        self.sink = None;
        self.playing = false;
        // 不释放 stream/handle，避免反复创建；如需释放可同时置 None
    }

    fn draw_plot(&self, ui: &mut egui::Ui) {
        let plot = Plot::new("dominant_freq_plot")
            .legend(Legend::default())
            .allow_scroll(true)
            .allow_zoom(true)
            .allow_boxed_zoom(true)
            .allow_drag(true)
            .auto_bounds(Vec2b::FALSE)
            .include_x(self.time_bounds.0)
            .include_x(self.time_bounds.1)
            .include_y(self.freq_bounds.0)
            .include_y(self.freq_bounds.1)
            .label_formatter(|name, value| {
                if !name.is_empty() {
                    format!("{name}\n时间: {:.3}s\n频率: {:.1}Hz", value.x, value.y)
                } else {
                    format!("时间: {:.3}s\n频率: {:.1}Hz", value.x, value.y)
                }
            });

        plot.show(ui, |plot_ui| {
            // 十二平均律水平线
            if self.show_note_lines {
                let dense = self.note_marks.len() > self.dense_threshold;
                let bounds = plot_ui.plot_bounds();
                let x_span = bounds.max()[0] - bounds.min()[0];
                let label_x = bounds.min()[0] + 0.01 * x_span;

                for (f, name, midi) in &self.note_marks {
                    let is_c = *midi % 12 == 0;
                    let is_a4 = *midi == 69;
                    let is_c4 = *midi == 60;
                    let show_label = if dense { is_c || is_a4 || is_c4 } else { true };

                    let mut line = HLine::new(*f).color(Color32::from_rgba_unmultiplied(120, 140, 200, 90));
                    if is_c4 {
                        // 中央 C 更显眼
                        line = HLine::new(*f).color(Color32::from_rgb(25, 130, 196));
                    }
                    plot_ui.hline(line);

                    if show_label {
                        let label = if is_c4 {
                            format!("{name} (中央C) {:.1}Hz", f)
                        } else {
                            format!("{name} {:.1}Hz", f)
                        };
                        plot_ui.text(
                            PlotText::new(PlotPoint {x: label_x, y: *f }, label)
                                .color(Color32::from_rgb(70, 70, 110))
                                .anchor(Align2([Align::Min, Align::Center])) // 左对齐，垂直居中
                                .name("notes"),
                        );
                    }
                }
            }

            // 主频轨迹
            let line = Line::new(PlotPoints::from_iter(self.track.iter().map(|p| [p[0], p[1]])))
                .name("主频轨迹")
                .color(Color32::from_rgb(220, 20, 60));
            plot_ui.line(line);

            // 全局峰值标记
            if let Some((t_peak, f_peak, _)) = self.global_peak {
                let peak_line = Line::new(PlotPoints::from_iter([[t_peak, f_peak], [t_peak, f_peak]]))
                    .name(format!("峰值 {:.3}s, {:.1}Hz", t_peak, f_peak))
                    .color(Color32::from_rgb(25, 130, 196));
                plot_ui.line(peak_line);
            }

            // 鼠标坐标提示（光标位置 -> 最近音名）
            if let Some(pointer) = plot_ui.pointer_coordinate() {
                let (name, f_note) = nearest_note(pointer.y);
                let txt = format!("最近音: {name} ≈ {:.1}Hz", f_note);
                plot_ui.text(
                    PlotText::new(PlotPoint {x: pointer.x, y: pointer.y}, txt)
                        .anchor(Align2([Align::Min, Align::Center]))
                        .color(Color32::from_rgb(50, 50, 50)),
                );
            }
        });
    }
}

// 最近的十二平均律音名
fn nearest_note(freq: f64) -> (String, f64) {
    if freq <= 0.0 {
        return ("N/A".into(), 0.0);
    }
    let midi = (69.0 + 12.0 * (freq / 440.0).log2()).round();
    let midi_i = midi.clamp(0.0, 127.0) as i32;
    let names = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"];
    let pc = (midi_i % 12) as usize;
    let octave = (midi_i / 12) - 1;
    let name = format!("{}{}", names[pc], octave);
    let f = 440.0 * 2f64.powf((midi_i as f64 - 69.0) / 12.0);
    (name, f)
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::TopBottomPanel::top("top").show(ctx, |ui| {
            ui.horizontal_wrapped(|ui| {
                ui.label(RichText::new(format!("文件: {}", self.file_name)).strong());
                ui.separator();
                ui.label(format!("时长: {:.3}s", self.duration));
                ui.separator();
                ui.label(format!("Nyquist: {:.0}Hz", self.fmax));
                ui.separator();
                ui.checkbox(&mut self.show_note_lines, "显示十二平均律标线");
                ui.separator();
                if ui.button(if self.playing { "停止播放" } else { "播放合成音" }).clicked() {
                    if self.playing {
                        self.stop_play();
                    } else {
                        self.start_play();
                    }
                }
                if ui.button("复位视图").clicked() {
                    self.time_bounds = (0.0, self.duration.max(1e-6));
                    self.freq_bounds = (0.0, self.fmax.max(1.0));
                }
                ui.label("提示：滚轮在指针处缩放，拖拽平移，框选放大");
            });
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            self.draw_plot(ui);
        });
    }
}
