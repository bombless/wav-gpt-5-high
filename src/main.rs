use eframe::egui;
use egui_plot::{HLine, Legend, Line, Plot, PlotPoint, PlotPoints, Text as PlotText, VLine};
use ecolor::Color32;
use egui::RichText;
use hound::{SampleFormat, WavReader};
use rodio::{buffer::SamplesBuffer, OutputStream, OutputStreamHandle, Sink};
use rustfft::{num_complex::Complex, FftPlanner};
use std::{env, error::Error, f32::consts::PI, path::Path, time::Instant};
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

    let (track, sampled_track, global_peak) = dominant_frequency_track(&mono, sr_in, win_size, hop_size)?;
    let fmax = sr_in / 2.0;

    // 准备 App 状态
    let file_name = Path::new(wav_path)
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("input.wav")
        .to_string();

    let sr_out = 44_100u32;

    let app = App::new(
        file_name,
        duration as f64,
        fmax as f64,
        track
            .iter()
            .map(|(t, f)| [*t as f64, *f as f64])
            .collect(),
        sampled_track
            .iter()
            .map(|(t, freqs)| (*t as f64, [freqs[0] as f64, freqs[1] as f64, freqs[2] as f64]))
            .collect(),
        equal_temperament_marks(20.0, fmax as f32)
            .into_iter()
            .map(|(f, name, midi)| (f as f64, name, midi))
            .collect(),
        global_peak.map(|(t, f, m)| (t as f64, f as f64, m)),
        sr_out,
        duration as f64,
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
) -> Result<(Vec<(f32, f32)>, Vec<(f32, [f32; 3])>, Option<(f32, f32, f32)>), Box<dyn Error>> {
    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(win_size);
    let hann: Vec<f32> = (0..win_size)
        .map(|n| 0.5 * (1.0 - (2.0 * PI * n as f32 / (win_size as f32 - 1.0)).cos()))
        .collect();

    let mut track = Vec::<(f32, f32)>::new();
    let mut sampled_track = Vec::<(f32, [f32; 3])>::new();
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

        // 找出所有频率的幅度
        let mut peaks: Vec<(usize, f32)> = (0..half)
            .map(|k| {
                let c = buf[k];
                let mag2 = c.re * c.re + c.im * c.im;
                (k, mag2)
            })
            .collect();

        // 按幅度降序排序
        peaks.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // 取前三个峰值
        let top3_freqs: [f32; 3] = [
            (peaks[0].0 as f32 * sr / win_size as f32).clamp(0.0, nyquist),
            if peaks.len() > 1 {
                (peaks[1].0 as f32 * sr / win_size as f32).clamp(0.0, nyquist)
            } else {
                0.0
            },
            if peaks.len() > 2 {
                (peaks[2].0 as f32 * sr / win_size as f32).clamp(0.0, nyquist)
            } else {
                0.0
            },
        ];

        let t = start as f32 / sr;
        let f_max = top3_freqs[0]; // 最大值
        let max_mag2 = peaks[0].1;

        track.push((t, f_max));
        sampled_track.push((t, top3_freqs));

        if max_mag2 > global_peak.map(|(_, _, m)| m).unwrap_or(-1.0) {
            global_peak = Some((t, f_max, max_mag2));
        }

        start += hop_size;
    }

    Ok((track, sampled_track, global_peak))
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

    let local_track = track.to_vec();

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

#[derive(Debug, Clone, Copy, PartialEq)]
enum PlaybackTrack {
    Max,      // 最大值（红线）
    Sample1,  // 采样1（紫线）
    Sample2,  // 采样2（紫线）
    Sample3,  // 采样3（紫线）
}

impl PlaybackTrack {
    fn label(&self) -> &str {
        match self {
            PlaybackTrack::Max => "最大值（红线）",
            PlaybackTrack::Sample1 => "采样频率 #1",
            PlaybackTrack::Sample2 => "采样频率 #2",
            PlaybackTrack::Sample3 => "采样频率 #3",
        }
    }
}

struct App {
    file_name: String,
    duration: f64,
    fmax: f64,
    track: Vec<[f64; 2]>,                      // (t, f_max) - 最大值轨迹
    sampled_track: Vec<(f64, [f64; 3])>,      // (t, [f1, f2, f3]) - 三个采样频率
    global_peak: Option<(f64, f64, f32)>,      // (t, f, mag2)
    note_marks: Vec<(f64, String, i32)>,       // (freq, name, midi)

    // 交互选项
    show_note_lines: bool,
    show_sampled_freqs: bool,                  // 是否显示三个采样频率
    dense_threshold: usize,
    time_bounds: (f64, f64),
    freq_bounds: (f64, f64),

    // 节拍功能
    bpm: f64,                                  // 每分钟节拍数
    show_beat_lines: bool,                     // 是否显示节拍线
    beats_per_bar: usize,                      // 每小节拍数（用于强调小节线）

    // 音频播放
    selected_track: PlaybackTrack,             // 选择播放的轨迹
    sr_out: u32,
    stream: Option<OutputStream>,
    handle: Option<OutputStreamHandle>,
    sink: Option<Sink>,
    playing: bool,
    play_start_time: Option<Instant>,
    play_position: f64,
}

impl App {
    fn new(
        file_name: String,
        duration: f64,
        fmax: f64,
        track: Vec<[f64; 2]>,
        sampled_track: Vec<(f64, [f64; 3])>,
        note_marks: Vec<(f64, String, i32)>,
        global_peak: Option<(f64, f64, f32)>,
        sr_out: u32,
        _total_duration: f64,
    ) -> Self {
        Self {
            file_name,
            duration,
            fmax,
            time_bounds: (0.0, duration.max(1e-6)),
            freq_bounds: (0.0, fmax.max(1.0)),
            track,
            sampled_track,
            global_peak,
            note_marks,
            show_note_lines: true,
            show_sampled_freqs: true,
            dense_threshold: 36,
            bpm: 120.0,                        // 默认 120 BPM
            show_beat_lines: true,             // 默认显示节拍线
            beats_per_bar: 4,                  // 默认 4/4 拍
            selected_track: PlaybackTrack::Max,
            sr_out,
            stream: None,
            handle: None,
            sink: None,
            playing: false,
            play_start_time: None,
            play_position: 0.0,
        }
    }

    // 根据选择的轨迹生成播放数据
    fn get_selected_track_data(&self) -> Vec<(f32, f32)> {
        match self.selected_track {
            PlaybackTrack::Max => {
                self.track.iter().map(|p| (p[0] as f32, p[1] as f32)).collect()
            }
            PlaybackTrack::Sample1 => {
                self.sampled_track.iter().map(|(t, freqs)| (*t as f32, freqs[0] as f32)).collect()
            }
            PlaybackTrack::Sample2 => {
                self.sampled_track.iter().map(|(t, freqs)| (*t as f32, freqs[1] as f32)).collect()
            }
            PlaybackTrack::Sample3 => {
                self.sampled_track.iter().map(|(t, freqs)| (*t as f32, freqs[2] as f32)).collect()
            }
        }
    }

    fn start_play(&mut self) {
        if self.playing {
            return;
        }

        // 生成选定轨迹的音频
        let track_data = self.get_selected_track_data();
        let synth = synth_sine_from_track(&track_data, self.sr_out, self.duration as f32, 0.25);

        if synth.is_empty() {
            return;
        }

        if self.stream.is_none() {
            if let Ok((stream, handle)) = OutputStream::try_default() {
                self.handle = Some(handle);
                self.stream = Some(stream);
            }
        }

        if let Some(handle) = &self.handle {
            if let Ok(sink) = Sink::try_new(handle) {
                let buf = SamplesBuffer::new(1, self.sr_out, synth);
                sink.append(buf);
                sink.set_volume(0.9);
                sink.play();
                self.sink = Some(sink);
                self.playing = true;
                self.play_start_time = Some(Instant::now());
                self.play_position = 0.0;
            }
        }
    }

    fn stop_play(&mut self) {
        if let Some(sink) = &self.sink {
            sink.stop();
        }
        self.sink = None;
        self.playing = false;
        self.play_start_time = None;
        self.play_position = 0.0;
    }

    fn update_play_position(&mut self) {
        if self.playing {
            if let Some(start_time) = self.play_start_time {
                let elapsed = start_time.elapsed().as_secs_f64();
                self.play_position = elapsed.min(self.duration);

                if let Some(sink) = &self.sink {
                    if sink.empty() {
                        self.playing = false;
                        self.play_start_time = None;
                        self.play_position = 0.0;
                    }
                }
            }
        }
    }

    fn draw_plot(&self, ui: &mut egui::Ui) {
        let plot = Plot::new("dominant_freq_plot")
            .legend(Legend::default())
            .allow_scroll(false)
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
            // 节拍线（蓝色竖线）
            if self.show_beat_lines && self.bpm > 0.0 {
                let beat_duration = 60.0 / self.bpm;  // 每拍的时长（秒）
                let bounds = plot_ui.plot_bounds();
                let y_span = bounds.max()[1] - bounds.min()[1];
                let label_y = bounds.min()[1] + 0.95 * y_span;  // 标签位置在顶部

                // 计算需要显示的节拍范围
                let start_beat = (bounds.min()[0] / beat_duration).floor() as i32;
                let end_beat = (bounds.max()[0] / beat_duration).ceil() as i32;

                for beat_num in start_beat..=end_beat {
                    if beat_num < 0 {
                        continue;
                    }

                    let beat_time = beat_num as f64 * beat_duration;

                    // 跳过超出音频时长的节拍
                    if beat_time > self.duration {
                        break;
                    }

                    // 判断是否为小节的第一拍（强拍）
                    let is_bar_start = beat_num as usize % self.beats_per_bar == 0;

                    // 强拍用更粗更深的蓝线，弱拍用细一点的浅蓝线
                    let (color, width) = if is_bar_start {
                        (Color32::from_rgb(0, 100, 200), 2.0)  // 深蓝色，粗线
                    } else {
                        (Color32::from_rgba_unmultiplied(100, 150, 255, 150), 1.0)  // 浅蓝色，细线
                    };

                    let beat_line = VLine::new(beat_time)
                        .color(color)
                        .width(width);
                    plot_ui.vline(beat_line);

                    // 只在小节的第一拍显示小节编号
                    if is_bar_start {
                        let bar_num = beat_num as usize / self.beats_per_bar + 1;
                        let label = format!("小节 {}", bar_num);
                        plot_ui.text(
                            PlotText::new(PlotPoint { x: beat_time, y: label_y }, label)
                                .color(Color32::from_rgb(0, 100, 200))
                                .anchor(Align2([Align::Center, Align::Max]))
                                .name("beats"),
                        );
                    }
                }
            }

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
                                .anchor(Align2([Align::Min, Align::Center]))
                                .name("notes"),
                        );
                    }
                }
            }

            // 绘制三个采样频率（紫色）
            if self.show_sampled_freqs {
                for i in 0..3 {
                    let points: Vec<[f64; 2]> = self.sampled_track.iter()
                        .map(|(t, freqs)| [*t, freqs[i]])
                        .collect();

                    // 如果当前选择播放这条轨迹，加粗显示
                    let (width, color) = match (i, self.selected_track) {
                        (0, PlaybackTrack::Sample1) => (2.5, Color32::from_rgb(200, 100, 255)),
                        (1, PlaybackTrack::Sample2) => (2.5, Color32::from_rgb(200, 100, 255)),
                        (2, PlaybackTrack::Sample3) => (2.5, Color32::from_rgb(200, 100, 255)),
                        _ => (1.5, Color32::from_rgb(147, 51, 234)),
                    };

                    let line = Line::new(PlotPoints::from_iter(points))
                        .name(format!("采样频率 #{}", i + 1))
                        .color(color)
                        .width(width);
                    plot_ui.line(line);
                }
            }

            // 主频轨迹（最大值，红色）
            let (max_width, max_color) = if self.selected_track == PlaybackTrack::Max {
                (3.0, Color32::from_rgb(255, 50, 80))
            } else {
                (2.0, Color32::from_rgb(220, 20, 60))
            };

            let line = Line::new(PlotPoints::from_iter(self.track.iter().map(|p| [p[0], p[1]])))
                .name("主频轨迹（最大值）")
                .color(max_color)
                .width(max_width);
            plot_ui.line(line);

            // 全局峰值标记
            if let Some((t_peak, f_peak, _)) = self.global_peak {
                let peak_line = Line::new(PlotPoints::from_iter([[t_peak, f_peak], [t_peak, f_peak]]))
                    .name(format!("峰值 {:.3}s, {:.1}Hz", t_peak, f_peak))
                    .color(Color32::from_rgb(25, 130, 196));
                plot_ui.line(peak_line);
            }

            // 播放位置竖线
            if self.playing && self.play_position > 0.0 {
                let play_line = VLine::new(self.play_position)
                    .name(format!("播放位置: {:.2}s", self.play_position))
                    .color(Color32::from_rgba_unmultiplied(0, 255, 0, 200))
                    .width(2.0);
                plot_ui.vline(play_line);
            }

            // 鼠标坐标提示
            if let Some(pointer) = plot_ui.pointer_coordinate() {
                let (name, f_note) = nearest_note(pointer.y);
                let txt = format!("最近音: {name} ≈ {:.1}Hz", f_note);
                plot_ui.text(
                    PlotText::new(PlotPoint {x: pointer.x + 0.2, y: pointer.y}, txt)
                        .anchor(Align2([Align::Min, Align::Min]))
                        .color(Color32::from_rgb(250, 50, 50)),
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
        self.update_play_position();

        if self.playing {
            ctx.request_repaint();
        }

        egui::TopBottomPanel::top("top").show(ctx, |ui| {
            ui.horizontal_wrapped(|ui| {
                ui.checkbox(&mut self.show_note_lines, "显示十二平均律标线");
                ui.separator();
                ui.checkbox(&mut self.show_sampled_freqs, "显示采样频率");
                ui.separator();

                // 节拍控制
                ui.checkbox(&mut self.show_beat_lines, "显示节拍线");
                if self.show_beat_lines {
                    ui.separator();
                    ui.label("BPM:");
                    ui.add(egui::DragValue::new(&mut self.bpm)
                        .speed(1.0)
                        .clamp_range(30.0..=300.0));

                    ui.separator();
                    ui.label("拍号:");
                    egui::ComboBox::from_id_source("beats_per_bar")
                        .selected_text(format!("{}/4", self.beats_per_bar))
                        .show_ui(ui, |ui| {
                            ui.selectable_value(&mut self.beats_per_bar, 3, "3/4");
                            ui.selectable_value(&mut self.beats_per_bar, 4, "4/4");
                            ui.selectable_value(&mut self.beats_per_bar, 5, "5/4");
                            ui.selectable_value(&mut self.beats_per_bar, 6, "6/4");
                        });
                }
                ui.separator();

                // 播放轨迹选择
                ui.label("播放轨迹:");
                let prev_selection = self.selected_track;
                egui::ComboBox::from_id_source("track_selector")
                    .selected_text(self.selected_track.label())
                    .show_ui(ui, |ui| {
                        ui.selectable_value(&mut self.selected_track, PlaybackTrack::Max, PlaybackTrack::Max.label());
                        ui.selectable_value(&mut self.selected_track, PlaybackTrack::Sample1, PlaybackTrack::Sample1.label());
                        ui.selectable_value(&mut self.selected_track, PlaybackTrack::Sample2, PlaybackTrack::Sample2.label());
                        ui.selectable_value(&mut self.selected_track, PlaybackTrack::Sample3, PlaybackTrack::Sample3.label());
                    });

                // 如果正在播放时切换了轨迹，停止当前播放
                if self.playing && prev_selection != self.selected_track {
                    self.stop_play();
                }

                ui.separator();
                if ui.button(if self.playing { "停止播放" } else { "播放合成音" }).clicked() {
                    if self.playing {
                        self.stop_play();
                    } else {
                        self.start_play();
                    }
                }
                ui.separator();
                ui.label(RichText::new(format!("文件: {}", self.file_name)).strong());
                ui.separator();
                ui.label(format!("时长: {:.3}s", self.duration));
                ui.separator();
                ui.label(format!("Nyquist: {:.0}Hz", self.fmax));

                if self.playing {
                    ui.separator();
                    ui.label(RichText::new(format!("▶ 播放中: {:.2}s / {:.2}s ({})",
                                                   self.play_position, self.duration, self.selected_track.label()))
                        .color(Color32::from_rgb(0, 200, 0)));
                }
            });
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            self.draw_plot(ui);
        });
    }
}
