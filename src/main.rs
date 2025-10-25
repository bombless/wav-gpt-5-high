
mod color_gemini;
mod music;

use eframe::egui;
use egui_plot::{HLine, Legend, Line, Plot, PlotPoint, PlotPoints, Points, Polygon, Text as PlotText, VLine};
use egui::{RichText, Color32, Align, Stroke, Align2, Vec2b, pos2, vec2, CornerRadius, Painter, Pos2, Rect, StrokeKind, Vec2, Window, Shape::LineSegment, FontId};
use rodio::{buffer::SamplesBuffer, OutputStream, OutputStreamHandle, Sink};
use std::{fs, env, error::Error, f32::consts::PI, path::Path, time::Instant};
use std::collections::HashMap;
use std::time::Duration;
use eframe::egui::CursorIcon;
use eframe::epaint::{PathShape, RectShape};
use egui_chinese_font::setup_chinese_fonts;
use egui_plot::MarkerShape::{Circle, Cross};
use serde::{Deserialize, Serialize};

const CONFIG_PATH: &'static str = "app.toml";

#[derive(Serialize, Deserialize, Default, Clone, PartialEq, Debug)]
struct TrackCfg {
    bpm: f64,
    beats_per_bar: usize,
    playback_track: PlaybackTrack,
}

#[derive(Serialize, Deserialize, Default)]
struct Config {
    // 键用字符串，避免 Windows 路径分隔符在 TOML key 里转义
    tracks: HashMap<String, TrackCfg>,
}

impl Config {
    fn load(path: &Path) -> anyhow::Result<Self> {
        Ok(if path.exists() {
            toml::from_str(&fs::read_to_string(path)?)?
        } else {
            Self::default()
        })
    }
    fn save(&self, path: &Path) -> anyhow::Result<()> {
        if let Some(p) = path.parent() {
            fs::create_dir_all(p)?;
        }
        fs::write(path, toml::to_string_pretty(self)?)?;
        Ok(())
    }
}

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

    let (mono, sample_rate) = music::read_wav_mono_f32(wav_path)?;
    let sr_in = sample_rate as f32;
    let duration = mono.len() as f32 / sr_in;

    let (track, sampled_track, tones_track) = music::dominant_frequency_track(&mono, sr_in, win_size, hop_size)?;
    let f_max = sr_in / 2.0;

    // 准备 App 状态
    let file_name = Path::new(wav_path)
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("input.wav")
        .to_string();


    let app = App::new(
        file_name,
        duration as f64,
        f_max as _,
        (sample_rate, mono),
        track
            .iter()
            .map(|(t, f)| [*t as f64, *f as f64])
            .collect(),
        sampled_track
            .iter()
            .map(|(t, freqs)| (*t as f64, [freqs[0] as f64, freqs[1] as f64, freqs[2] as f64]))
            .collect(),
        music::equal_temperament_marks(20.0, f_max)
            .into_iter()
            .map(|(f, name, midi)| (f as f64, name, midi))
            .collect(),
        tones_track,
    )?;

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
            Ok(Box::new(app))
        }),
    )?;

    Ok(())
}


// ========================== GUI 应用 ==========================

#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
enum PlaybackTrack {
    #[default]
    Max,
    Sample1,
    Sample2,
    Sample3,
    Original,
    BeatNotes,
}

impl PlaybackTrack {
    fn label(&self) -> &str {
        match self {
            PlaybackTrack::Max => "最大值（红线）",
            PlaybackTrack::Sample1 => "采样频率 #1",
            PlaybackTrack::Sample2 => "采样频率 #2",
            PlaybackTrack::Sample3 => "采样频率 #3",
            PlaybackTrack::Original => "原音",
            PlaybackTrack::BeatNotes => "节拍音符",
        }
    }
}

struct App {
    file_name: String,
    duration: f64,
    original: (u32, Vec<f32>),
    track: Vec<[f64; 2]>,
    sampled_track: Vec<(f64, [f64; 3])>,
    note_marks: Vec<(f64, String, i32)>,
    tones_track: Vec<(f32, Vec<(String, f64, f32)>)>,
    cached_notes: music::CachedNotes,

    show_pie_chart: bool,
    show_note_lines: bool,
    show_sampled_freqs: bool,
    dense_threshold: usize,
    time_bounds: (f64, f64),
    freq_bounds: (f64, f64),

    bpm: f64,
    show_beat_lines: bool,
    beats_per_bar: usize,
    show_beat_notes: bool,  // 是否显示节拍音符标注

    selected_track: PlaybackTrack,
    stream: Option<OutputStream>,
    handle: Option<OutputStreamHandle>,
    sink: Option<Sink>,
    playing: bool,
    play_start_time: Option<Instant>,
    play_position: f64,
    play_from_start: bool,

    last_time: Option<Instant>,
    last_check_time: Instant,
    frames_since_last_check_time: usize,
    fps_frame_gap: f64,
    fps_frame_count: f64,
    
    config: Config,
}

impl App {
    fn new(
        file_name: String,
        duration: f64,
        f_max: f64,
        original: (u32, Vec<f32>),
        track: Vec<[f64; 2]>,
        sampled_track: Vec<(f64, [f64; 3])>,
        note_marks: Vec<(f64, String, i32)>,
        tones_track: Vec<(f32, Vec<(String, f64, f32)>)>,
    ) -> anyhow::Result<Self> {

        let mut beats_per_bar = 4;
        let mut bpm = 120.0;
        let mut selected_track = PlaybackTrack::Max;

        let config = Config::load(Path::new(CONFIG_PATH)).unwrap_or_default();

        if let Some(track_config) = config.tracks.get(&file_name) {
            if (3..=6).contains(&track_config.beats_per_bar) {
                beats_per_bar = track_config.beats_per_bar;
            }
            if track_config.bpm>= 1.0 {
                bpm = track_config.bpm;
            }
            selected_track = track_config.playback_track;
        }

        Ok(Self {
            file_name,
            duration,
            time_bounds: (0.0, duration.max(1e-6)),
            freq_bounds: (0.0, f_max.max(1.0)),
            cached_notes: music::CachedNotes::analyze_beat_notes(120.0, duration, &track, &tones_track, 4),
            original,
            track,
            sampled_track,
            note_marks,
            tones_track,
            show_pie_chart: false,
            show_note_lines: false, // 默认不显示十二平均律线
            show_sampled_freqs: true,
            dense_threshold: 36,
            bpm,
            show_beat_lines: false, // 默认不显示节拍线
            beats_per_bar,
            show_beat_notes: true,  // 默认显示节拍音符
            selected_track,
            stream: None,
            handle: None,
            sink: None,
            playing: false,
            play_start_time: None,
            play_position: 0.0,
            play_from_start: true,

            last_time: None,
            last_check_time: Instant::now(),
            frames_since_last_check_time: 0,
            fps_frame_gap: 0.0,
            fps_frame_count: 0.0,
            
            config,
        })
    }

    fn get_selected_track_data(&self) -> Option<Vec<(f32, f32)>> {
        match self.selected_track {
            PlaybackTrack::Max => {
                Some(self.track.iter().map(|p| (p[0] as f32, p[1] as f32)).collect())
            }
            PlaybackTrack::Sample1 => {
                Some(self.sampled_track.iter().map(|(t, freqs)| (*t as f32, freqs[0] as f32)).collect())
            }
            PlaybackTrack::Sample2 => {
                Some(self.sampled_track.iter().map(|(t, freqs)| (*t as f32, freqs[1] as f32)).collect())
            }
            PlaybackTrack::Sample3 => {
                Some(self.sampled_track.iter().map(|(t, freqs)| (*t as f32, freqs[2] as f32)).collect())
            }
            _ => None,
        }
    }

    fn start_play(&mut self) {
        if self.playing {
            return;
        }


        let mut sr_out = 44_100u32;
        let beat_duration = 60.0 / self.bpm;

        // 根据选择的轨迹类型生成音频
        let synth = if self.selected_track == PlaybackTrack::BeatNotes {
            // 播放节拍音符
            self.cached_notes.update(self.bpm, self.duration, &self.track, &self.tones_track, self.beats_per_bar);
            music::synth_beat_notes(&*self.cached_notes.track, sr_out, self.duration as f32, 0.3, beat_duration)
        } else if self.selected_track == PlaybackTrack::Original {
            sr_out = self.original.0;
            self.original.1.clone()
        } else {
            // 播放连续频率轨迹
            let track_data = self.get_selected_track_data().unwrap();
            music::synth_sine_from_track(&track_data, sr_out, self.duration as f32, 0.25)
        };

        self.play_position = if !self.play_from_start && self.play_position > 0.0 {
            beat_duration * ((self.play_position / beat_duration).ceil() - 1.0)
        } else {
            0.0
        };

        let skip_samples = if self.play_position > 0.0 {
            ((self.play_position * sr_out as f64).ceil() - 1.0) as _
        } else {
            0
        };

        let mut synth = if skip_samples > 0 { synth.into_iter().skip(skip_samples).collect::<Vec<_>>() } else { synth };


        let fade_length = (sr_out / 100 * 3) as usize;
        music::fade(fade_length, &mut synth);

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
                let buf = SamplesBuffer::new(1, sr_out, synth);
                sink.append(buf);
                sink.set_volume(0.7);
                sink.play();
                self.sink = Some(sink);
                self.playing = true;
                self.play_start_time = Instant::now().checked_sub(Duration::from_millis((self.play_position * 1000.0) as u64));
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
        // self.play_position = 0.0;
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

    fn draw_plot(&mut self, ui: &mut egui::Ui) {
        let plot = Plot::new("dominant_freq_plot")
            .legend(Legend::default())
            .allow_scroll(false)
            .allow_zoom(true)
            .allow_boxed_zoom(true)
            .allow_drag(true)
            .default_x_bounds(self.time_bounds.0, self.time_bounds.1)
            .default_y_bounds(self.freq_bounds.0, self.freq_bounds.1)
            .auto_bounds(Vec2b::new(true, true))
            // .include_x(self.time_bounds.0)
            // .include_x(self.time_bounds.1)
            // .include_y(self.freq_bounds.0)
            // .include_y(self.freq_bounds.1)
            .label_formatter(|name, value| {
                if !name.is_empty() {
                    format!("{name}\n时间: {:.3}s\n频率: {:.1}Hz", value.x, value.y)
                } else {
                    format!("时间: {:.3}s\n频率: {:.1}Hz", value.x, value.y)
                }
            });

        let (ctrl, mouse_click) = ui.input(|i| {
            (i.modifiers.ctrl, i.pointer.any_click())
        });


        self.cached_notes.update(self.bpm, self.duration, &self.track, &self.tones_track, self.beats_per_bar);

        let configuration = plot.show(ui, |plot_ui| {



            let editing_beat_id = &self.cached_notes.configuring; // 显示哪一个小节里的音符备选列表
            let mut set_mouse_cursor = false;
            let mut click_type = if mouse_click { ClickType::Miss } else { ClickType::Nope };

            if ctrl && mouse_click && let Some(pos) = plot_ui.pointer_coordinate() {
                click_type = ClickType::SetPlayPosition { pos: pos.x };
            }

            // 十二平均律水平线
            if self.show_note_lines {
                let dense = self.note_marks.len() > self.dense_threshold;
                let bounds = plot_ui.plot_bounds();
                let x_span = bounds.max()[0] - bounds.min()[0];
                let label_x = bounds.min()[0].max(bounds.max()[0] - 0.01).min(bounds.min()[0] + 0.01 * x_span + 0.01);

                for (f, name, midi) in &self.note_marks {
                    let is_c = *midi % 12 == 0;
                    let is_a4 = *midi == 69;
                    let is_c4 = *midi == 60;
                    let show_label = if dense { is_c || is_a4 || is_c4 } else { true };

                    let line = if is_c4 {
                        HLine::new("", *f).color(Color32::from_rgb(25, 130, 196))
                    } else {
                        HLine::new("", *f).color(Color32::from_rgba_unmultiplied(120, 140, 200, 90))
                    };
                    plot_ui.hline(line);

                    if show_label {
                        let label = if is_c4 {
                            format!("{name} (中央C) {:.1}Hz", f)
                        } else {
                            format!("{name} {:.1}Hz", f)
                        };
                        plot_ui.text(
                            PlotText::new("", PlotPoint {x: label_x.clamp(self.time_bounds.0, self.time_bounds.1), y: f.clamp(self.freq_bounds.0, self.freq_bounds.1) }, label)
                                .color(Color32::from_rgb(70, 70, 110))
                                .anchor(Align2([Align::Min, Align::Center])),
                        );
                    }
                }
            }

            // 绘制三个采样频率
            if self.show_sampled_freqs {
                for i in 0..3 {
                    let points: Vec<[f64; 2]> = self.sampled_track.iter()
                        .map(|(t, freqs)| [*t, freqs[i]])
                        .collect();

                    let (width, color) = match (i, self.selected_track) {
                        (0, PlaybackTrack::Sample1) => (2.5, Color32::from_rgb(200, 100, 255)),
                        (1, PlaybackTrack::Sample2) => (2.5, Color32::from_rgb(200, 100, 255)),
                        (2, PlaybackTrack::Sample3) => (2.5, Color32::from_rgb(200, 100, 255)),
                        _ => (1.5, Color32::from_rgb(147, 51, 234)),
                    };

                    let line = Line::new("show_sampled_freqs", PlotPoints::from_iter(points))
                        .name(format!("采样频率 #{}", i + 1))
                        .color(color)
                        .width(width);
                    plot_ui.line(line);
                }
            }

            // 主频轨迹
            let (max_width, max_color) = if self.selected_track == PlaybackTrack::Max {
                (3.0, Color32::from_rgb(255, 50, 80))
            } else {
                (2.0, Color32::from_rgb(220, 20, 60))
            };

            let line = Line::new("主频轨迹（最大值）", PlotPoints::from_iter(self.track.iter().cloned()))
                .name("主频轨迹（最大值）")
                .color(max_color)
                .width(max_width);
            plot_ui.line(line);


            // 节拍线和音符标注
            if (self.show_beat_lines || self.show_beat_notes) && self.bpm > 0.0 {
                let beat_duration = 60.0 / self.bpm;
                let bounds = plot_ui.plot_bounds();
                let y_span = bounds.max()[1] - bounds.min()[1];


                let start_beat = (bounds.min()[0] / beat_duration).floor() as i32;
                let end_beat = (bounds.max()[0] / beat_duration).ceil() as i32;

                let click_pos = if mouse_click { plot_ui.pointer_coordinate() } else { None };

                for beat_num in start_beat..=end_beat {
                    if beat_num < 0 {
                        continue;
                    }

                    let beat_time = beat_num as f64 * beat_duration;

                    if beat_time > self.duration {
                        break;
                    }

                    if self.show_beat_lines {

                        let is_bar_start = beat_num as usize % self.beats_per_bar == 0;

                        // 绘制节拍线
                        let (color, width) = if is_bar_start {
                            (Color32::from_rgb(0, 100, 200), 2.0)
                        } else {
                            (Color32::from_rgba_unmultiplied(100, 150, 255, 150), 1.0)
                        };

                        let beat_line = VLine::new("节拍线", beat_time)
                            .color(color)
                            .width(width)
                            .name("");
                        plot_ui.vline(beat_line);


                        // 小节编号标签（在底部）
                        if is_bar_start {
                            let bar_num = beat_num as usize / self.beats_per_bar + 1;
                            let label = format!("小节 {}", bar_num);
                            let label_y = bounds.min()[1] + 0.02 * y_span;
                            plot_ui.text(
                                PlotText::new("小节编号", PlotPoint { x: beat_time.clamp(self.time_bounds.0, self.time_bounds.1), y: label_y.clamp(self.freq_bounds.0, self.freq_bounds.1) }, label)
                                    .color(Color32::from_rgb(0, 100, 200))
                                    .anchor(Align2([Align::Min, Align::Min])),
                            );
                        }
                    }




                    // 绘制节拍音符标注框
                    if self.show_beat_notes {
                        if let Some(music::Beat { id, full, note_name, note_freq, is_bar_start: is_strong, configuration, candidates, ..}) = self.cached_notes.track.iter()
                            .find(|music::Beat { beat_start: t, ..}| (*t - beat_time).abs() < beat_duration * 0.1) {

                            // 矩形位置：在图表顶部
                            let rect_y_center = bounds.max()[1] - 0.05 * y_span;
                            let rect_height = 0.08 * y_span;
                            let rect_width = beat_duration * 0.8;

                            let rect = plot_ui.transform().rect_from_values(&PlotPoint {x: 0.0, y: 0.0}, &PlotPoint {x: rect_width, y: rect_height});

                            // 如果不够空间画音调就放弃不画了

                            if rect.right() - rect.left() > FontId::default().size * 3.0 {

                                // 矩形的四个角
                                let rect_x_min = beat_time;
                                let rect_x_max = beat_time + rect_width;
                                let rect_y_min = rect_y_center - rect_height / 2.0;
                                let rect_y_max = rect_y_center + rect_height / 2.0;


                                let rect_x_min = rect_x_min.clamp(self.time_bounds.0, self.time_bounds.1);
                                let rect_x_max = rect_x_max.clamp(self.time_bounds.0, self.time_bounds.1);
                                let rect_y_min = rect_y_min.clamp(self.freq_bounds.0, self.freq_bounds.1);
                                let rect_y_max = rect_y_max.clamp(self.freq_bounds.0, self.freq_bounds.1);

                                if let Some(PlotPoint { x, y }) = click_pos {
                                    if x >= rect_x_min && x <= rect_x_max && y >= rect_y_min && y <= rect_y_max && editing_beat_id.is_none() {
                                        click_type = ClickType::EditNote {id: *id};
                                    }
                                }


                                // 绘制矩形边框（四条线）
                                let border_color = if *is_strong {
                                    Color32::from_rgb(40, 80, 160)
                                } else {
                                    Color32::from_rgb(60, 100, 180)
                                };

                                let rectangle = Polygon::new("音符名称", PlotPoints::from(vec![
                                    [rect_x_min, rect_y_min],
                                    [rect_x_min, rect_y_max],
                                    [rect_x_max, rect_y_max],
                                    [rect_x_max, rect_y_min],

                                ])).fill_color(border_color).stroke(Stroke::new(0.0, border_color)).name("");
                                plot_ui.polygon(rectangle);


                                // 在矩形中心标注音符名称
                                let label = if let Some((name, freq)) = configuration {
                                    let points = Points::new("", PlotPoints::from([rect_x_min, rect_y_max])).
                                        color(Color32::GREEN).shape(Circle).radius(5.0);
                                    plot_ui.points(points);
                                    format!("{name}\n{freq:.1}Hz")
                                } else {
                                    format!("{note_name}\n{note_freq:.1}Hz")
                                };
                                plot_ui.text(
                                    PlotText::new("音符名称",
                                                  PlotPoint { x: beat_time + rect_width / 2.0, y: rect_y_center.clamp(self.freq_bounds.0, self.freq_bounds.1) },
                                                  label
                                    )
                                        .color(Color32::WHITE)
                                        .anchor(Align2::CENTER_CENTER)
                                        .name(""),

                                );

                                let cancel_button_radius = 10.0;

                                if let Some(PlotPoint {x, y}) = plot_ui.pointer_coordinate() {
                                    if (x - rect_x_max).abs() < cancel_button_radius && (y - rect_y_max).abs() < cancel_button_radius {
                                        set_mouse_cursor = true;

                                    }

                                }

                                plot_ui.points(Points::new("消去按钮", [rect_x_max, rect_y_max]).color(Color32::RED).radius(cancel_button_radius as f32).shape(Cross),);

                                if *editing_beat_id == Some(*id) {
                                    let mut y_offset = -rect_height;
                                    let rect_y_min = rect_y_min + rect_height * 0.75;
                                    for (name, freq, _) in candidates {
                                        y_offset -= rect_height * 0.3;
                                        if let Some(PlotPoint { x, y }) = click_pos {
                                            if x >= rect_x_min && x <= rect_x_max && y >= rect_y_min + y_offset && y <= rect_y_max + y_offset {
                                                click_type = ClickType::ChooseNote {
                                                    id: *id,
                                                    name: name.clone(),
                                                    f: *freq,
                                                };
                                            }
                                        }
                                        let rectangle = Polygon::new("音符备选框", PlotPoints::from(vec![
                                            [rect_x_min, rect_y_min + y_offset],
                                            [rect_x_min, rect_y_max + y_offset],
                                            [rect_x_max, rect_y_max + y_offset],
                                            [rect_x_max, rect_y_min + y_offset],

                                        ])).fill_color(Color32::from_rgb(255, 0, 0)).stroke(Stroke::new(0.0, border_color));
                                        plot_ui.polygon(rectangle.name(""));

                                        plot_ui.text(
                                            PlotText::new("音符备选文字",
                                                          PlotPoint { x: beat_time + rect_width / 2.0, y: y_offset + rect_y_max },
                                                          name
                                            )
                                                .color(Color32::WHITE)
                                                .anchor(Align2::CENTER_TOP)
                                                .name(""),
                                        );
                                    }
                                }

                                let rect_x_min = rect_x_max;
                                let rect_x_max = rect_x_min + beat_duration * 0.2;
                                let rect_y_min = rect_y_center - rect_height / 4.0;
                                let rect_y_max = rect_y_center + rect_height / 4.0;

                                if let Some(PlotPoint { x, y }) = click_pos {
                                    if x > rect_x_min && x < rect_x_max && y > rect_y_min && y < rect_y_max {
                                        click_type = ClickType::SetNoteFull {
                                            id: *id,
                                            full: !*full,
                                        }
                                    }
                                }

                                if *full {
                                    let rectangle = Polygon::new("音符连线（唱满一个节拍）", PlotPoints::from(vec![
                                        [rect_x_min, rect_y_min],
                                        [rect_x_min, rect_y_max],
                                        [rect_x_max, rect_y_max],
                                        [rect_x_max, rect_y_min],

                                    ])).fill_color(Color32::YELLOW).stroke(Stroke::new(0.0, border_color));
                                    plot_ui.polygon(rectangle.name(""));
                                }
                            }

                        }
                    }

                }

            }

            // 播放位置竖线
            if self.play_position > 0.0 {
                let play_line = VLine::new("播放位置", self.play_position)
                    .color(Color32::from_rgba_unmultiplied(0, 255, 0, 200))
                    .width(2.0);
                plot_ui.vline(play_line);
            }

            // 鼠标坐标提示
            if let Some(pointer) = plot_ui.pointer_coordinate() {
                let (name, f_note) = music::nearest_note(pointer.y);
                let txt = format!("最近音: {name} ≈ {:.1}Hz", f_note);
                plot_ui.text(
                    PlotText::new("鼠标坐标提示", PlotPoint {x: pointer.x.clamp(self.time_bounds.0, self.time_bounds.1), y: pointer.y.clamp(self.freq_bounds.0, self.freq_bounds.1)}, txt)
                        .anchor(Align2([Align::Min, Align::Max]))
                        .color(Color32::from_rgb(250, 50, 50)),
                );
            }
            Configuration {
                click_type,
                set_mouse_cursor,
            }
        });

        struct Configuration {
            click_type: ClickType,
            set_mouse_cursor: bool,
        }

        enum ClickType {
            Nope,
            Miss,
            EditNote {
                id: usize,
            },
            ChooseNote {
                id: usize,
                name: String,
                f: f64,
            },
            SetPlayPosition {
                pos: f64,
            },
            SetNoteFull {
                id: usize,
                full: bool,
            },
        }

        if configuration.inner.set_mouse_cursor {
            ui.ctx().output_mut(|output| {
                output.cursor_icon = CursorIcon::PointingHand;
            });
        }

        match configuration.inner.click_type {
            ClickType::EditNote {id } => {
                self.cached_notes.configuring = Some(id);
            }
            ClickType::Miss => {
                self.cached_notes.configuring = None;
            }
            ClickType::ChooseNote { id, name, f } => {
                for b in &mut self.cached_notes.track {
                    if b.id == id {
                        b.configuration = Some((name.clone(), f));
                    }
                }
            }
            ClickType::SetPlayPosition { pos } => {
                self.play_position = pos;
            }
            ClickType::SetNoteFull { id, full } => {
                for b in &mut self.cached_notes.track {
                    if b.id == id {
                        b.full = full;
                    }
                }
            }
            ClickType::Nope => {}
        }

    }
}



fn draw_pie_chart(painter: &Painter,
                   data: &[(String, f32, Color32)],
                   center: Pos2,
                   radius: f32) {
    let monospace_font = FontId::monospace(10.0);

    let total: f32 = data.iter().map(|(_, x, _)| *x).sum();

    let line_height = FontId::default().size;

    let center = center + vec2(0.0, line_height);

    let mut start_angle = -PI / 2.0;

    let mut label_height_offset = line_height;

    let mut cases_square_end = None;

    for (name, value, color) in data {
        let sweep = value / total * 2.0f32 * PI;

        let mut points = vec![center];
        for i in 0..=200 {
            let angle: f32 = start_angle + sweep * (i as f32 / 200.0);
            let point = center + radius * Vec2::new(angle.cos(), angle.sin());
            points.push(point);
        }
        points.push(center);  // 闭合路径

        painter.add(PathShape::convex_polygon(
            points,
            *color,
            Stroke::new(1.0, Color32::BLACK),
        ));


        if value / total >= 0.1 {
            let label_angle = start_angle + sweep / 2.0;
            let mut label_pos = center + (radius * 1.3) * Vec2::new(label_angle.cos(), label_angle.sin());
            label_pos.y = label_pos.y.clamp(center.y - radius - line_height, center.y + radius + line_height);
            painter.text(label_pos, Align2::CENTER_CENTER, name, monospace_font.clone(), Color32::BLACK);

        } else if value / total >= 0.01 {

            painter.add(LineSegment {points: [pos2(center.x - radius, center.y + (radius * 1.3) + label_height_offset), pos2(center.x, center.y + (radius * 1.3) + label_height_offset)], stroke: Stroke::new(10.0, *color)});
            painter.text(pos2(center.x + line_height, center.y + (radius * 1.3) + label_height_offset), Align2::LEFT_CENTER, name, monospace_font.clone(), Color32::BLACK);
            label_height_offset += line_height;

            cases_square_end = Some(label_height_offset)
        }

        start_angle += sweep;
    }

    if let Some(square_end) = cases_square_end {
        let rect = Rect { min: pos2(center.x - radius - line_height, center.y + (radius * 1.3)), max: pos2(center.x + radius + line_height, center.y + (radius * 1.3) + square_end) };
        painter.add(RectShape::new(rect, CornerRadius::ZERO, Color32::TRANSPARENT, Stroke::new(1.0, Color32::BLACK), StrokeKind::Middle));

    }

}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {

        let elapsed_last_check = self.last_check_time.elapsed().as_secs_f64();

        if elapsed_last_check > 2.0 {
            if let Some(last_time) = &self.last_time {
                let elapsed_last_frame = last_time.elapsed().as_secs_f64();
                self.fps_frame_gap = 1.0 / elapsed_last_frame;
            }
            self.fps_frame_count = self.frames_since_last_check_time as f64 / elapsed_last_check;
            self.frames_since_last_check_time = 0;

            self.last_check_time = Instant::now();
        }

        self.last_time = Some(Instant::now());
        self.frames_since_last_check_time += 1;


        self.update_play_position();

        if self.playing {
            ctx.request_repaint();
        }

        egui::TopBottomPanel::top("top").show(ctx, |ui| {
            ui.horizontal_wrapped(|ui| {
                ui.checkbox(&mut self.show_pie_chart, "显示频率图");
                ui.separator();
                ui.checkbox(&mut self.show_note_lines, "显示十二平均律标线");
                ui.separator();
                ui.checkbox(&mut self.show_sampled_freqs, "显示采样频率");
                ui.separator();

                // 节拍控制
                ui.checkbox(&mut self.show_beat_lines, "显示节拍线");
                ui.separator();
                ui.checkbox(&mut self.show_beat_notes, "显示节拍音符");
                ui.separator();
                ui.checkbox(&mut self.play_from_start, "每次从头播放");
                let bpm = self.bpm;
                let beats_per_bar = self.beats_per_bar;
                if self.show_beat_lines || self.show_beat_notes {
                    ui.separator();
                    ui.label("BPM:");
                    ui.add(egui::DragValue::new(&mut self.bpm)
                        .speed(1.0)
                        .range(30.0..=300.0));

                    ui.separator();
                    ui.label("拍号:");
                    egui::ComboBox::from_id_salt("beats_per_bar")
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
                egui::ComboBox::from_id_salt("track_selector")
                    .selected_text(self.selected_track.label())
                    .show_ui(ui, |ui| {
                        ui.selectable_value(&mut self.selected_track, PlaybackTrack::Max, PlaybackTrack::Max.label());
                        ui.selectable_value(&mut self.selected_track, PlaybackTrack::Sample1, PlaybackTrack::Sample1.label());
                        ui.selectable_value(&mut self.selected_track, PlaybackTrack::Sample2, PlaybackTrack::Sample2.label());
                        ui.selectable_value(&mut self.selected_track, PlaybackTrack::Sample3, PlaybackTrack::Sample3.label());
                        ui.selectable_value(&mut self.selected_track, PlaybackTrack::Original, PlaybackTrack::Original.label());
                        ui.selectable_value(&mut self.selected_track, PlaybackTrack::BeatNotes, PlaybackTrack::BeatNotes.label());  // 新增
                    });

                if self.playing && prev_selection != self.selected_track || bpm != self.bpm || beats_per_bar != self.beats_per_bar {
                    self.config.tracks.insert(self.file_name.clone(), TrackCfg {
                        bpm: self.bpm,
                        beats_per_bar: self.beats_per_bar,
                        playback_track: self.selected_track,
                    });
                    self.config.save(&Path::new(CONFIG_PATH)).unwrap();
                }

                // 如果正在播放时切换轨迹，先停止播放
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
            });

            ui.horizontal_wrapped(|ui| {
                ui.label(RichText::new(format!("文件: {}", self.file_name)).strong());
                ui.separator();
                ui.label(format!("时长: {:.3}s", self.duration));

                ui.separator();
                ui.label(format!("fps_frame_gap {:.1}Hz", self.fps_frame_gap));
                ui.separator();
                ui.label(format!("fps_frame_count {:.1}Hz", self.fps_frame_count));

                if self.playing {
                    ui.separator();
                    ui.label(RichText::new(format!("▶ 播放中: {:.2}s / {:.2}s ({})",
                                                   self.play_position, self.duration, self.selected_track.label()))
                        .color(Color32::from_rgb(0, 200, 0)));
                }
            });
        });

        if self.show_pie_chart {
            Window::new("频率图").fade_in(true).collapsible(false).scroll([true, true]).show(ctx, |ui| {
                let mut data = Vec::new();
                let mut iter = color_gemini::ColorIterator::default();
                if let Some(music::Beat {candidates, ..}) = self.cached_notes.track.iter().find(|music::Beat {beat_start, ..}| self.play_position <= *beat_start) {
                    let candidates = candidates.iter().filter(|(_, _, w)| *w > 0.0001).collect::<Vec<_>>();
                    // ui.label(format!("ratio{}\n{candidates:?}", candidates[0].2 / candidates[1].2));
                    for (name, freq, weight) in candidates {
                        let (r, g, b) = iter.next().unwrap();
                        let color = Color32::from_rgb(r, g, b);
                        data.push((format!("{name:<5} {freq:>6.1}Hz"), *weight, color));
                    }
                    let painter = ui.painter();
                    let cursor = ui.cursor();
                    let half_length = ui.available_height().min(ui.available_width()) * 0.5;
                    let y_diff = if ui.available_height() < ui.available_width() {
                        0.0
                    } else {
                        ui.available_height() - ui.available_width()
                    };
                    let radius = half_length * 0.6;

                    draw_pie_chart(painter, &data, cursor.left_top() + vec2(half_length, half_length - y_diff * 0.5), radius);
                }
            });
        }

        egui::CentralPanel::default().show(ctx, |ui| {
            self.draw_plot(ui);
        });
    }
}