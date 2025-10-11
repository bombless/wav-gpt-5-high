use hound::{SampleFormat, WavReader};
use plotters::prelude::*;
use rustfft::{num_complex::Complex, FftPlanner};
use std::{env, error::Error, f32::consts::PI, path::Path};

fn main() -> Result<(), Box<dyn Error>> {
    // 命令行参数：wav_path [win_size] [hop_size] [out_png]
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!(
            "用法: {} <input.wav> [win_size=2048] [hop_size=512] [out=dominant_freq.png]",
            args.get(0).map(|s| s.as_str()).unwrap_or("prog")
        );
        std::process::exit(1);
    }
    let wav_path = &args[1];
    let win_size: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(2048);
    let hop_size: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(512);
    let out_path = args
        .get(4)
        .cloned()
        .unwrap_or_else(|| "dominant_freq.png".to_string());

    // 读取 WAV 并混合为单声道 f32
    let (mono, sample_rate) = read_wav_mono_f32(wav_path)?;
    let sr_in = sample_rate as f32;
    let duration = mono.len() as f32 / sr_in;

    if win_size < 8 || hop_size == 0 || hop_size > win_size {
        return Err("win_size 必须 >= 8，且 0 < hop_size <= win_size".into());
    }

    // 计算主频轨迹
    let (track, global_peak) = dominant_frequency_track(&mono, sr_in, win_size, hop_size)?;

    println!(
        "总时长: {:.3}s, 输入采样率: {} Hz, 窗口/步长: {}/{}",
        duration, sample_rate, win_size, hop_size
    );
    if let Some((t_peak, f_peak, mag)) = global_peak {
        println!(
            "全局最大峰值 -> 时间: {:.3}s, 频率: {:.1} Hz, 幅度(平方): {:.3}",
            t_peak, f_peak, mag
        );
    }

    // 绘图：时间-频率折线，标注全局最大峰
    draw_track_plot(
        &track,
        global_peak,
        sr_in / 2.0,
        duration,
        &out_path,
        Path::new(wav_path)
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("input.wav"),
    )?;
    println!("输出图片: {}", out_path);

    // 以 44_100 Hz 合成跟随主频的正弦波，并播放
    let sr_out: u32 = 44_100;
    let amp = 0.25; // 合成音量，避免削波
    let synth = synth_sine_from_track(&track, sr_out, duration, amp);

    if synth.is_empty() {
        eprintln!("没有生成可播放的合成数据（主频轨迹为空）。");
        return Ok(());
    }

    play_audio_blocking(&synth, sr_out)?;
    println!("播放完成。");
    Ok(())
}

fn read_wav_mono_f32(path: &str) -> Result<(Vec<f32>, u32), Box<dyn Error>> {
    let mut reader = WavReader::open(path)?;
    let spec = reader.spec();
    let sr = spec.sample_rate;
    let ch = spec.channels as usize;

    let mono: Vec<f32> = match (spec.sample_format, spec.bits_per_sample) {
        (SampleFormat::Int, 16) => {
            let samples: Result<Vec<i16>, _> = reader.samples::<i16>().collect();
            let samples = samples?;
            mixdown_to_mono_i16(&samples, ch)
        }
        (SampleFormat::Float, 32) => {
            let samples: Result<Vec<f32>, _> = reader.samples::<f32>().collect();
            let samples = samples?;
            mixdown_to_mono_f32(&samples, ch)
        }
        _ => {
            return Err("仅支持 16-bit PCM 或 32-bit float 的 WAV".into());
        }
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
    let mut global_peak: Option<(f32, f32, f32)> = None; // (time, freq, mag2)
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

fn draw_track_plot(
    track: &[(f32, f32)],
    global_peak: Option<(f32, f32, f32)>,
    fmax: f32,
    tmax: f32,
    out_path: &str,
    title_name: &str,
) -> Result<(), Box<dyn Error>> {
    let root = BitMapBackend::new(out_path, (1280, 720)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption(
            format!("主频轨迹 | {}", title_name),
            ("sans-serif", 28).into_font(),
        )
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(0f32..tmax.max(1e-6), 0f32..fmax.max(1.0))?;

    chart
        .configure_mesh()
        .x_desc("时间 (秒)")
        .y_desc("频率 (Hz)")
        .light_line_style(&RGBColor(220, 220, 220))
        .draw()?;

    chart
        .draw_series(LineSeries::new(
            track.iter().cloned(),
            &RGBColor(220, 20, 60),
        ))?
        .label("主频")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RGBColor(220, 20, 60)));

    if let Some((t_peak, f_peak, _)) = global_peak {
        chart.draw_series(std::iter::once(Circle::new(
            (t_peak, f_peak),
            5,
            RGBColor(25, 130, 196).filled(),
        )))?;
        chart.draw_series(std::iter::once(Text::new(
            format!("peak: {:.2}s, {:.1}Hz", t_peak, f_peak),
            (t_peak, f_peak),
            ("sans-serif", 16).into_font().color(&RGBColor(25, 130, 196)),
        )))?;
    }

    chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()?;

    root.present()?;
    Ok(())
}

// === 合成与播放 ===

fn synth_sine_from_track(
    track: &[(f32, f32)],
    sr_out: u32,
    duration: f32,
    amp: f32,
) -> Vec<f32> {
    if track.is_empty() || duration <= 0.0 {
        return vec![];
    }

    // 若轨迹时间不从 0 开始，这里假设从 0 补齐
    let mut t0 = track[0].0;
    let mut local_track = track.to_vec();
    if t0 > 0.0 {
        // 在起点补上一点
        local_track.insert(0, (0.0, track[0].1));
        t0 = 0.0;
    }

    let n = (duration * sr_out as f32).round() as usize;
    let sr_out_f = sr_out as f32;
    let nyq_out = sr_out_f / 2.0;

    let mut y = Vec::with_capacity(n);
    let mut phase = 0.0f32;
    let mut k = 0usize;

    for i in 0..n {
        let t = i as f32 / sr_out_f;

        // 移动段索引，使 t 位于 [t_k, t_{k+1}] 或最后一段
        while k + 1 < local_track.len() && t > local_track[k + 1].0 {
            k += 1;
        }

        // 线性插值频率，确保频率随时间平滑变化
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
            .clamp(0.0, nyq_out); // 防止超出输出奈奎斯特导致混叠

        // 相位积分（保持相位连续，避免点击）
        phase += 2.0 * PI * f_inst / sr_out_f;
        y.push(amp * phase.sin());
    }

    // 简单淡入淡出，避免头尾的小点击（各 20ms）
    let fade = (0.02 * sr_out_f) as usize;
    for i in 0..fade.min(y.len()) {
        let g = i as f32 / fade as f32;
        y[i] *= g;
        let j = y.len() - 1 - i;
        y[j] *= g;
    }

    y
}

fn play_audio_blocking(samples: &[f32], sample_rate: u32) -> Result<(), Box<dyn Error>> {
    use rodio::{buffer::SamplesBuffer, OutputStream, Sink};

    let (_stream, handle) = OutputStream::try_default()?;
    let sink = Sink::try_new(&handle)?;
    // 单声道播放；如果需要立体声，可复制到两个声道
    let buf = SamplesBuffer::new(1, sample_rate, samples.to_vec());
    sink.append(buf);
    sink.set_volume(0.9);
    sink.sleep_until_end(); // 阻塞直到播放完毕
    Ok(())
}
