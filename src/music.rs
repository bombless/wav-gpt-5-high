
// ========================== 数据处理 ==========================

use std::collections::{HashMap, HashSet};
use std::error::Error;
use std::f32::consts::PI;
use hound::{SampleFormat, WavReader};
use rustfft::FftPlanner;
use rustfft::num_complex::Complex;

pub(crate) fn read_wav_mono_f32(path: &str) -> Result<(Vec<f32>, u32), Box<dyn Error>> {
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

pub(crate) fn mixdown_to_mono_i16(samples: &[i16], channels: usize) -> Vec<f32> {
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

pub(crate) fn mixdown_to_mono_f32(samples: &[f32], channels: usize) -> Vec<f32> {
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

pub(crate) fn dominant_frequency_track(
    mono: &[f32],
    sr: f32,
    win_size: usize,
    hop_size: usize,
) -> Result<(Vec<(f32, f32)>, Vec<(f32, [f32; 3])>, Vec<(f32, Vec<(String, f64, f32)>)>), Box<dyn Error>> {
    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(win_size);
    let hann: Vec<f32> = (0..win_size)
        .map(|n| 0.5 * (1.0 - (2.0 * PI * n as f32 / (win_size as f32 - 1.0)).cos()))
        .collect();

    let mut track = Vec::<(f32, f32)>::new();
    let mut sampled_track = Vec::<(f32, [f32; 3])>::new();
    let mut tones_track = Vec::new();

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

        let mut peaks: Vec<(usize, f32)> = (0..half)
            .map(|k| {
                let c = buf[k];
                let mag2 = c.re * c.re + c.im * c.im;
                (k, mag2)
            })
            .collect();

        peaks.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

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

        let mut top9_tones = Vec::new();
        let mut tones = HashSet::new();
        for p in peaks {
            let (name, freq) = nearest_note((p.0 as f32 * sr / win_size as f32) as _);
            if !(freq > 0.0) {
                continue;
            }
            if !tones.contains(&name) {
                tones.insert(name.clone());
                top9_tones.push((name, freq, p.1));
                if top9_tones.len() >= 9 {
                    break;
                }
            }
        }

        let t = start as f32 / sr;
        let f_max = top3_freqs[0];

        track.push((t, f_max));
        sampled_track.push((t, top3_freqs));
        tones_track.push((t, top9_tones));

        start += hop_size;
    }

    Ok((track, sampled_track, tones_track))
}

pub(crate) fn equal_temperament_marks(f_min: f32, f_max: f32) -> Vec<(f32, String, i32)> {
    let names = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"];
    let mut v = Vec::new();
    for midi in 0..=127 {
        let f = 440.0 * 2f32.powf((midi as f32 - 69.0) / 12.0);
        if f >= f_min && f <= f_max {
            let pc = (midi % 12) as usize;
            let octave = (midi / 12) - 1;
            let name = format!("{}{}", names[pc], octave);
            v.push((f, name, midi));
        }
    }
    v
}

// ========================== 合成与播放 ==========================
pub(crate) fn synth_beat_notes(
    beat_notes: &[Beat],
    sr_out: u32,
    duration: f32,
    amp: f32,
    beat_duration: f64,
) -> Vec<f32> {
    if beat_notes.is_empty() || duration <= 0.0 {
        return vec![];
    }

    let n = (duration * sr_out as f32).round() as usize;
    let sr_out_f = sr_out as f32;
    let mut y = vec![0.0f32; n];

    for Beat {beat_start: beat_time, note_freq, configuration, is_bar_start, ..} in beat_notes {
        let freq = if let Some((_, f)) = configuration { *f as f32 } else { *note_freq as f32 };
        let start_sample = (*beat_time as f32 * sr_out_f) as usize;

        // 音符持续时间为节拍的90%，留10%间隙
        let note_duration = (beat_duration * 0.9) as f32;
        let end_sample = ((beat_time + note_duration as f64) as f32 * sr_out_f) as usize;

        if start_sample >= n {
            break;
        }

        let note_samples = (end_sample.min(n) - start_sample).max(1);

        // ADSR 包络参数
        let attack = (0.01 * sr_out_f) as usize;  // 10ms 上升
        let decay = (0.05 * sr_out_f) as usize;   // 50ms 衰减
        let sustain_level = if *is_bar_start { 0.8 } else { 0.6 };  // 小节开始音量更大
        let release = (0.1 * sr_out_f) as usize;  // 100ms 释放

        let mut phase = 0.0f32;

        for i in 0..note_samples {
            let sample_idx = start_sample + i;
            if sample_idx >= n {
                break;
            }

            // 计算 ADSR 包络
            let envelope = if i < attack {
                // Attack: 线性上升
                i as f32 / attack as f32
            } else if i < attack + decay {
                // Decay: 从1.0衰减到sustain_level
                1.0 - (1.0 - sustain_level) * ((i - attack) as f32 / decay as f32)
            } else if i < note_samples.saturating_sub(release) {
                // Sustain: 保持恒定
                sustain_level
            } else {
                // Release: 线性下降到0
                sustain_level * ((note_samples - i) as f32 / release as f32)
            };

            // 生成正弦波
            phase += 2.0 * PI * freq / sr_out_f;
            if phase > 2.0 * PI {
                phase -= 2.0 * PI;
            }

            y[sample_idx] += amp * envelope * phase.sin();
        }
    }

    // 整体淡入淡出
    let fade_length = (0.02 * sr_out_f) as usize;
    fade(fade_length, &mut y);

    y
}

fn fade(fade_length: usize, buffer: &mut [f32]) {
    for i in 0..fade_length.min(buffer.len()) {
        let g = i as f32 / fade_length as f32;
        buffer[i] *= g;
        let j = buffer.len() - 1 - i;
        buffer[j] *= g;
    }
}

pub(crate) fn synth_sine_from_track(
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

    let fade_length = (0.02 * sr_out_f) as usize;
    fade(fade_length, &mut y);

    y
}



pub(crate) struct CachedNotes {
    pub(crate) track: Vec<Beat>,
    bpm: f64,
    duration: f64,
    beats_per_bar: usize,
    pub(crate) configuring: Option<usize>,
}


pub(crate) struct Beat {
    pub(crate) id: usize,
    pub(crate) beat_start: f64,
    pub(crate) note_name: String,
    pub(crate) note_freq: f64,
    pub(crate) is_bar_start: bool,
    pub(crate) candidates: Vec<(String, f64, f32)>,
    pub(crate) configuration: Option<(String, f64)>,
}

impl CachedNotes {

    pub(crate) fn update(&mut self, bpm: f64, duration: f64, track: &[[f64; 2]], tones_track: &[(f32, Vec<(String, f64, f32)>)], beats_per_bar: usize) {
        if self.bpm != bpm || self.duration != duration || self.beats_per_bar != beats_per_bar {
            *self = Self::analyze_beat_notes(bpm, duration, track, tones_track, beats_per_bar);
        }
    }


    // 分析每个节拍的主导音符
    pub(crate) fn analyze_beat_notes(bpm: f64, duration: f64, track: &[[f64; 2]], tones_track: &[(f32, Vec<(String, f64, f32)>)], beats_per_bar: usize) -> CachedNotes {
        if bpm <= 0.0 {
            return CachedNotes {
                track: vec![],
                bpm,
                duration,
                beats_per_bar,
                configuring: None,
            };
        }

        let beat_duration = 60.0 / bpm;
        let mut beat_notes = Vec::new();

        let num_beats = (duration / beat_duration).ceil() as usize;



        for beat_idx in 0..num_beats {
            let beat_start = beat_idx as f64 * beat_duration;
            let beat_end = beat_start + beat_duration;

            if beat_start > duration {
                break;
            }

            // 收集这个节拍内的所有频率数据
            let mut freqs_in_beat = Vec::new();

            for point in track {
                if point[0] >= beat_start && point[0] < beat_end && point[1] > 20.0 {
                    freqs_in_beat.push(point[1]);
                }
            }

            let mut tones = HashMap::new();

            for &(t, ref candidates) in tones_track {
                if t >= beat_start as f32 && t < beat_end as f32 {
                    for &(ref note, freq, weight) in candidates {
                        tones.entry(note).or_insert((freq, 0.0)).1 += weight;
                    }
                }
            }

            let mut sort_vec = tones.iter().collect::<Vec<_>>();
            sort_vec.sort_by(|a, b| b.1.1.partial_cmp(&a.1.1).unwrap());
            let candidates = sort_vec.into_iter().take(15).map(|(&k, &(freq, w))| (k.clone(), freq, w)).collect();



            if !freqs_in_beat.is_empty() {
                // 计算中位数频率（比平均值更稳定）
                freqs_in_beat.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let median_freq = freqs_in_beat[freqs_in_beat.len() / 2];

                let (note_name, note_freq) = nearest_note(median_freq);
                let is_bar_start = beat_idx % beats_per_bar == 0;

                beat_notes.push(Beat {id: beat_idx, beat_start, note_name, note_freq, is_bar_start, candidates, configuration: Default::default()});
            }
        }

        CachedNotes {
            track: beat_notes,
            bpm,
            duration,
            beats_per_bar,
            configuring: None,
        }
    }
}


pub(crate) fn nearest_note(freq: f64) -> (String, f64) {
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