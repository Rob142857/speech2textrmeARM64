use pyo3::prelude::*;
use ndarray::Array2;
use realfft::RealFftPlanner;

fn hann(n: usize) -> Vec<f32> { (0..n).map(|i| { let x = (std::f32::consts::PI * 2.0 * i as f32) / (n as f32); 0.5 - 0.5 * x.cos() }).collect() }

fn mel_filter_bank(n_fft: usize, n_mels: usize, sr: usize) -> Array2<f32> {
    // Simplified triangular mel filterbank (HTK-ish)
    let f_min = 0.0;
    let f_max = sr as f32 / 2.0;
    let mel = |f: f32| -> f32 { 2595.0 * (1.0 + f / 700.0).log10() };
    let inv_mel = |m: f32| -> f32 { 700.0 * (10f32.powf(m / 2595.0) - 1.0) };
    let m_min = mel(f_min);
    let m_max = mel(f_max);
    let m_points: Vec<f32> = (0..n_mels + 2).map(|i| m_min + (i as f32) * (m_max - m_min) / ((n_mels + 1) as f32)).collect();
    let f_points: Vec<f32> = m_points.iter().map(|m| inv_mel(*m)).collect();
    let bins: Vec<usize> = f_points.iter().map(|f| ((n_fft/2 +1) as f32 * f / f_max).floor() as usize).collect();
    let mut fb = Array2::<f32>::zeros((n_mels, n_fft/2 +1));
    for m in 1..=n_mels { // triangular filters
        let left = bins[m-1]; let center = bins[m]; let right = bins[m+1];
        for k in left..center { fb[[m-1,k]] = (k - left) as f32 / (center - left).max(1) as f32; }
        for k in center..right { fb[[m-1,k]] = (right - k) as f32 / (right - center).max(1) as f32; }
    }
    fb
}

#[pyfunction]
fn pcm_to_mel<'py>(py: Python<'py>, pcm: Vec<f32>, sample_rate: usize, n_fft: usize, hop_length: usize, n_mels: usize) -> PyResult<&'py PyAny> {
    if sample_rate != 16000 { eprintln!("⚠️ resample externally for best quality (expected 16k)" ); }
    let win = hann(n_fft);
    let mut planner = RealFftPlanner::<f32>::new();
    let r2c = planner.plan_fft_forward(n_fft);
    let mut spectrum = vec![0f32; n_fft];
    let mut scratch = r2c.make_scratch_vec();
    let mut mel_spec: Vec<f32> = Vec::new();
    let fb = mel_filter_bank(n_fft, n_mels, sample_rate);
    let step = hop_length;
    let mut frame = vec![0f32; n_fft];
    let norm = 1.0 / (n_fft as f32);
    let frames = if pcm.len() < n_fft { 0 } else { (pcm.len() - n_fft) / step + 1 };
    for i in 0..frames {
        let start = i*step; let slice = &pcm[start..start+n_fft];
        for j in 0..n_fft { frame[j] = slice[j] * win[j]; }
        // copy real data into spectrum buffer
        for (d, s) in r2c.make_input_vec(&mut spectrum).iter_mut().zip(frame.iter()) { *d = *s; }
        r2c.process_with_scratch(r2c.make_input_vec(&mut spectrum), r2c.make_output_vec(&mut spectrum), &mut scratch).unwrap();
        // magnitude
        let mut mags = vec![0f32; n_fft/2 +1];
        let out = r2c.make_output_vec(&mut spectrum);
        for (k, c) in out.iter().enumerate() { mags[k] = c.norm_sqr() * norm; }
        // apply mel
        for m in 0..n_mels {
            let mut sum = 0f32; for k in 0..(n_fft/2 +1) { sum += fb[[m,k]] * mags[k]; }
            mel_spec.push((sum + 1e-10).ln());
        }
    }
    let frames_out = if n_mels == 0 { 0 } else { mel_spec.len() / n_mels };
    // Convert to numpy array (frames, n_mels)
    let np = py.import("numpy")?;
    let arr = np.call_method1("array", (mel_spec,))?;
    let reshaped = arr.call_method1("reshape", (frames_out, n_mels))?;
    Ok(reshaped)
}

#[pymodule]
fn mel_native(py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(pcm_to_mel, m)?)?;
    // Provide simple version info
    m.add("__version__", "0.1.0")?;
    py.run("import numpy as _np", None, None)?; // ensure numpy import early
    Ok(())
}
