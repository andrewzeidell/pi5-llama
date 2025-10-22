use std::env;
use std::time::Instant;

mod backend;
mod cpu;
mod vk;

use anyhow::Result;
use backend::{Backend, Softmax};
use cpu::CpuBackend;
use vk::VkBackend;

fn bench_softmax(rows: usize, cols: usize, iters: usize) -> Result<()> {
    let mut cpu = Backend::Cpu(CpuBackend::new());
    let mut gpu = Backend::Vk(VkBackend::new()?);

    // deterministic but non-trivial input
    let mut x_cpu: Vec<f32> = (0..rows*cols).map(|i| ((i % 101) as f32) * 0.01 - 0.5).collect();
    let mut x_gpu = x_cpu.clone();

    // warmup
    cpu.softmax_rows(rows, cols, &mut x_cpu)?;
    gpu.softmax_rows(rows, cols, &mut x_gpu)?;

    // check correctness
    let diff = x_cpu.iter().zip(&x_gpu).map(|(a,b)| (a-b).abs()).fold(0.0, f32::max);
    println!("Max abs diff after warmup: {:.6}", diff);

    // CPU bench
    let t0 = Instant::now();
    for _ in 0..iters {
        let mut tmp = x_cpu.clone();
        cpu.softmax_rows(rows, cols, &mut tmp)?;
    }
    let cpu_ms = t0.elapsed().as_secs_f64() * 1e3 / (iters as f64);

    // GPU bench
    let t1 = Instant::now();
    for _ in 0..iters {
        let mut tmp = x_gpu.clone();
        gpu.softmax_rows(rows, cols, &mut tmp)?;
    }
    let gpu_ms = t1.elapsed().as_secs_f64() * 1e3 / (iters as f64);

    println!("CPU softmax: {:.3} ms (avg over {} iters)", cpu_ms, iters);
    println!("GPU softmax: {:.3} ms (avg over {} iters)", gpu_ms, iters);
    Ok(())
}

fn cpu_attention_reference(
    m: usize, n: usize, d: usize, dv: usize,
    q: &[f32], k: &[f32], v: &[f32], o_ref: &mut [f32],
) {
    for i in 0..m {
        // compute scores = Q[i]·K^T / sqrt(d)
        let mut scores = vec![0f32; n];
        for j in 0..n {
            let mut s = 0.0;
            for kk in 0..d {
                s += q[i*d + kk] * k[j*d + kk];
            }
            scores[j] = s / (d as f32).sqrt();
        }

        // softmax(scores)
        let maxs = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = scores.iter().map(|x| (x - maxs).exp()).collect();
        let sumexp: f32 = exps.iter().sum();

        // weighted sum over V
        for dv_i in 0..dv {
            let mut acc = 0.0;
            for j in 0..n {
                acc += (exps[j] / sumexp) * v[j*dv + dv_i];
            }
            o_ref[i*dv + dv_i] = acc;
        }
    }
}


fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() >= 2 && args[1] == "softmax" {
        let rows: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(1024);
        let cols: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(512);
        let iters: usize = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(10);
        println!("Running softmax rows={} cols={} iters={}", rows, cols, iters);
        return bench_softmax(rows, cols, iters);
    }
    if args.len() >= 2 && args[1] == "fused" {
        let m = 4usize;    // queries
        let n = 8usize;    // keys/values
        let d = 32usize;   // head dim
        let dv = 32usize;  // value dim
    
        let q: Vec<f32> = (0..m*d).map(|i| (i as f32).sin()*0.1).collect();
        let k: Vec<f32> = (0..n*d).map(|i| (i as f32).cos()*0.1).collect();
        let v: Vec<f32> = (0..n*dv).map(|i| ((i%17) as f32)*0.01).collect();
        let mut o = vec![0.0f32; m*dv];
    
        let mut gpu = Backend::Vk(VkBackend::new()?);
        gpu.attention_fused(m, n, d, dv, &q, &k, &v, &mut o)?;
        println!("O[0..8] = {:?}", &o[..o.len().min(8)]);
        return Ok(());
    }
    if args.len() >= 2 && args[1] == "bench_fused" {
        
        let m = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(128);
        let n = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(128);
        let d = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(64);
        let dv = args.get(5).and_then(|s| s.parse().ok()).unwrap_or(64);
    
        println!("Benchmarking fused attention (m={}, n={}, d={}, dv={})", m, n, d, dv);
    
        // Generate reproducible test data
        let q: Vec<f32> = (0..m*d).map(|i| ((i % 101) as f32) * 0.01 - 0.5).collect();
        let k: Vec<f32> = (0..n*d).map(|i| ((i % 37) as f32) * 0.02 - 0.4).collect();
        let v: Vec<f32> = (0..n*dv).map(|i| ((i % 53) as f32) * 0.015 - 0.3).collect();
        let mut o_cpu = vec![0.0f32; m*dv];
        let mut o_gpu = vec![0.0f32; m*dv];
    
        // ---- CPU reference timing ----
        let t0 = Instant::now();
        cpu_attention_reference(m, n, d, dv, &q, &k, &v, &mut o_cpu);
        let t_cpu = t0.elapsed().as_secs_f64() * 1e3;
    
        // ---- GPU fused timing ----
        // ---- GPU fused timing ----
        let mut backend = backend::Backend::Vk(crate::vk::VkBackend::new()?);
        
        // Initialize persistent fused attention pipeline + buffers
        if let backend::Backend::Vk(ref mut vk_backend) = backend {
            vk_backend.init_attention_fused(m, n, d, dv)?;
        }
        
        // Run multiple GPU passes to show consistent reuse timing
        use std::time::Instant;
        let runs = 3;
        let mut t_gpu_total = 0.0;
        for _ in 0..runs {
            let t1 = Instant::now();
            backend.attention_fused(m, n, d, dv, &q, &k, &v, &mut o_gpu)?;
            t_gpu_total += t1.elapsed().as_secs_f64() * 1e3;
        }
        let t_gpu = t_gpu_total / (runs as f64);
    
        // ---- Compare ----
        let diff = o_cpu.iter().zip(&o_gpu)
            .map(|(a,b)| (a-b).abs())
            .fold(0.0, f32::max);
        println!("Max abs diff: {:.6}", diff);
        println!("CPU fused attention: {:.3} ms", t_cpu);
        println!("GPU fused attention: {:.3} ms", t_gpu);
        println!("O[0..8] = {:?}", &o_gpu[..o_gpu.len().min(8)]);
    
        return Ok(());
    }

    println!("Usage:");
    println!("  cargo run --release -- softmax <rows> <cols> <iters>");
    println!("Example:");
    println!("  cargo run --release -- softmax 2048 512 10");
    Ok(())
}
