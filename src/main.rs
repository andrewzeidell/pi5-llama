use std::time::Instant;
mod backend;
mod cpu;
mod vk;

use backend::{Backend, MatMul};
use cpu::CpuBackend;
use vk::VkBackend;

fn main() -> anyhow::Result<()> {
    let m = 512;
    let n = 512;
    let k = 512;
    let a = vec![1.0f32; m*k];
    let b = vec![1.0f32; k*n];
    let mut c_cpu = vec![0.0f32; m*n];
    let mut c_gpu = vec![0.0f32; m*n];

    // --- CPU ---
    let mut cpu = Backend::Cpu(CpuBackend::new());
    let t0 = Instant::now();
    cpu.matmul(m, n, k, &a, &b, &mut c_cpu)?;
    let dt_cpu = t0.elapsed().as_secs_f64() * 1e3;

    // --- GPU ---
    let mut vk = Backend::Vk(VkBackend::new()?);
    let t1 = Instant::now();
    vk.matmul(m, n, k, &a, &b, &mut c_gpu)?;
    let dt_gpu = t1.elapsed().as_secs_f64() * 1e3;

    // --- Verify ---
    let diff = c_cpu.iter().zip(&c_gpu)
        .map(|(x,y)| (x - y).abs()).fold(0.0, f32::max);

    // FLOPs ≈ 2*m*n*k
    let gflops = 2.0 * (m as f64) * (n as f64) * (k as f64) / 1e9;
    println!("Size: {m}x{n}x{k}");
    println!("CPU: {:.2} ms ({:.2} GFLOP/s)", dt_cpu, gflops / (dt_cpu / 1e3));
    println!("GPU: {:.2} ms ({:.2} GFLOP/s)", dt_gpu, gflops / (dt_gpu / 1e3));
    println!("Max abs diff: {:.6}", diff);
    Ok(())
}
