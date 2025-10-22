use std::time::Instant;
mod backend;
mod cpu;
mod vk;

use backend::{Backend, MatMul, Softmax};
use cpu::CpuBackend;
use vk::VkBackend;

fn main() -> anyhow::Result<()> {
    let rows = 4;
    let cols = 8;
    let mut data = vec![
        1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0,
        5.0, 4.0, 3.0, 2.0, 5.0, 4.0, 3.0, 2.0,
        1.0, 2.0, 5.0, 3.0, 2.0, 2.0, 5.0, 3.0,
        2.0, 3.0, 1.0, 0.0, 2.0, 3.0, 1.0, 0.0,
    ];

    let mut cpu = Backend::Cpu(CpuBackend::new());
    let mut gpu = Backend::Vk(VkBackend::new()?);

    let mut x_cpu = data.clone();
    let mut x_gpu = data.clone();

    let t0 = Instant::now();
    cpu.softmax_rows(rows, cols, &mut x_cpu)?;
    println!("CPU softmax: {:.3} ms", t0.elapsed().as_secs_f64() * 1e3);

    let t1 = Instant::now();
    gpu.softmax_rows(rows, cols, &mut x_gpu)?;
    println!("GPU softmax: {:.3} ms", t1.elapsed().as_secs_f64() * 1e3);

    let diff = x_cpu.iter().zip(&x_gpu).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
    println!("Max abs diff: {:.6}", diff);
    Ok(())
}
