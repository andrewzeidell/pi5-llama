mod backend;
mod cpu;
mod vk;
//mod attn; // optional

use backend::{Backend, MatMul};
use cpu::CpuBackend;
use vk::VkBackend;

fn main() -> anyhow::Result<()> {
    // Example test run
    let m = 32;
    let n = 32;
    let k = 32;
    let a = vec![1.0f32; m*k];
    let b = vec![1.0f32; k*n];
    let mut c_cpu = vec![0.0f32; m*n];
    let mut c_gpu = vec![0.0f32; m*n];

    let mut cpu = Backend::Cpu(CpuBackend::new());
    cpu.matmul(m, n, k, &a, &b, &mut c_cpu)?;

    let mut vk = Backend::Vk(VkBackend::new()?);
    vk.matmul(m, n, k, &a, &b, &mut c_gpu)?;

    let diff = c_cpu.iter().zip(&c_gpu)
        .map(|(x,y)| (x - y).abs()).fold(0.0, f32::max);
    println!("Max abs diff: {:.6}", diff);
    Ok(())
}
