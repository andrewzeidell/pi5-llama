// src/backend.rs
use anyhow::Result;

pub trait MatMul {
    fn matmul(&mut self, m: usize, n: usize, k: usize,
              a: &[f32], b: &[f32], c: &mut [f32]) -> Result<()>;
}
pub trait Softmax {
    fn softmax_rows(&mut self, rows: usize, cols: usize, x: &mut [f32]) -> Result<()>;
}

pub struct CpuBackend; // defined in cpu.rs (imported via crate::cpu::CpuBackend)
pub struct VkBackend;  // defined in vk.rs  (imported via crate::vk::VkBackend)

pub enum Backend {
    Cpu(crate::cpu::CpuBackend),
    Vk(crate::vk::VkBackend),
}

impl MatMul for Backend {
    fn matmul(&mut self, m: usize, n: usize, k: usize, a: &[f32], b: &[f32], c: &mut [f32]) -> Result<()> {
        match self {
            Backend::Cpu(bk) => bk.matmul(m, n, k, a, b, c),
            Backend::Vk(bk)  => bk.matmul(m, n, k, a, b, c),
        }
    }
}

impl Softmax for Backend {
    fn softmax_rows(&mut self, rows: usize, cols: usize, x: &mut [f32]) -> Result<()> {
        // CPU fallback for now
        for i in 0..rows {
            let row = &mut x[i*cols..(i+1)*cols];
            let maxv = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0;
            for v in row.iter_mut() { *v = (*v - maxv).exp(); sum += *v; }
            let inv = 1.0 / sum;
            for v in row.iter_mut() { *v *= inv; }
        }
        Ok(())
    }
}
