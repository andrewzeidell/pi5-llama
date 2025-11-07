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
        match self {
            Backend::Cpu(bk) => bk.softmax_rows(rows, cols, x),
            Backend::Vk(bk)  => bk.softmax_rows(rows, cols, x),
        }
    }
}

pub trait LayerNorm {
    fn layernorm(&mut self, rows: usize, cols: usize, x: &mut [f32]) -> Result<()>;
}
pub trait Gelu {
    fn gelu(&mut self, rows: usize, cols: usize, x: &mut [f32]) -> Result<()>;
}

impl LayerNorm for Backend {
    fn layernorm(&mut self, rows: usize, cols: usize, x: &mut [f32]) -> Result<()> {
        match self {
            Backend::Cpu(_) => anyhow::bail!("CPU backend missing LayerNorm"),
            Backend::Vk(bk) => bk.layernorm(rows, cols, x),
        }
    }
}

impl Gelu for Backend {
    fn gelu(&mut self, rows: usize, cols: usize, x: &mut [f32]) -> Result<()> {
        match self {
            Backend::Cpu(_) => anyhow::bail!("CPU backend missing GELU"),
            Backend::Vk(bk) => bk.gelu(rows, cols, x),
        }
    }
}
impl Backend {
    pub fn attention_fused(
        &mut self,
        m: usize,
        n: usize,
        d: usize,
        dv: usize,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        o: &mut [f32],
    ) -> anyhow::Result<()> {
        match self {
            Backend::Cpu(_) => {
                anyhow::bail!("CPU backend does not implement fused attention yet")
            }
            Backend::Vk(bk) => bk.attention_fused(m, n, d, dv, q, k, v, o),
        }
    }
}
