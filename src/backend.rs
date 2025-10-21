// src/backend.rs
pub trait MatMul {
    /// C[m x n] = A[m x k] * B[k x n]
    /// All buffers are row-major, f32 for now (we'll add f16/i8 later).
    fn matmul(&mut self, m: usize, n: usize, k: usize,
              a: &[f32], b: &[f32], c: &mut [f32]) -> anyhow::Result<()>;
}

pub trait Softmax {
    /// In-place softmax over last dim (per row).
    fn softmax_rows(&mut self, rows: usize, cols: usize, x: &mut [f32]) -> anyhow::Result<()>;
}

pub enum Backend {
    Cpu(CpuBackend),
    Vk(VkBackend),
}

impl MatMul for Backend {
    fn matmul(&mut self, m: usize, n: usize, k: usize, a: &[f32], b: &[f32], c: &mut [f32]) -> anyhow::Result<()> {
        match self {
            Backend::Cpu(bk) => bk.matmul(m, n, k, a, b, c),
            Backend::Vk(bk)  => bk.matmul(m, n, k, a, b, c),
        }
    }
}

impl Softmax for Backend {
    fn softmax_rows(&mut self, rows: usize, cols: usize, x: &mut [f32]) -> anyhow::Result<()> {
        match self {
            Backend::Cpu(bk) => bk.softmax_rows(rows, cols, x),
            Backend::Vk(bk)  => bk.softmax_rows(rows, cols, x), // (CPU fallback at first)
        }
    }
}
