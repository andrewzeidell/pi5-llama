// src/cpu.rs
use anyhow::Result;
use crate::backend::Softmax;

pub struct CpuBackend;
impl CpuBackend { pub fn new() -> Self { Self } }

impl Softmax for crate::cpu::CpuBackend {
    fn softmax_rows(&mut self, rows: usize, cols: usize, x: &mut [f32]) -> Result<()> {
        for r in 0..rows {
            let start = r * cols;
            let end = start + cols;
            let row = &mut x[start..end];
            let max_v = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0;
            for v in row.iter_mut() {
                *v = (*v - max_v).exp();
                sum += *v;
            }
            for v in row.iter_mut() {
                *v /= sum;
            }
        }
        Ok(())
    }
}
