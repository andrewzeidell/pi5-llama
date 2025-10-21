// src/cpu.rs
use crate::backend::MatMul;
use anyhow::Result;

pub struct CpuBackend;

impl CpuBackend {
    pub fn new() -> Self { Self }
}

impl MatMul for CpuBackend {
    fn matmul(&mut self, m: usize, n: usize, k: usize,
              a: &[f32], b: &[f32], c: &mut [f32]) -> Result<()> {
        const BM: usize = 64;
        const BN: usize = 64;
        const BK: usize = 64;

        for i0 in (0..m).step_by(BM) {
            for j0 in (0..n).step_by(BN) {
                for p0 in (0..k).step_by(BK) {
                    let imax = (i0 + BM).min(m);
                    let jmax = (j0 + BN).min(n);
                    let pmax = (p0 + BK).min(k);
                    for i in i0..imax {
                        for p in p0..pmax {
                            let a_ip = a[i*k + p];
                            let b_row = &b[p*n..p*n+n];
                            let c_row = &mut c[i*n..i*n+n];
                            // Unrolled inner over j
                            let mut j = j0;
                            while j + 4 <= jmax {
                                c_row[j+0] += a_ip * b_row[j+0];
                                c_row[j+1] += a_ip * b_row[j+1];
                                c_row[j+2] += a_ip * b_row[j+2];
                                c_row[j+3] += a_ip * b_row[j+3];
                                j += 4;
                            }
                            while j < jmax {
                                c_row[j] += a_ip * b_row[j];
                                j += 1;
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }
}
