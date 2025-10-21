// src/attn.rs
use anyhow::Result;
use crate::backend::{MatMul, Softmax, Backend};

pub struct Attention {
    pub backend: Backend,
    pub d: usize, // head dim
}

impl Attention {
    // q,k,v: [seq, d], output: [seq, d]
    pub fn forward(&mut self, seq: usize, q: &[f32], k: &[f32], v: &[f32], out: &mut [f32]) -> Result<()> {
        let d = self.d;
        // scores = q * k^T → [seq, seq]
        let mut scores = vec![0.0f32; seq*seq];
        // Reuse buffers B as transposed K without copying: by building a transposed view
        // For simplicity here, materialize K^T once (we'll add an in-shader transpose kernel later).
        let mut kt = vec![0.0f32; d*seq];
        for i in 0..seq {
            for j in 0..d { kt[j*seq + i] = k[i*d + j]; }
        }
        self.backend.matmul(seq, seq, d, q, &kt, &mut scores)?;
        // scale by 1/sqrt(d)
        let scale = 1.0f32 / (d as f32).sqrt();
        for s in &mut scores { *s *= scale; }

        // softmax per row (CPU for now; numerically stable)
        for i in 0..seq {
            let row = &mut scores[i*seq..(i+1)*seq];
            let maxv = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0;
            for x in row.iter_mut() { *x = (*x - maxv).exp(); sum += *x; }
            let inv = 1.0/sum;
            for x in row.iter_mut() { *x *= inv; }
        }

        // out = softmax(scores) * V → [seq, d]
        self.backend.matmul(seq, d, seq, &scores, v, out)?;
        Ok(())
    }
}
