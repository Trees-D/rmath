#![allow(dead_code)]

use std::fmt::Display;

const MULTIPLIER: u64 = 6364136223846793005;

#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Pcg32 {
    state: u64,
    increment: u64,
}

impl Display for Pcg32 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Rng(state: {}, increment: {})",
            self.state, self.increment
        )
    }
}

impl Default for Pcg32 {
    fn default() -> Self {
        Self::new(0xcafef00dd15ea5e5, 0xa02bdbf7bb3c0a7)
    }
}

impl Pcg32 {
    pub fn new(state: u64, stream: u64) -> Self {
        let increment = (stream << 1) | 1;
        let mut pcg = Self { state, increment };
        pcg.state = pcg.state.wrapping_add(pcg.increment);
        pcg
    }

    fn step(&mut self) {
        self.state = self
            .state
            .wrapping_mul(MULTIPLIER)
            .wrapping_add(self.increment)
    }

    pub fn next_u32(&mut self) -> u32 {
        let state = self.state;
        self.step();
        let rot = (state >> 59) as u32;
        let xsh = (((state >> 18) ^ state) >> 27) as u32;
        xsh.rotate_right(rot)
    }

    pub fn next_u64(&mut self) -> u64 {
        let x = u64::from(self.next_u32());
        let y = u64::from(self.next_u32());
        (y << 32) | x
    }

    pub fn next_f32(&mut self) -> f32 {
        0.99999994f32.min(self.next_u32() as f32 * 2.3283064365386963e-10f32)
    }

    pub fn next_f64(&mut self) -> f64 {
        0.99999999999999989f64.min(self.next_u32() as f64 * 2.3283064365386963e-10f64)
    }

    pub fn advance(&mut self, steps: u64) {
        let mut acc_mult = 1u64;
        let mut acc_plus = 0u64;
        let mut cur_mult = MULTIPLIER;
        let mut cur_plus = self.increment;
        let mut n = steps;

        while n > 0 {
            if (n & 1) != 0 {
                acc_mult = acc_mult.wrapping_mul(cur_mult);
                acc_plus = acc_plus.wrapping_mul(cur_mult).wrapping_add(cur_plus);
            }
            cur_plus = cur_mult.wrapping_add(1).wrapping_mul(cur_plus);
            cur_mult = cur_mult.wrapping_mul(cur_mult);
            n >>= 1;
        }

        self.state = acc_mult.wrapping_mul(self.state).wrapping_add(acc_plus)
    }
}
