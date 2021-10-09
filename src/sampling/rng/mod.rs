#![allow(dead_code)]

use std::fmt::Display;

mod pcg32;
pub use pcg32::Pcg32;

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Rng {
    Pcg32(Pcg32),
    Default,
}

pub fn rng_pcg32() -> Rng {
    Rng::Pcg32(Pcg32::default())
}

impl Display for Rng {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Rng::Pcg32(rng) => write!(f, "Rng(rng: {})", rng),
            Rng::Default => write!(f, "Rng(default)"),
        }
    }
}

impl Default for Rng {
    fn default() -> Self {
        Self::Default
    }
}

impl Rng {
    pub fn next_u32(&mut self) -> u32 {
        match self {
            Rng::Pcg32(rng) => rng.next_u32(),
            Rng::Default => panic!("`rmath::sampling::Rng::next_u32`: empty rng."),
        }
    }

    pub fn next_u64(&mut self) -> u64 {
        match self {
            Rng::Pcg32(rng) => rng.next_u64(),
            Rng::Default => panic!("`rmath::sampling::Rng::next_u64`: empty rng."),
        }
    }

    pub fn next_f32(&mut self) -> f32 {
        match self {
            Rng::Pcg32(rng) => rng.next_f32(),
            Rng::Default => panic!("`rmath::sampling::Rng::next_f32`: empty rng."),
        }
    }

    pub fn next_f64(&mut self) -> f64 {
        match self {
            Rng::Pcg32(rng) => rng.next_f64(),
            Rng::Default => panic!("`rmath::sampling::Rng::next_f64`: empty rng."),
        }
    }

    pub fn advance(&mut self, steps: u64) {
        match self {
            Rng::Pcg32(rng) => rng.advance(steps),
            Rng::Default => panic!("`rmath::sampling::Rng::advance`: empty rng."),
        }
    }
}
