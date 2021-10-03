#![allow(dead_code)]

use std::{
    fmt::Display,
    ops::{Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign},
};

use super::{RGBAf, RGB, RGB24, RGBA, RGBA32};

fn convert_f32_to_u8(v: f32) -> u8 {
    let v = (v * 255.0 + 0.5) as i32;
    v.max(0).min(255) as u8
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub struct RGBf {
    r: f32,
    g: f32,
    b: f32,
}

impl Display for RGBf {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "RGBf(r: {}, g: {}, b: {})", self.r, self.g, self.b)
    }
}

impl Default for RGBf {
    fn default() -> Self {
        Self {
            r: 0.0,
            g: 0.0,
            b: 0.0,
        }
    }
}

impl Add<RGBf> for RGBf {
    type Output = RGBf;

    fn add(self, rhs: RGBf) -> Self::Output {
        RGBf::new(self.r + rhs.r, self.g + rhs.g, self.b + rhs.b)
    }
}

impl Add<f32> for RGBf {
    type Output = RGBf;

    fn add(self, rhs: f32) -> Self::Output {
        RGBf::new(self.r + rhs, self.g + rhs, self.b + rhs)
    }
}

impl Add<RGBf> for f32 {
    type Output = RGBf;

    fn add(self, rhs: RGBf) -> Self::Output {
        RGBf::new(self + rhs.r, self + rhs.g, self + rhs.b)
    }
}

impl AddAssign<RGBf> for RGBf {
    fn add_assign(&mut self, rhs: RGBf) {
        *self = *self + rhs;
    }
}

impl AddAssign<f32> for RGBf {
    fn add_assign(&mut self, rhs: f32) {
        *self = *self + rhs;
    }
}

impl Sub<RGBf> for RGBf {
    type Output = RGBf;

    fn sub(self, rhs: RGBf) -> Self::Output {
        RGBf::new(self.r - rhs.r, self.g - rhs.g, self.b - rhs.b)
    }
}

impl Sub<f32> for RGBf {
    type Output = RGBf;

    fn sub(self, rhs: f32) -> Self::Output {
        RGBf::new(self.r - rhs, self.g - rhs, self.b - rhs)
    }
}

impl Sub<RGBf> for f32 {
    type Output = RGBf;

    fn sub(self, rhs: RGBf) -> Self::Output {
        RGBf::new(self - rhs.r, self - rhs.g, self - rhs.b)
    }
}

impl SubAssign<RGBf> for RGBf {
    fn sub_assign(&mut self, rhs: RGBf) {
        *self = *self - rhs;
    }
}

impl SubAssign<f32> for RGBf {
    fn sub_assign(&mut self, rhs: f32) {
        *self = *self - rhs;
    }
}

impl Mul<RGBf> for RGBf {
    type Output = RGBf;

    fn mul(self, rhs: RGBf) -> Self::Output {
        RGBf::new(self.r * rhs.r, self.g * rhs.g, self.b * rhs.b)
    }
}

impl Mul<f32> for RGBf {
    type Output = RGBf;

    fn mul(self, rhs: f32) -> Self::Output {
        RGBf::new(self.r * rhs, self.g * rhs, self.b * rhs)
    }
}

impl Mul<RGBf> for f32 {
    type Output = RGBf;

    fn mul(self, rhs: RGBf) -> Self::Output {
        RGBf::new(self * rhs.r, self * rhs.g, self * rhs.b)
    }
}

impl MulAssign<RGBf> for RGBf {
    fn mul_assign(&mut self, rhs: RGBf) {
        *self = *self * rhs;
    }
}

impl MulAssign<f32> for RGBf {
    fn mul_assign(&mut self, rhs: f32) {
        *self = *self * rhs;
    }
}

impl Div<RGBf> for RGBf {
    type Output = RGBf;

    fn div(self, rhs: RGBf) -> Self::Output {
        RGBf::new(self.r / rhs.r, self.g / rhs.g, self.b / rhs.b)
    }
}

impl Div<f32> for RGBf {
    type Output = RGBf;

    fn div(self, rhs: f32) -> Self::Output {
        RGBf::new(self.r / rhs, self.g / rhs, self.b / rhs)
    }
}

impl Div<RGBf> for f32 {
    type Output = RGBf;

    fn div(self, rhs: RGBf) -> Self::Output {
        RGBf::new(self / rhs.r, self / rhs.g, self / rhs.b)
    }
}

impl DivAssign<RGBf> for RGBf {
    fn div_assign(&mut self, rhs: RGBf) {
        *self = *self / rhs;
    }
}

impl DivAssign<f32> for RGBf {
    fn div_assign(&mut self, rhs: f32) {
        *self = *self / rhs;
    }
}

impl Neg for RGBf {
    type Output = RGBf;

    fn neg(self) -> Self::Output {
        RGBf::new(-self.r, -self.g, -self.b)
    }
}

impl Index<usize> for RGBf {
    type Output = f32;

    fn index(&self, index: usize) -> &Self::Output {
        match index {
            0 => &self.r,
            1 => &self.g,
            2 => &self.b,
            _ => panic!("`rmath::color::RGBf::index`: index out of bounds."),
        }
    }
}

impl IndexMut<usize> for RGBf {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        match index {
            0 => &mut self.r,
            1 => &mut self.g,
            2 => &mut self.b,
            _ => panic!("`rmath::color::RGBf::index_mut`: index out of bounds."),
        }
    }
}

impl From<f32> for RGBf {
    fn from(rgbf: f32) -> Self {
        Self::new(rgbf, rgbf, rgbf)
    }
}

impl From<(f32, f32, f32)> for RGBf {
    fn from(rgbf: (f32, f32, f32)) -> Self {
        Self::new(rgbf.0, rgbf.1, rgbf.2)
    }
}

impl From<[f32; 3]> for RGBf {
    fn from(rgbf: [f32; 3]) -> Self {
        Self::new(rgbf[0], rgbf[1], rgbf[2])
    }
}

impl From<u32> for RGBf {
    fn from(rgb_: u32) -> Self {
        let r = ((rgb_ >> 16) & 0xff) as f32 / 255.0;
        let g = ((rgb_ >> 8) & 0xff) as f32 / 255.0;
        let b = (rgb_ & 0xff) as f32 / 255.0;
        Self::new(r, g, b)
    }
}

impl From<u8> for RGBf {
    fn from(rgb: u8) -> Self {
        let rgb = rgb as f32 / 255.0;
        Self::new(rgb, rgb, rgb)
    }
}

impl From<(u8, u8, u8)> for RGBf {
    fn from(rgb: (u8, u8, u8)) -> Self {
        let (r, g, b) = rgb;
        let r = r as f32 / 255.0;
        let g = g as f32 / 255.0;
        let b = b as f32 / 255.0;
        Self::new(r, g, b)
    }
}

impl From<[u8; 3]> for RGBf {
    fn from(rgb: [u8; 3]) -> Self {
        let r = rgb[0] as f32 / 255.0;
        let g = rgb[1] as f32 / 255.0;
        let b = rgb[2] as f32 / 255.0;
        Self::new(r, g, b)
    }
}

impl From<RGB> for RGBf {
    fn from(color: RGB) -> Self {
        color.to_rgbf()
    }
}

impl From<RGBA> for RGBf {
    fn from(color: RGBA) -> Self {
        color.to_rgbf()
    }
}

impl From<RGBAf> for RGBf {
    fn from(color: RGBAf) -> Self {
        color.to_rgbf()
    }
}

impl From<RGB24> for RGBf {
    fn from(color: RGB24) -> Self {
        color.to_rgbf()
    }
}

impl From<RGBA32> for RGBf {
    fn from(color: RGBA32) -> Self {
        color.to_rgbf()
    }
}

impl RGBf {
    pub fn new(r: f32, g: f32, b: f32) -> Self {
        Self { r, g, b }
    }

    pub fn black() -> Self {
        Self::new(0.0, 0.0, 0.0)
    }

    pub fn white() -> Self {
        Self::new(1.0, 1.0, 1.0)
    }

    pub fn red() -> Self {
        Self::new(1.0, 0.0, 0.0)
    }

    pub fn green() -> Self {
        Self::new(0.0, 1.0, 0.0)
    }

    pub fn blue() -> Self {
        Self::new(0.0, 0.0, 1.0)
    }

    pub fn r(self) -> f32 {
        self.r
    }

    pub fn g(self) -> f32 {
        self.g
    }

    pub fn b(self) -> f32 {
        self.b
    }
}

impl RGBf {
    pub fn sum(self) -> f32 {
        self.r + self.g + self.b
    }

    pub fn gray(self) -> f32 {
        self.sum() / 3.0
    }

    pub fn luma1(self) -> f32 {
        0.299 * self.r + 0.587 * self.g + 0.144 * self.b
    }

    pub fn luma2(self) -> f32 {
        0.2126 * self.r + 0.7152 * self.g + 0.0722 * self.b
    }

    pub fn min_element(self) -> f32 {
        self.r.min(self.g).min(self.b)
    }

    pub fn max_element(self) -> f32 {
        self.r.max(self.g).max(self.b)
    }

    pub fn is_finite(self) -> bool {
        self.r.is_finite() && self.g.is_finite() && self.b.is_finite()
    }

    pub fn is_nan(self) -> bool {
        self.r.is_nan() || self.g.is_nan() || self.b.is_nan()
    }

    pub fn is_infinite(self) -> bool {
        self.r.is_infinite() || self.g.is_infinite() || self.b.is_infinite()
    }

    pub fn clamp(self, min: Self, max: Self) -> Self {
        ruby_assert!(min.r <= max.r);
        ruby_assert!(min.g <= max.g);
        ruby_assert!(min.b <= max.b);

        self.min(max).max(min)
    }

    pub fn min(self, rhs: Self) -> Self {
        Self::new(self.r.min(rhs.r), self.g.min(rhs.g), self.b.min(rhs.b))
    }

    pub fn max(self, rhs: Self) -> Self {
        Self::new(self.r.max(rhs.r), self.g.max(rhs.g), self.b.max(rhs.b))
    }

    pub fn abs(self) -> Self {
        Self::new(self.r.abs(), self.g.abs(), self.b.abs())
    }

    pub fn round(self) -> Self {
        Self::new(self.r.round(), self.g.round(), self.b.round())
    }

    pub fn floor(self) -> Self {
        Self::new(self.r.floor(), self.g.floor(), self.b.floor())
    }

    pub fn ceil(self) -> Self {
        Self::new(self.r.ceil(), self.g.ceil(), self.b.ceil())
    }

    pub fn trunc(self) -> Self {
        Self::new(self.r.trunc(), self.g.trunc(), self.b.trunc())
    }

    pub fn fract(self) -> Self {
        Self::new(self.r.fract(), self.g.fract(), self.b.fract())
    }

    pub fn sqrt(self) -> Self {
        Self::new(self.r.sqrt(), self.g.sqrt(), self.b.sqrt())
    }

    pub fn exp(self) -> Self {
        Self::new(self.r.exp(), self.g.exp(), self.b.exp())
    }

    pub fn exp2(self) -> Self {
        Self::new(self.r.exp2(), self.g.exp2(), self.b.exp2())
    }

    pub fn ln(self) -> Self {
        Self::new(self.r.ln(), self.g.ln(), self.b.ln())
    }

    pub fn log(self, base: f32) -> Self {
        Self::new(self.r.log(base), self.g.log(base), self.b.log(base))
    }

    pub fn log2(self) -> Self {
        Self::new(self.r.log2(), self.g.log2(), self.b.log2())
    }

    pub fn log10(self) -> Self {
        Self::new(self.r.log10(), self.g.log10(), self.b.log10())
    }

    pub fn cbrt(self) -> Self {
        Self::new(self.r.cbrt(), self.g.cbrt(), self.b.cbrt())
    }

    pub fn powf(self, n: f32) -> Self {
        Self::new(self.r.powf(n), self.g.powf(n), self.b.powf(n))
    }

    pub fn sin(self) -> Self {
        Self::new(self.r.sin(), self.g.sin(), self.b.sin())
    }

    pub fn cos(self) -> Self {
        Self::new(self.r.cos(), self.g.cos(), self.b.cos())
    }

    pub fn tan(self) -> Self {
        Self::new(self.r.tan(), self.g.tan(), self.b.tan())
    }

    pub fn sin_cos(self) -> (Self, Self) {
        (self.sin(), self.cos())
    }

    pub fn recip(self) -> Self {
        Self::new(self.r.recip(), self.g.recip(), self.b.recip())
    }

    pub fn saturate(self) -> Self {
        self.clamp(Self::black(), Self::white())
    }

    pub fn lerp(self, rhs: Self, s: f32) -> Self {
        (rhs - self) * s + self
    }

    pub fn gamma_correct(self) -> Self {
        self.powf(1.0 / 2.2)
    }
}

impl RGBf {
    pub fn to_array(self) -> [f32; 3] {
        [self.r, self.g, self.b]
    }

    pub fn to_tuple(self) -> (f32, f32, f32) {
        (self.r, self.g, self.b)
    }

    pub fn to_rgb(self) -> RGB {
        RGB::new(self.r as f64, self.g as f64, self.b as f64)
    }

    pub fn to_rgba(self) -> RGBA {
        RGBA::new(self.r as f64, self.g as f64, self.b as f64, 1.0)
    }

    pub fn to_rgba_alpha(self, alpha: f64) -> RGBA {
        RGBA::new(self.r as f64, self.g as f64, self.b as f64, alpha)
    }

    pub fn to_rgbaf(self) -> RGBAf {
        RGBAf::new(self.r, self.g, self.b, 1.0)
    }

    pub fn to_rgbaf_alpha(self, alpha: f32) -> RGBAf {
        RGBAf::new(self.r, self.g, self.b, alpha)
    }

    pub fn to_rgb24(self) -> RGB24 {
        RGB24::new(
            convert_f32_to_u8(self.r),
            convert_f32_to_u8(self.g),
            convert_f32_to_u8(self.b),
        )
    }

    pub fn to_rgba32(self) -> RGBA32 {
        RGBA32::new(
            convert_f32_to_u8(self.r),
            convert_f32_to_u8(self.g),
            convert_f32_to_u8(self.b),
            255,
        )
    }

    pub fn to_rgba32_alpha(self, alpha: u8) -> RGBA32 {
        RGBA32::new(
            convert_f32_to_u8(self.r),
            convert_f32_to_u8(self.g),
            convert_f32_to_u8(self.b),
            alpha,
        )
    }
}
