#![allow(dead_code)]

use std::{
    fmt::Display,
    ops::{Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign},
};

use super::{vec4f, Vec2d, Vec3d, Vec4f};

#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Vec4d {
    x: f64,
    y: f64,
    z: f64,
    w: f64,
}

pub fn vec4d(x: f64, y: f64, z: f64, w: f64) -> Vec4d {
    Vec4d::new(x, y, z, w)
}

impl Display for Vec4d {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Vec4d(x: {}, y: {}, z: {}, w: {})",
            self.x, self.y, self.z, self.w
        )
    }
}

impl Default for Vec4d {
    fn default() -> Self {
        Self::new(0.0, 0.0, 0.0, 0.0)
    }
}

impl Add<Vec4d> for Vec4d {
    type Output = Vec4d;

    fn add(self, rhs: Vec4d) -> Self::Output {
        Vec4d::new(
            self.x + rhs.x,
            self.y + rhs.y,
            self.z + rhs.z,
            self.w + rhs.w,
        )
    }
}

impl Add<f64> for Vec4d {
    type Output = Vec4d;

    fn add(self, rhs: f64) -> Self::Output {
        Vec4d::new(self.x + rhs, self.y + rhs, self.z + rhs, self.w + rhs)
    }
}

impl Add<Vec4d> for f64 {
    type Output = Vec4d;

    fn add(self, rhs: Vec4d) -> Self::Output {
        Vec4d::new(self + rhs.x, self + rhs.y, self + rhs.z, self + rhs.w)
    }
}

impl AddAssign<Vec4d> for Vec4d {
    fn add_assign(&mut self, rhs: Vec4d) {
        self.x += rhs.x;
        self.y += rhs.y;
        self.z += rhs.z;
        self.w += rhs.w;
    }
}

impl AddAssign<f64> for Vec4d {
    fn add_assign(&mut self, rhs: f64) {
        self.x += rhs;
        self.y += rhs;
        self.z += rhs;
        self.w += rhs;
    }
}

impl Sub<Vec4d> for Vec4d {
    type Output = Vec4d;

    fn sub(self, rhs: Vec4d) -> Self::Output {
        Vec4d::new(
            self.x - rhs.x,
            self.y - rhs.y,
            self.z - rhs.z,
            self.w - rhs.w,
        )
    }
}

impl Sub<f64> for Vec4d {
    type Output = Vec4d;

    fn sub(self, rhs: f64) -> Self::Output {
        Vec4d::new(self.x - rhs, self.y - rhs, self.z - rhs, self.w - rhs)
    }
}

impl Sub<Vec4d> for f64 {
    type Output = Vec4d;

    fn sub(self, rhs: Vec4d) -> Self::Output {
        Vec4d::new(self - rhs.x, self - rhs.y, self - rhs.z, self - rhs.w)
    }
}

impl SubAssign<Vec4d> for Vec4d {
    fn sub_assign(&mut self, rhs: Vec4d) {
        self.x -= rhs.x;
        self.y -= rhs.y;
        self.z -= rhs.z;
        self.w -= rhs.w;
    }
}

impl SubAssign<f64> for Vec4d {
    fn sub_assign(&mut self, rhs: f64) {
        self.x -= rhs;
        self.y -= rhs;
        self.z -= rhs;
        self.w -= rhs;
    }
}

impl Mul<Vec4d> for Vec4d {
    type Output = Vec4d;

    fn mul(self, rhs: Vec4d) -> Self::Output {
        Vec4d::new(
            self.x * rhs.x,
            self.y * rhs.y,
            self.z * rhs.z,
            self.w * rhs.w,
        )
    }
}

impl Mul<f64> for Vec4d {
    type Output = Vec4d;

    fn mul(self, rhs: f64) -> Self::Output {
        Vec4d::new(self.x * rhs, self.y * rhs, self.z * rhs, self.w * rhs)
    }
}

impl Mul<Vec4d> for f64 {
    type Output = Vec4d;

    fn mul(self, rhs: Vec4d) -> Self::Output {
        Vec4d::new(self * rhs.x, self * rhs.y, self * rhs.z, self * rhs.w)
    }
}

impl MulAssign<Vec4d> for Vec4d {
    fn mul_assign(&mut self, rhs: Vec4d) {
        self.x *= rhs.x;
        self.y *= rhs.y;
        self.z *= rhs.z;
        self.w *= rhs.w;
    }
}

impl MulAssign<f64> for Vec4d {
    fn mul_assign(&mut self, rhs: f64) {
        self.x *= rhs;
        self.y *= rhs;
        self.z *= rhs;
        self.w *= rhs;
    }
}

impl Div<Vec4d> for Vec4d {
    type Output = Vec4d;

    fn div(self, rhs: Vec4d) -> Self::Output {
        Vec4d::new(
            self.x / rhs.x,
            self.y / rhs.y,
            self.z / rhs.z,
            self.w / rhs.w,
        )
    }
}

impl Div<f64> for Vec4d {
    type Output = Vec4d;

    fn div(self, rhs: f64) -> Self::Output {
        Vec4d::new(self.x / rhs, self.y / rhs, self.z / rhs, self.w / rhs)
    }
}

impl Div<Vec4d> for f64 {
    type Output = Vec4d;

    fn div(self, rhs: Vec4d) -> Self::Output {
        Vec4d::new(self / rhs.x, self / rhs.y, self / rhs.z, self / rhs.w)
    }
}

impl DivAssign<Vec4d> for Vec4d {
    fn div_assign(&mut self, rhs: Vec4d) {
        self.x /= rhs.x;
        self.y /= rhs.y;
        self.z /= rhs.z;
        self.w /= rhs.w;
    }
}

impl DivAssign<f64> for Vec4d {
    fn div_assign(&mut self, rhs: f64) {
        self.x /= rhs;
        self.y /= rhs;
        self.z /= rhs;
        self.w /= rhs;
    }
}

impl Neg for Vec4d {
    type Output = Vec4d;

    fn neg(self) -> Self::Output {
        Self::new(-self.x, -self.y, -self.z, -self.w)
    }
}

impl Index<usize> for Vec4d {
    type Output = f64;

    fn index(&self, index: usize) -> &Self::Output {
        match index {
            0 => &self.x,
            1 => &self.y,
            2 => &self.z,
            3 => &self.w,
            _ => panic!("`rmath::algebra::Vec4d::index`: index out of bounds."),
        }
    }
}

impl IndexMut<usize> for Vec4d {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        match index {
            0 => &mut self.x,
            1 => &mut self.y,
            2 => &mut self.z,
            3 => &mut self.w,
            _ => panic!("`rmath::algebra::Vec4d::index_mut`: index out of bounds."),
        }
    }
}

impl From<f64> for Vec4d {
    fn from(v: f64) -> Self {
        Self::new(v, v, v, v)
    }
}

impl From<(f64, f64, f64, f64)> for Vec4d {
    fn from(v: (f64, f64, f64, f64)) -> Self {
        let (x, y, z, w) = v;
        Self::new(x, y, z, w)
    }
}

impl From<(Vec2d, f64, f64)> for Vec4d {
    fn from(v: (Vec2d, f64, f64)) -> Self {
        let (xy, z, w) = v;
        let (x, y) = xy.to_tuple();
        Self::new(x, y, z, w)
    }
}

impl From<(Vec3d, f64)> for Vec4d {
    fn from(v: (Vec3d, f64)) -> Self {
        let (xyz, w) = v;
        let (x, y, z) = xyz.to_tuple();
        Self::new(x, y, z, w)
    }
}

impl Vec4d {
    pub fn new(x: f64, y: f64, z: f64, w: f64) -> Self {
        Self { x, y, z, w }
    }

    pub fn one() -> Self {
        Self::new(1.0, 1.0, 1.0, 1.0)
    }

    pub fn zero() -> Self {
        Self::new(0.0, 0.0, 0.0, 0.0)
    }
}

impl Vec4d {
    pub fn floor(self) -> Self {
        Self::new(
            self.x.floor(),
            self.y.floor(),
            self.z.floor(),
            self.w.floor(),
        )
    }

    pub fn ceil(self) -> Self {
        Self::new(self.x.ceil(), self.y.ceil(), self.z.ceil(), self.w.ceil())
    }

    pub fn round(self) -> Self {
        Self::new(
            self.x.round(),
            self.y.round(),
            self.z.round(),
            self.w.round(),
        )
    }

    pub fn trunc(self) -> Self {
        Self::new(
            self.x.trunc(),
            self.y.trunc(),
            self.z.trunc(),
            self.w.trunc(),
        )
    }

    pub fn fract(self) -> Self {
        Self::new(
            self.x.fract(),
            self.y.fract(),
            self.z.fract(),
            self.w.fract(),
        )
    }

    pub fn abs(self) -> Self {
        Self::new(self.x.abs(), self.y.abs(), self.z.abs(), self.w.abs())
    }

    pub fn signum(self) -> Self {
        Self::new(
            self.x.signum(),
            self.y.signum(),
            self.z.signum(),
            self.w.signum(),
        )
    }

    pub fn powf(self, n: f64) -> Self {
        Self::new(
            self.x.powf(n),
            self.y.powf(n),
            self.z.powf(n),
            self.w.powf(n),
        )
    }

    pub fn sqrt(self) -> Self {
        Self::new(self.x.sqrt(), self.y.sqrt(), self.z.sqrt(), self.w.sqrt())
    }

    pub fn exp(self) -> Self {
        Self::new(self.x.exp(), self.y.exp(), self.z.exp(), self.w.exp())
    }

    pub fn exp2(self) -> Self {
        Self::new(self.x.exp2(), self.y.exp2(), self.z.exp2(), self.w.exp2())
    }

    pub fn ln(self) -> Self {
        Self::new(self.x.ln(), self.y.ln(), self.z.ln(), self.w.ln())
    }

    pub fn log(self, base: f64) -> Self {
        Self::new(
            self.x.log(base),
            self.y.log(base),
            self.z.log(base),
            self.w.log(base),
        )
    }

    pub fn log2(self) -> Self {
        Self::new(self.x.log2(), self.y.log2(), self.z.log2(), self.w.log2())
    }

    pub fn log10(self) -> Self {
        Self::new(
            self.x.log10(),
            self.y.log10(),
            self.z.log10(),
            self.w.log10(),
        )
    }

    pub fn cbrt(self) -> Self {
        Self::new(self.x.cbrt(), self.y.cbrt(), self.z.cbrt(), self.w.cbrt())
    }

    pub fn sin(self) -> Self {
        Self::new(self.x.sin(), self.y.sin(), self.z.sin(), self.w.sin())
    }

    pub fn cos(self) -> Self {
        Self::new(self.x.cos(), self.y.cos(), self.z.cos(), self.w.cos())
    }

    pub fn tan(self) -> Self {
        Self::new(self.x.tan(), self.y.tan(), self.z.tan(), self.w.tan())
    }

    pub fn sin_cos(self) -> (Self, Self) {
        (self.sin(), self.cos())
    }

    pub fn lerp(self, rhs: Self, s: f64) -> Self {
        self + (rhs - self) * s
    }

    pub fn lerp_vec(self, rhs: Self, s: Self) -> Self {
        self + (rhs - self) * s
    }

    pub fn is_nan(self) -> bool {
        self.x.is_nan() || self.y.is_nan() || self.z.is_nan() || self.w.is_nan()
    }

    pub fn is_infinite(self) -> bool {
        self.x.is_infinite() || self.y.is_infinite() || self.z.is_infinite() || self.w.is_infinite()
    }

    pub fn is_finite(self) -> bool {
        self.x.is_finite() && self.y.is_finite() && self.z.is_finite() && self.w.is_finite()
    }

    pub fn recip(self) -> Self {
        Self::new(
            self.x.recip(),
            self.y.recip(),
            self.z.recip(),
            self.w.recip(),
        )
    }

    pub fn max(self, rhs: Self) -> Self {
        Self::new(
            self.x.max(rhs.x),
            self.y.max(rhs.y),
            self.z.max(rhs.z),
            self.w.max(rhs.w),
        )
    }

    pub fn min(self, rhs: Self) -> Self {
        Self::new(
            self.x.min(rhs.x),
            self.y.min(rhs.y),
            self.z.min(rhs.z),
            self.w.min(rhs.w),
        )
    }

    pub fn clamp(self, min: Self, max: Self) -> Self {
        ruby_assert!(min.x <= max.x);
        ruby_assert!(min.y <= max.y);
        ruby_assert!(min.z <= max.z);
        ruby_assert!(min.w <= max.w);

        self.min(max).max(min)
    }

    pub fn saturate(self) -> Self {
        self.clamp(Self::zero(), Self::one())
    }

    pub fn min_element(self) -> f64 {
        self.x.min(self.y).min(self.z).min(self.w)
    }

    pub fn max_element(self) -> f64 {
        self.x.max(self.y).max(self.z).max(self.w)
    }
}

impl Vec4d {
    pub fn dot(self, rhs: Self) -> f64 {
        self.x * rhs.x + self.y * rhs.y + self.z * rhs.z
    }

    pub fn length(self) -> f64 {
        self.dot(self).sqrt()
    }

    pub fn length_squared(self) -> f64 {
        self.dot(self)
    }

    pub fn length_recip(self) -> f64 {
        self.length().recip()
    }

    pub fn distance(self, rhs: Self) -> f64 {
        (rhs - self).length()
    }

    pub fn distance_squared(self, rhs: Self) -> f64 {
        (rhs - self).length_squared()
    }

    pub fn normalize(self) -> Self {
        let normalized = self * self.length_recip();
        ruby_assert!(normalized.is_finite());
        normalized
    }

    pub fn try_normalize(self) -> Option<Self> {
        let recip = self.length_recip();
        if recip.is_finite() && recip > 0.0 {
            Some(self * recip)
        } else {
            None
        }
    }

    pub fn normalize_or_zero(self) -> Self {
        let recip = self.length_recip();
        if recip.is_finite() && recip > 0.0 {
            self * recip
        } else {
            Self::zero()
        }
    }

    pub fn is_normalized(self) -> bool {
        (self.length_squared() - 1.0f64).abs() <= f64::EPSILON
    }
}

impl Vec4d {
    pub fn to_array(self) -> [f64; 3] {
        [self.x, self.y, self.z]
    }

    pub fn to_tuple(self) -> (f64, f64, f64) {
        (self.x, self.y, self.z)
    }

    pub fn as_vec4f(self) -> Vec4f {
        vec4f(self.x as f32, self.y as f32, self.z as f32, self.w as f32)
    }
}

impl Vec4d {
    pub fn x(self) -> f64 {
        self.x
    }

    pub fn y(self) -> f64 {
        self.y
    }

    pub fn z(self) -> f64 {
        self.z
    }

    pub fn w(self) -> f64 {
        self.w
    }

    pub fn xx(self) -> Vec2d {
        Vec2d::new(self.x, self.x)
    }

    pub fn xy(self) -> Vec2d {
        Vec2d::new(self.x, self.y)
    }

    pub fn xz(self) -> Vec2d {
        Vec2d::new(self.x, self.z)
    }

    pub fn xw(self) -> Vec2d {
        Vec2d::new(self.x, self.w)
    }

    pub fn yx(self) -> Vec2d {
        Vec2d::new(self.y, self.x)
    }

    pub fn yy(self) -> Vec2d {
        Vec2d::new(self.y, self.y)
    }

    pub fn yz(self) -> Vec2d {
        Vec2d::new(self.y, self.z)
    }

    pub fn yw(self) -> Vec2d {
        Vec2d::new(self.y, self.w)
    }

    pub fn zx(self) -> Vec2d {
        Vec2d::new(self.z, self.x)
    }

    pub fn zy(self) -> Vec2d {
        Vec2d::new(self.z, self.y)
    }

    pub fn zz(self) -> Vec2d {
        Vec2d::new(self.z, self.z)
    }

    pub fn zw(self) -> Vec2d {
        Vec2d::new(self.z, self.w)
    }

    pub fn wx(self) -> Vec2d {
        Vec2d::new(self.w, self.x)
    }

    pub fn wy(self) -> Vec2d {
        Vec2d::new(self.w, self.y)
    }

    pub fn wz(self) -> Vec2d {
        Vec2d::new(self.w, self.z)
    }

    pub fn ww(self) -> Vec2d {
        Vec2d::new(self.w, self.w)
    }

    pub fn xxx(self) -> Vec3d {
        Vec3d::new(self.x, self.x, self.x)
    }

    pub fn xxy(self) -> Vec3d {
        Vec3d::new(self.x, self.x, self.y)
    }

    pub fn xxz(self) -> Vec3d {
        Vec3d::new(self.x, self.x, self.z)
    }

    pub fn xxw(self) -> Vec3d {
        Vec3d::new(self.x, self.x, self.w)
    }

    pub fn xyx(self) -> Vec3d {
        Vec3d::new(self.x, self.y, self.x)
    }

    pub fn xyy(self) -> Vec3d {
        Vec3d::new(self.x, self.y, self.y)
    }

    pub fn xyz(self) -> Vec3d {
        Vec3d::new(self.x, self.y, self.z)
    }

    pub fn xyw(self) -> Vec3d {
        Vec3d::new(self.x, self.y, self.w)
    }

    pub fn xzx(self) -> Vec3d {
        Vec3d::new(self.x, self.z, self.x)
    }

    pub fn xzy(self) -> Vec3d {
        Vec3d::new(self.x, self.z, self.y)
    }

    pub fn xzz(self) -> Vec3d {
        Vec3d::new(self.x, self.z, self.z)
    }

    pub fn xzw(self) -> Vec3d {
        Vec3d::new(self.x, self.z, self.w)
    }

    pub fn xwx(self) -> Vec3d {
        Vec3d::new(self.x, self.w, self.x)
    }

    pub fn xwy(self) -> Vec3d {
        Vec3d::new(self.x, self.w, self.y)
    }

    pub fn xwz(self) -> Vec3d {
        Vec3d::new(self.x, self.w, self.z)
    }

    pub fn xww(self) -> Vec3d {
        Vec3d::new(self.x, self.w, self.w)
    }

    pub fn yxx(self) -> Vec3d {
        Vec3d::new(self.y, self.x, self.x)
    }

    pub fn yxy(self) -> Vec3d {
        Vec3d::new(self.y, self.x, self.y)
    }

    pub fn yxz(self) -> Vec3d {
        Vec3d::new(self.y, self.x, self.z)
    }

    pub fn yxw(self) -> Vec3d {
        Vec3d::new(self.y, self.x, self.w)
    }

    pub fn yyx(self) -> Vec3d {
        Vec3d::new(self.y, self.y, self.x)
    }

    pub fn yyy(self) -> Vec3d {
        Vec3d::new(self.y, self.y, self.y)
    }

    pub fn yyz(self) -> Vec3d {
        Vec3d::new(self.y, self.y, self.z)
    }

    pub fn yyw(self) -> Vec3d {
        Vec3d::new(self.y, self.y, self.w)
    }

    pub fn yzx(self) -> Vec3d {
        Vec3d::new(self.y, self.z, self.x)
    }

    pub fn yzy(self) -> Vec3d {
        Vec3d::new(self.y, self.z, self.y)
    }

    pub fn yzz(self) -> Vec3d {
        Vec3d::new(self.y, self.z, self.z)
    }

    pub fn yzw(self) -> Vec3d {
        Vec3d::new(self.y, self.z, self.w)
    }

    pub fn ywx(self) -> Vec3d {
        Vec3d::new(self.y, self.w, self.x)
    }

    pub fn ywy(self) -> Vec3d {
        Vec3d::new(self.y, self.w, self.y)
    }

    pub fn ywz(self) -> Vec3d {
        Vec3d::new(self.y, self.w, self.z)
    }

    pub fn yww(self) -> Vec3d {
        Vec3d::new(self.y, self.w, self.w)
    }

    pub fn zxx(self) -> Vec3d {
        Vec3d::new(self.z, self.x, self.x)
    }

    pub fn zxy(self) -> Vec3d {
        Vec3d::new(self.z, self.x, self.y)
    }

    pub fn zxz(self) -> Vec3d {
        Vec3d::new(self.z, self.x, self.z)
    }

    pub fn zxw(self) -> Vec3d {
        Vec3d::new(self.z, self.x, self.w)
    }

    pub fn zyx(self) -> Vec3d {
        Vec3d::new(self.z, self.y, self.x)
    }

    pub fn zyy(self) -> Vec3d {
        Vec3d::new(self.z, self.y, self.y)
    }

    pub fn zyz(self) -> Vec3d {
        Vec3d::new(self.z, self.y, self.z)
    }

    pub fn zyw(self) -> Vec3d {
        Vec3d::new(self.z, self.y, self.w)
    }

    pub fn zzx(self) -> Vec3d {
        Vec3d::new(self.z, self.z, self.x)
    }

    pub fn zzy(self) -> Vec3d {
        Vec3d::new(self.z, self.z, self.y)
    }

    pub fn zzz(self) -> Vec3d {
        Vec3d::new(self.z, self.z, self.z)
    }

    pub fn zzw(self) -> Vec3d {
        Vec3d::new(self.z, self.z, self.w)
    }

    pub fn zwx(self) -> Vec3d {
        Vec3d::new(self.z, self.w, self.x)
    }

    pub fn zwy(self) -> Vec3d {
        Vec3d::new(self.z, self.w, self.y)
    }

    pub fn zwz(self) -> Vec3d {
        Vec3d::new(self.z, self.w, self.z)
    }

    pub fn zww(self) -> Vec3d {
        Vec3d::new(self.z, self.w, self.w)
    }

    pub fn wxx(self) -> Vec3d {
        Vec3d::new(self.w, self.x, self.x)
    }

    pub fn wxy(self) -> Vec3d {
        Vec3d::new(self.w, self.x, self.y)
    }

    pub fn wxz(self) -> Vec3d {
        Vec3d::new(self.w, self.x, self.z)
    }

    pub fn wxw(self) -> Vec3d {
        Vec3d::new(self.w, self.x, self.w)
    }

    pub fn wyx(self) -> Vec3d {
        Vec3d::new(self.w, self.y, self.x)
    }

    pub fn wyy(self) -> Vec3d {
        Vec3d::new(self.w, self.y, self.y)
    }

    pub fn wyz(self) -> Vec3d {
        Vec3d::new(self.w, self.y, self.z)
    }

    pub fn wyw(self) -> Vec3d {
        Vec3d::new(self.w, self.y, self.w)
    }

    pub fn wzx(self) -> Vec3d {
        Vec3d::new(self.w, self.z, self.x)
    }

    pub fn wzy(self) -> Vec3d {
        Vec3d::new(self.w, self.z, self.y)
    }

    pub fn wzz(self) -> Vec3d {
        Vec3d::new(self.w, self.z, self.z)
    }

    pub fn wzw(self) -> Vec3d {
        Vec3d::new(self.w, self.z, self.w)
    }

    pub fn wwx(self) -> Vec3d {
        Vec3d::new(self.w, self.w, self.x)
    }

    pub fn wwy(self) -> Vec3d {
        Vec3d::new(self.w, self.w, self.y)
    }

    pub fn wwz(self) -> Vec3d {
        Vec3d::new(self.w, self.w, self.z)
    }

    pub fn www(self) -> Vec3d {
        Vec3d::new(self.w, self.w, self.w)
    }

    pub fn xxxx(self) -> Self {
        Self::new(self.x, self.x, self.x, self.x)
    }

    pub fn xxxy(self) -> Self {
        Self::new(self.x, self.x, self.x, self.y)
    }

    pub fn xxxz(self) -> Self {
        Self::new(self.x, self.x, self.x, self.z)
    }

    pub fn xxxw(self) -> Self {
        Self::new(self.x, self.x, self.x, self.w)
    }

    pub fn xxyx(self) -> Self {
        Self::new(self.x, self.x, self.y, self.x)
    }

    pub fn xxyy(self) -> Self {
        Self::new(self.x, self.x, self.y, self.y)
    }

    pub fn xxyz(self) -> Self {
        Self::new(self.x, self.x, self.y, self.z)
    }

    pub fn xxyw(self) -> Self {
        Self::new(self.x, self.x, self.y, self.w)
    }

    pub fn xxzx(self) -> Self {
        Self::new(self.x, self.x, self.z, self.x)
    }

    pub fn xxzy(self) -> Self {
        Self::new(self.x, self.x, self.z, self.y)
    }

    pub fn xxzz(self) -> Self {
        Self::new(self.x, self.x, self.z, self.z)
    }

    pub fn xxzw(self) -> Self {
        Self::new(self.x, self.x, self.z, self.w)
    }

    pub fn xxwx(self) -> Self {
        Self::new(self.x, self.x, self.w, self.x)
    }

    pub fn xxwy(self) -> Self {
        Self::new(self.x, self.x, self.w, self.y)
    }

    pub fn xxwz(self) -> Self {
        Self::new(self.x, self.x, self.w, self.z)
    }

    pub fn xxww(self) -> Self {
        Self::new(self.x, self.x, self.w, self.w)
    }

    pub fn xyxx(self) -> Self {
        Self::new(self.x, self.y, self.x, self.x)
    }

    pub fn xyxy(self) -> Self {
        Self::new(self.x, self.y, self.x, self.y)
    }

    pub fn xyxz(self) -> Self {
        Self::new(self.x, self.y, self.x, self.z)
    }

    pub fn xyxw(self) -> Self {
        Self::new(self.x, self.y, self.x, self.w)
    }

    pub fn xyyx(self) -> Self {
        Self::new(self.x, self.y, self.y, self.x)
    }

    pub fn xyyy(self) -> Self {
        Self::new(self.x, self.y, self.y, self.y)
    }

    pub fn xyyz(self) -> Self {
        Self::new(self.x, self.y, self.y, self.z)
    }

    pub fn xyyw(self) -> Self {
        Self::new(self.x, self.y, self.y, self.w)
    }

    pub fn xyzx(self) -> Self {
        Self::new(self.x, self.y, self.z, self.x)
    }

    pub fn xyzy(self) -> Self {
        Self::new(self.x, self.y, self.z, self.y)
    }

    pub fn xyzz(self) -> Self {
        Self::new(self.x, self.y, self.z, self.z)
    }

    pub fn xyzw(self) -> Self {
        Self::new(self.x, self.y, self.z, self.w)
    }

    pub fn xywx(self) -> Self {
        Self::new(self.x, self.y, self.w, self.x)
    }

    pub fn xywy(self) -> Self {
        Self::new(self.x, self.y, self.w, self.y)
    }

    pub fn xywz(self) -> Self {
        Self::new(self.x, self.y, self.w, self.z)
    }

    pub fn xyww(self) -> Self {
        Self::new(self.x, self.y, self.w, self.w)
    }

    pub fn xzxx(self) -> Self {
        Self::new(self.x, self.z, self.x, self.x)
    }

    pub fn xzxy(self) -> Self {
        Self::new(self.x, self.z, self.x, self.y)
    }

    pub fn xzxz(self) -> Self {
        Self::new(self.x, self.z, self.x, self.z)
    }

    pub fn xzxw(self) -> Self {
        Self::new(self.x, self.z, self.x, self.w)
    }

    pub fn xzyx(self) -> Self {
        Self::new(self.x, self.z, self.y, self.x)
    }

    pub fn xzyy(self) -> Self {
        Self::new(self.x, self.z, self.y, self.y)
    }

    pub fn xzyz(self) -> Self {
        Self::new(self.x, self.z, self.y, self.z)
    }

    pub fn xzyw(self) -> Self {
        Self::new(self.x, self.z, self.y, self.w)
    }

    pub fn xzzx(self) -> Self {
        Self::new(self.x, self.z, self.z, self.x)
    }

    pub fn xzzy(self) -> Self {
        Self::new(self.x, self.z, self.z, self.y)
    }

    pub fn xzzz(self) -> Self {
        Self::new(self.x, self.z, self.z, self.z)
    }

    pub fn xzzw(self) -> Self {
        Self::new(self.x, self.z, self.z, self.w)
    }

    pub fn xzwx(self) -> Self {
        Self::new(self.x, self.z, self.w, self.x)
    }

    pub fn xzwy(self) -> Self {
        Self::new(self.x, self.z, self.w, self.y)
    }

    pub fn xzwz(self) -> Self {
        Self::new(self.x, self.z, self.w, self.z)
    }

    pub fn xzww(self) -> Self {
        Self::new(self.x, self.z, self.w, self.w)
    }

    pub fn xwxx(self) -> Self {
        Self::new(self.x, self.w, self.x, self.x)
    }

    pub fn xwxy(self) -> Self {
        Self::new(self.x, self.w, self.x, self.y)
    }

    pub fn xwxz(self) -> Self {
        Self::new(self.x, self.w, self.x, self.z)
    }

    pub fn xwxw(self) -> Self {
        Self::new(self.x, self.w, self.x, self.w)
    }

    pub fn xwyx(self) -> Self {
        Self::new(self.x, self.w, self.y, self.x)
    }

    pub fn xwyy(self) -> Self {
        Self::new(self.x, self.w, self.y, self.y)
    }

    pub fn xwyz(self) -> Self {
        Self::new(self.x, self.w, self.y, self.z)
    }

    pub fn xwyw(self) -> Self {
        Self::new(self.x, self.w, self.y, self.w)
    }

    pub fn xwzx(self) -> Self {
        Self::new(self.x, self.w, self.z, self.x)
    }

    pub fn xwzy(self) -> Self {
        Self::new(self.x, self.w, self.z, self.y)
    }

    pub fn xwzz(self) -> Self {
        Self::new(self.x, self.w, self.z, self.z)
    }

    pub fn xwzw(self) -> Self {
        Self::new(self.x, self.w, self.z, self.w)
    }

    pub fn xwwx(self) -> Self {
        Self::new(self.x, self.w, self.w, self.x)
    }

    pub fn xwwy(self) -> Self {
        Self::new(self.x, self.w, self.w, self.y)
    }

    pub fn xwwz(self) -> Self {
        Self::new(self.x, self.w, self.w, self.z)
    }

    pub fn xwww(self) -> Self {
        Self::new(self.x, self.w, self.w, self.w)
    }

    pub fn yxxx(self) -> Self {
        Self::new(self.y, self.x, self.x, self.x)
    }

    pub fn yxxy(self) -> Self {
        Self::new(self.y, self.x, self.x, self.y)
    }

    pub fn yxxz(self) -> Self {
        Self::new(self.y, self.x, self.x, self.z)
    }

    pub fn yxxw(self) -> Self {
        Self::new(self.y, self.x, self.x, self.w)
    }

    pub fn yxyx(self) -> Self {
        Self::new(self.y, self.x, self.y, self.x)
    }

    pub fn yxyy(self) -> Self {
        Self::new(self.y, self.x, self.y, self.y)
    }

    pub fn yxyz(self) -> Self {
        Self::new(self.y, self.x, self.y, self.z)
    }

    pub fn yxyw(self) -> Self {
        Self::new(self.y, self.x, self.y, self.w)
    }

    pub fn yxzx(self) -> Self {
        Self::new(self.y, self.x, self.z, self.x)
    }

    pub fn yxzy(self) -> Self {
        Self::new(self.y, self.x, self.z, self.y)
    }

    pub fn yxzz(self) -> Self {
        Self::new(self.y, self.x, self.z, self.z)
    }

    pub fn yxzw(self) -> Self {
        Self::new(self.y, self.x, self.z, self.w)
    }

    pub fn yxwx(self) -> Self {
        Self::new(self.y, self.x, self.w, self.x)
    }

    pub fn yxwy(self) -> Self {
        Self::new(self.y, self.x, self.w, self.y)
    }

    pub fn yxwz(self) -> Self {
        Self::new(self.y, self.x, self.w, self.z)
    }

    pub fn yxww(self) -> Self {
        Self::new(self.y, self.x, self.w, self.w)
    }

    pub fn yyxx(self) -> Self {
        Self::new(self.y, self.y, self.x, self.x)
    }

    pub fn yyxy(self) -> Self {
        Self::new(self.y, self.y, self.x, self.y)
    }

    pub fn yyxz(self) -> Self {
        Self::new(self.y, self.y, self.x, self.z)
    }

    pub fn yyxw(self) -> Self {
        Self::new(self.y, self.y, self.x, self.w)
    }

    pub fn yyyx(self) -> Self {
        Self::new(self.y, self.y, self.y, self.x)
    }

    pub fn yyyy(self) -> Self {
        Self::new(self.y, self.y, self.y, self.y)
    }

    pub fn yyyz(self) -> Self {
        Self::new(self.y, self.y, self.y, self.z)
    }

    pub fn yyyw(self) -> Self {
        Self::new(self.y, self.y, self.y, self.w)
    }

    pub fn yyzx(self) -> Self {
        Self::new(self.y, self.y, self.z, self.x)
    }

    pub fn yyzy(self) -> Self {
        Self::new(self.y, self.y, self.z, self.y)
    }

    pub fn yyzz(self) -> Self {
        Self::new(self.y, self.y, self.z, self.z)
    }

    pub fn yyzw(self) -> Self {
        Self::new(self.y, self.y, self.z, self.w)
    }

    pub fn yywx(self) -> Self {
        Self::new(self.y, self.y, self.w, self.x)
    }

    pub fn yywy(self) -> Self {
        Self::new(self.y, self.y, self.w, self.y)
    }

    pub fn yywz(self) -> Self {
        Self::new(self.y, self.y, self.w, self.z)
    }

    pub fn yyww(self) -> Self {
        Self::new(self.y, self.y, self.w, self.w)
    }

    pub fn yzxx(self) -> Self {
        Self::new(self.y, self.z, self.x, self.x)
    }

    pub fn yzxy(self) -> Self {
        Self::new(self.y, self.z, self.x, self.y)
    }

    pub fn yzxz(self) -> Self {
        Self::new(self.y, self.z, self.x, self.z)
    }

    pub fn yzxw(self) -> Self {
        Self::new(self.y, self.z, self.x, self.w)
    }

    pub fn yzyx(self) -> Self {
        Self::new(self.y, self.z, self.y, self.x)
    }

    pub fn yzyy(self) -> Self {
        Self::new(self.y, self.z, self.y, self.y)
    }

    pub fn yzyz(self) -> Self {
        Self::new(self.y, self.z, self.y, self.z)
    }

    pub fn yzyw(self) -> Self {
        Self::new(self.y, self.z, self.y, self.w)
    }

    pub fn yzzx(self) -> Self {
        Self::new(self.y, self.z, self.z, self.x)
    }

    pub fn yzzy(self) -> Self {
        Self::new(self.y, self.z, self.z, self.y)
    }

    pub fn yzzz(self) -> Self {
        Self::new(self.y, self.z, self.z, self.z)
    }

    pub fn yzzw(self) -> Self {
        Self::new(self.y, self.z, self.z, self.w)
    }

    pub fn yzwx(self) -> Self {
        Self::new(self.y, self.z, self.w, self.x)
    }

    pub fn yzwy(self) -> Self {
        Self::new(self.y, self.z, self.w, self.y)
    }

    pub fn yzwz(self) -> Self {
        Self::new(self.y, self.z, self.w, self.z)
    }

    pub fn yzww(self) -> Self {
        Self::new(self.y, self.z, self.w, self.w)
    }

    pub fn ywxx(self) -> Self {
        Self::new(self.y, self.w, self.x, self.x)
    }

    pub fn ywxy(self) -> Self {
        Self::new(self.y, self.w, self.x, self.y)
    }

    pub fn ywxz(self) -> Self {
        Self::new(self.y, self.w, self.x, self.z)
    }

    pub fn ywxw(self) -> Self {
        Self::new(self.y, self.w, self.x, self.w)
    }

    pub fn ywyx(self) -> Self {
        Self::new(self.y, self.w, self.y, self.x)
    }

    pub fn ywyy(self) -> Self {
        Self::new(self.y, self.w, self.y, self.y)
    }

    pub fn ywyz(self) -> Self {
        Self::new(self.y, self.w, self.y, self.z)
    }

    pub fn ywyw(self) -> Self {
        Self::new(self.y, self.w, self.y, self.w)
    }

    pub fn ywzx(self) -> Self {
        Self::new(self.y, self.w, self.z, self.x)
    }

    pub fn ywzy(self) -> Self {
        Self::new(self.y, self.w, self.z, self.y)
    }

    pub fn ywzz(self) -> Self {
        Self::new(self.y, self.w, self.z, self.z)
    }

    pub fn ywzw(self) -> Self {
        Self::new(self.y, self.w, self.z, self.w)
    }

    pub fn ywwx(self) -> Self {
        Self::new(self.y, self.w, self.w, self.x)
    }

    pub fn ywwy(self) -> Self {
        Self::new(self.y, self.w, self.w, self.y)
    }

    pub fn ywwz(self) -> Self {
        Self::new(self.y, self.w, self.w, self.z)
    }

    pub fn ywww(self) -> Self {
        Self::new(self.y, self.w, self.w, self.w)
    }

    pub fn zxxx(self) -> Self {
        Self::new(self.z, self.x, self.x, self.x)
    }

    pub fn zxxy(self) -> Self {
        Self::new(self.z, self.x, self.x, self.y)
    }

    pub fn zxxz(self) -> Self {
        Self::new(self.z, self.x, self.x, self.z)
    }

    pub fn zxxw(self) -> Self {
        Self::new(self.z, self.x, self.x, self.w)
    }

    pub fn zxyx(self) -> Self {
        Self::new(self.z, self.x, self.y, self.x)
    }

    pub fn zxyy(self) -> Self {
        Self::new(self.z, self.x, self.y, self.y)
    }

    pub fn zxyz(self) -> Self {
        Self::new(self.z, self.x, self.y, self.z)
    }

    pub fn zxyw(self) -> Self {
        Self::new(self.z, self.x, self.y, self.w)
    }

    pub fn zxzx(self) -> Self {
        Self::new(self.z, self.x, self.z, self.x)
    }

    pub fn zxzy(self) -> Self {
        Self::new(self.z, self.x, self.z, self.y)
    }

    pub fn zxzz(self) -> Self {
        Self::new(self.z, self.x, self.z, self.z)
    }

    pub fn zxzw(self) -> Self {
        Self::new(self.z, self.x, self.z, self.w)
    }

    pub fn zxwx(self) -> Self {
        Self::new(self.z, self.x, self.w, self.x)
    }

    pub fn zxwy(self) -> Self {
        Self::new(self.z, self.x, self.w, self.y)
    }

    pub fn zxwz(self) -> Self {
        Self::new(self.z, self.x, self.w, self.z)
    }

    pub fn zxww(self) -> Self {
        Self::new(self.z, self.x, self.w, self.w)
    }

    pub fn zyxx(self) -> Self {
        Self::new(self.z, self.y, self.x, self.x)
    }

    pub fn zyxy(self) -> Self {
        Self::new(self.z, self.y, self.x, self.y)
    }

    pub fn zyxz(self) -> Self {
        Self::new(self.z, self.y, self.x, self.z)
    }

    pub fn zyxw(self) -> Self {
        Self::new(self.z, self.y, self.x, self.w)
    }

    pub fn zyyx(self) -> Self {
        Self::new(self.z, self.y, self.y, self.x)
    }

    pub fn zyyy(self) -> Self {
        Self::new(self.z, self.y, self.y, self.y)
    }

    pub fn zyyz(self) -> Self {
        Self::new(self.z, self.y, self.y, self.z)
    }

    pub fn zyyw(self) -> Self {
        Self::new(self.z, self.y, self.y, self.w)
    }

    pub fn zyzx(self) -> Self {
        Self::new(self.z, self.y, self.z, self.x)
    }

    pub fn zyzy(self) -> Self {
        Self::new(self.z, self.y, self.z, self.y)
    }

    pub fn zyzz(self) -> Self {
        Self::new(self.z, self.y, self.z, self.z)
    }

    pub fn zyzw(self) -> Self {
        Self::new(self.z, self.y, self.z, self.w)
    }

    pub fn zywx(self) -> Self {
        Self::new(self.z, self.y, self.w, self.x)
    }

    pub fn zywy(self) -> Self {
        Self::new(self.z, self.y, self.w, self.y)
    }

    pub fn zywz(self) -> Self {
        Self::new(self.z, self.y, self.w, self.z)
    }

    pub fn zyww(self) -> Self {
        Self::new(self.z, self.y, self.w, self.w)
    }

    pub fn zzxx(self) -> Self {
        Self::new(self.z, self.z, self.x, self.x)
    }

    pub fn zzxy(self) -> Self {
        Self::new(self.z, self.z, self.x, self.y)
    }

    pub fn zzxz(self) -> Self {
        Self::new(self.z, self.z, self.x, self.z)
    }

    pub fn zzxw(self) -> Self {
        Self::new(self.z, self.z, self.x, self.w)
    }

    pub fn zzyx(self) -> Self {
        Self::new(self.z, self.z, self.y, self.x)
    }

    pub fn zzyy(self) -> Self {
        Self::new(self.z, self.z, self.y, self.y)
    }

    pub fn zzyz(self) -> Self {
        Self::new(self.z, self.z, self.y, self.z)
    }

    pub fn zzyw(self) -> Self {
        Self::new(self.z, self.z, self.y, self.w)
    }

    pub fn zzzx(self) -> Self {
        Self::new(self.z, self.z, self.z, self.x)
    }

    pub fn zzzy(self) -> Self {
        Self::new(self.z, self.z, self.z, self.y)
    }

    pub fn zzzz(self) -> Self {
        Self::new(self.z, self.z, self.z, self.z)
    }

    pub fn zzzw(self) -> Self {
        Self::new(self.z, self.z, self.z, self.w)
    }

    pub fn zzwx(self) -> Self {
        Self::new(self.z, self.z, self.w, self.x)
    }

    pub fn zzwy(self) -> Self {
        Self::new(self.z, self.z, self.w, self.y)
    }

    pub fn zzwz(self) -> Self {
        Self::new(self.z, self.z, self.w, self.z)
    }

    pub fn zzww(self) -> Self {
        Self::new(self.z, self.z, self.w, self.w)
    }

    pub fn zwxx(self) -> Self {
        Self::new(self.z, self.w, self.x, self.x)
    }

    pub fn zwxy(self) -> Self {
        Self::new(self.z, self.w, self.x, self.y)
    }

    pub fn zwxz(self) -> Self {
        Self::new(self.z, self.w, self.x, self.z)
    }

    pub fn zwxw(self) -> Self {
        Self::new(self.z, self.w, self.x, self.w)
    }

    pub fn zwyx(self) -> Self {
        Self::new(self.z, self.w, self.y, self.x)
    }

    pub fn zwyy(self) -> Self {
        Self::new(self.z, self.w, self.y, self.y)
    }

    pub fn zwyz(self) -> Self {
        Self::new(self.z, self.w, self.y, self.z)
    }

    pub fn zwyw(self) -> Self {
        Self::new(self.z, self.w, self.y, self.w)
    }

    pub fn zwzx(self) -> Self {
        Self::new(self.z, self.w, self.z, self.x)
    }

    pub fn zwzy(self) -> Self {
        Self::new(self.z, self.w, self.z, self.y)
    }

    pub fn zwzz(self) -> Self {
        Self::new(self.z, self.w, self.z, self.z)
    }

    pub fn zwzw(self) -> Self {
        Self::new(self.z, self.w, self.z, self.w)
    }

    pub fn zwwx(self) -> Self {
        Self::new(self.z, self.w, self.w, self.x)
    }

    pub fn zwwy(self) -> Self {
        Self::new(self.z, self.w, self.w, self.y)
    }

    pub fn zwwz(self) -> Self {
        Self::new(self.z, self.w, self.w, self.z)
    }

    pub fn zwww(self) -> Self {
        Self::new(self.z, self.w, self.w, self.w)
    }

    pub fn wxxx(self) -> Self {
        Self::new(self.w, self.x, self.x, self.x)
    }

    pub fn wxxy(self) -> Self {
        Self::new(self.w, self.x, self.x, self.y)
    }

    pub fn wxxz(self) -> Self {
        Self::new(self.w, self.x, self.x, self.z)
    }

    pub fn wxxw(self) -> Self {
        Self::new(self.w, self.x, self.x, self.w)
    }

    pub fn wxyx(self) -> Self {
        Self::new(self.w, self.x, self.y, self.x)
    }

    pub fn wxyy(self) -> Self {
        Self::new(self.w, self.x, self.y, self.y)
    }

    pub fn wxyz(self) -> Self {
        Self::new(self.w, self.x, self.y, self.z)
    }

    pub fn wxyw(self) -> Self {
        Self::new(self.w, self.x, self.y, self.w)
    }

    pub fn wxzx(self) -> Self {
        Self::new(self.w, self.x, self.z, self.x)
    }

    pub fn wxzy(self) -> Self {
        Self::new(self.w, self.x, self.z, self.y)
    }

    pub fn wxzz(self) -> Self {
        Self::new(self.w, self.x, self.z, self.z)
    }

    pub fn wxzw(self) -> Self {
        Self::new(self.w, self.x, self.z, self.w)
    }

    pub fn wxwx(self) -> Self {
        Self::new(self.w, self.x, self.w, self.x)
    }

    pub fn wxwy(self) -> Self {
        Self::new(self.w, self.x, self.w, self.y)
    }

    pub fn wxwz(self) -> Self {
        Self::new(self.w, self.x, self.w, self.z)
    }

    pub fn wxww(self) -> Self {
        Self::new(self.w, self.x, self.w, self.w)
    }

    pub fn wyxx(self) -> Self {
        Self::new(self.w, self.y, self.x, self.x)
    }

    pub fn wyxy(self) -> Self {
        Self::new(self.w, self.y, self.x, self.y)
    }

    pub fn wyxz(self) -> Self {
        Self::new(self.w, self.y, self.x, self.z)
    }

    pub fn wyxw(self) -> Self {
        Self::new(self.w, self.y, self.x, self.w)
    }

    pub fn wyyx(self) -> Self {
        Self::new(self.w, self.y, self.y, self.x)
    }

    pub fn wyyy(self) -> Self {
        Self::new(self.w, self.y, self.y, self.y)
    }

    pub fn wyyz(self) -> Self {
        Self::new(self.w, self.y, self.y, self.z)
    }

    pub fn wyyw(self) -> Self {
        Self::new(self.w, self.y, self.y, self.w)
    }

    pub fn wyzx(self) -> Self {
        Self::new(self.w, self.y, self.z, self.x)
    }

    pub fn wyzy(self) -> Self {
        Self::new(self.w, self.y, self.z, self.y)
    }

    pub fn wyzz(self) -> Self {
        Self::new(self.w, self.y, self.z, self.z)
    }

    pub fn wyzw(self) -> Self {
        Self::new(self.w, self.y, self.z, self.w)
    }

    pub fn wywx(self) -> Self {
        Self::new(self.w, self.y, self.w, self.x)
    }

    pub fn wywy(self) -> Self {
        Self::new(self.w, self.y, self.w, self.y)
    }

    pub fn wywz(self) -> Self {
        Self::new(self.w, self.y, self.w, self.z)
    }

    pub fn wyww(self) -> Self {
        Self::new(self.w, self.y, self.w, self.w)
    }

    pub fn wzxx(self) -> Self {
        Self::new(self.w, self.z, self.x, self.x)
    }

    pub fn wzxy(self) -> Self {
        Self::new(self.w, self.z, self.x, self.y)
    }

    pub fn wzxz(self) -> Self {
        Self::new(self.w, self.z, self.x, self.z)
    }

    pub fn wzxw(self) -> Self {
        Self::new(self.w, self.z, self.x, self.w)
    }

    pub fn wzyx(self) -> Self {
        Self::new(self.w, self.z, self.y, self.x)
    }

    pub fn wzyy(self) -> Self {
        Self::new(self.w, self.z, self.y, self.y)
    }

    pub fn wzyz(self) -> Self {
        Self::new(self.w, self.z, self.y, self.z)
    }

    pub fn wzyw(self) -> Self {
        Self::new(self.w, self.z, self.y, self.w)
    }

    pub fn wzzx(self) -> Self {
        Self::new(self.w, self.z, self.z, self.x)
    }

    pub fn wzzy(self) -> Self {
        Self::new(self.w, self.z, self.z, self.y)
    }

    pub fn wzzz(self) -> Self {
        Self::new(self.w, self.z, self.z, self.z)
    }

    pub fn wzzw(self) -> Self {
        Self::new(self.w, self.z, self.z, self.w)
    }

    pub fn wzwx(self) -> Self {
        Self::new(self.w, self.z, self.w, self.x)
    }

    pub fn wzwy(self) -> Self {
        Self::new(self.w, self.z, self.w, self.y)
    }

    pub fn wzwz(self) -> Self {
        Self::new(self.w, self.z, self.w, self.z)
    }

    pub fn wzww(self) -> Self {
        Self::new(self.w, self.z, self.w, self.w)
    }

    pub fn wwxx(self) -> Self {
        Self::new(self.w, self.w, self.x, self.x)
    }

    pub fn wwxy(self) -> Self {
        Self::new(self.w, self.w, self.x, self.y)
    }

    pub fn wwxz(self) -> Self {
        Self::new(self.w, self.w, self.x, self.z)
    }

    pub fn wwxw(self) -> Self {
        Self::new(self.w, self.w, self.x, self.w)
    }

    pub fn wwyx(self) -> Self {
        Self::new(self.w, self.w, self.y, self.x)
    }

    pub fn wwyy(self) -> Self {
        Self::new(self.w, self.w, self.y, self.y)
    }

    pub fn wwyz(self) -> Self {
        Self::new(self.w, self.w, self.y, self.z)
    }

    pub fn wwyw(self) -> Self {
        Self::new(self.w, self.w, self.y, self.w)
    }

    pub fn wwzx(self) -> Self {
        Self::new(self.w, self.w, self.z, self.x)
    }

    pub fn wwzy(self) -> Self {
        Self::new(self.w, self.w, self.z, self.y)
    }

    pub fn wwzz(self) -> Self {
        Self::new(self.w, self.w, self.z, self.z)
    }

    pub fn wwzw(self) -> Self {
        Self::new(self.w, self.w, self.z, self.w)
    }

    pub fn wwwx(self) -> Self {
        Self::new(self.w, self.w, self.w, self.x)
    }

    pub fn wwwy(self) -> Self {
        Self::new(self.w, self.w, self.w, self.y)
    }

    pub fn wwwz(self) -> Self {
        Self::new(self.w, self.w, self.w, self.z)
    }

    pub fn wwww(self) -> Self {
        Self::new(self.w, self.w, self.w, self.w)
    }
}
