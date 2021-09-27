#![allow(dead_code)]

use std::{
    fmt::Display,
    ops::{Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign},
};

use super::{vec3d, Vec2f, Vec3d, Vec4f};

#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Vec3f {
    x: f32,
    y: f32,
    z: f32,
}

pub fn vec3f(x: f32, y: f32, z: f32) -> Vec3f {
    Vec3f::new(x, y, z)
}

impl Display for Vec3f {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Vec3f(x: {}, y: {}, z: {})", self.x, self.y, self.z)
    }
}

impl Default for Vec3f {
    fn default() -> Self {
        Self::new(0.0, 0.0, 0.0)
    }
}

impl Add<Vec3f> for Vec3f {
    type Output = Vec3f;

    fn add(self, rhs: Vec3f) -> Self::Output {
        Vec3f::new(self.x + rhs.x, self.y + rhs.y, self.z + rhs.z)
    }
}

impl Add<f32> for Vec3f {
    type Output = Vec3f;

    fn add(self, rhs: f32) -> Self::Output {
        Vec3f::new(self.x + rhs, self.y + rhs, self.z + rhs)
    }
}

impl Add<Vec3f> for f32 {
    type Output = Vec3f;

    fn add(self, rhs: Vec3f) -> Self::Output {
        Vec3f::new(self + rhs.x, self + rhs.y, self + rhs.z)
    }
}

impl AddAssign<Vec3f> for Vec3f {
    fn add_assign(&mut self, rhs: Vec3f) {
        self.x += rhs.x;
        self.y += rhs.y;
        self.z += rhs.z;
    }
}

impl AddAssign<f32> for Vec3f {
    fn add_assign(&mut self, rhs: f32) {
        self.x += rhs;
        self.y += rhs;
        self.z += rhs;
    }
}

impl Sub<Vec3f> for Vec3f {
    type Output = Vec3f;

    fn sub(self, rhs: Vec3f) -> Self::Output {
        Vec3f::new(self.x - rhs.x, self.y - rhs.y, self.z - rhs.z)
    }
}

impl Sub<f32> for Vec3f {
    type Output = Vec3f;

    fn sub(self, rhs: f32) -> Self::Output {
        Vec3f::new(self.x - rhs, self.y - rhs, self.z - rhs)
    }
}

impl Sub<Vec3f> for f32 {
    type Output = Vec3f;

    fn sub(self, rhs: Vec3f) -> Self::Output {
        Vec3f::new(self - rhs.x, self - rhs.y, self - rhs.z)
    }
}

impl SubAssign<Vec3f> for Vec3f {
    fn sub_assign(&mut self, rhs: Vec3f) {
        self.x -= rhs.x;
        self.y -= rhs.y;
        self.z -= rhs.z;
    }
}

impl SubAssign<f32> for Vec3f {
    fn sub_assign(&mut self, rhs: f32) {
        self.x -= rhs;
        self.y -= rhs;
        self.z -= rhs;
    }
}

impl Mul<Vec3f> for Vec3f {
    type Output = Vec3f;

    fn mul(self, rhs: Vec3f) -> Self::Output {
        Vec3f::new(self.x * rhs.x, self.y * rhs.y, self.z * rhs.z)
    }
}

impl Mul<f32> for Vec3f {
    type Output = Vec3f;

    fn mul(self, rhs: f32) -> Self::Output {
        Vec3f::new(self.x * rhs, self.y * rhs, self.z * rhs)
    }
}

impl Mul<Vec3f> for f32 {
    type Output = Vec3f;

    fn mul(self, rhs: Vec3f) -> Self::Output {
        Vec3f::new(self * rhs.x, self * rhs.y, self * rhs.z)
    }
}

impl MulAssign<Vec3f> for Vec3f {
    fn mul_assign(&mut self, rhs: Vec3f) {
        self.x *= rhs.x;
        self.y *= rhs.y;
        self.z *= rhs.z;
    }
}

impl MulAssign<f32> for Vec3f {
    fn mul_assign(&mut self, rhs: f32) {
        self.x *= rhs;
        self.y *= rhs;
        self.z *= rhs;
    }
}

impl Div<Vec3f> for Vec3f {
    type Output = Vec3f;

    fn div(self, rhs: Vec3f) -> Self::Output {
        Vec3f::new(self.x / rhs.x, self.y / rhs.y, self.z / rhs.z)
    }
}

impl Div<f32> for Vec3f {
    type Output = Vec3f;

    fn div(self, rhs: f32) -> Self::Output {
        Vec3f::new(self.x / rhs, self.y / rhs, self.z / rhs)
    }
}

impl Div<Vec3f> for f32 {
    type Output = Vec3f;

    fn div(self, rhs: Vec3f) -> Self::Output {
        Vec3f::new(self / rhs.x, self / rhs.y, self / rhs.z)
    }
}

impl DivAssign<Vec3f> for Vec3f {
    fn div_assign(&mut self, rhs: Vec3f) {
        self.x /= rhs.x;
        self.y /= rhs.y;
        self.z /= rhs.z;
    }
}

impl DivAssign<f32> for Vec3f {
    fn div_assign(&mut self, rhs: f32) {
        self.x /= rhs;
        self.y /= rhs;
        self.z /= rhs;
    }
}

impl Neg for Vec3f {
    type Output = Vec3f;

    fn neg(self) -> Self::Output {
        Self::new(-self.x, -self.y, -self.z)
    }
}

impl Index<usize> for Vec3f {
    type Output = f32;

    fn index(&self, index: usize) -> &Self::Output {
        match index {
            0 => &self.x,
            1 => &self.y,
            2 => &self.z,
            _ => panic!("`rmath::algebra::Vec3f::index`: index out of bounds."),
        }
    }
}

impl IndexMut<usize> for Vec3f {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        match index {
            0 => &mut self.x,
            1 => &mut self.y,
            2 => &mut self.z,
            _ => panic!("`rmath::algebra::Vec3f::index_mut`: index out of bounds."),
        }
    }
}

impl From<f32> for Vec3f {
    fn from(v: f32) -> Self {
        Self::new(v, v, v)
    }
}

impl From<(f32, f32, f32)> for Vec3f {
    fn from(v: (f32, f32, f32)) -> Self {
        let (x, y, z) = v;
        Self::new(x, y, z)
    }
}

impl From<(Vec2f, f32)> for Vec3f {
    fn from(v: (Vec2f, f32)) -> Self {
        let (xy, z) = v;
        let (x, y) = xy.to_tuple();
        Self::new(x, y, z)
    }
}

impl From<Vec4f> for Vec3f {
    fn from(v: Vec4f) -> Self {
        v.xyz()
    }
}

impl Vec3f {
    pub fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    pub fn one() -> Self {
        Self::new(1.0, 1.0, 1.0)
    }

    pub fn zero() -> Self {
        Self::new(0.0, 0.0, 0.0)
    }
}

impl Vec3f {
    pub fn floor(self) -> Self {
        Self::new(self.x.floor(), self.y.floor(), self.z.floor())
    }

    pub fn ceil(self) -> Self {
        Self::new(self.x.ceil(), self.y.ceil(), self.z.ceil())
    }

    pub fn round(self) -> Self {
        Self::new(self.x.round(), self.y.round(), self.z.round())
    }

    pub fn trunc(self) -> Self {
        Self::new(self.x.trunc(), self.y.trunc(), self.z.trunc())
    }

    pub fn fract(self) -> Self {
        Self::new(self.x.fract(), self.y.fract(), self.z.fract())
    }

    pub fn abs(self) -> Self {
        Self::new(self.x.abs(), self.y.abs(), self.z.abs())
    }

    pub fn signum(self) -> Self {
        Self::new(self.x.signum(), self.y.signum(), self.z.signum())
    }

    pub fn powf(self, n: f32) -> Self {
        Self::new(self.x.powf(n), self.y.powf(n), self.z.powf(n))
    }

    pub fn sqrt(self) -> Self {
        Self::new(self.x.sqrt(), self.y.sqrt(), self.z.sqrt())
    }

    pub fn exp(self) -> Self {
        Self::new(self.x.exp(), self.y.exp(), self.z.exp())
    }

    pub fn exp2(self) -> Self {
        Self::new(self.x.exp2(), self.y.exp2(), self.z.exp2())
    }

    pub fn ln(self) -> Self {
        Self::new(self.x.ln(), self.y.ln(), self.z.ln())
    }

    pub fn log(self, base: f32) -> Self {
        Self::new(self.x.log(base), self.y.log(base), self.z.log(base))
    }

    pub fn log2(self) -> Self {
        Self::new(self.x.log2(), self.y.log2(), self.z.log2())
    }

    pub fn log10(self) -> Self {
        Self::new(self.x.log10(), self.y.log10(), self.z.log10())
    }

    pub fn cbrt(self) -> Self {
        Self::new(self.x.cbrt(), self.y.cbrt(), self.z.cbrt())
    }

    pub fn sin(self) -> Self {
        Self::new(self.x.sin(), self.y.sin(), self.z.sin())
    }

    pub fn cos(self) -> Self {
        Self::new(self.x.cos(), self.y.cos(), self.z.cos())
    }

    pub fn tan(self) -> Self {
        Self::new(self.x.tan(), self.y.tan(), self.z.tan())
    }

    pub fn sin_cos(self) -> (Self, Self) {
        (self.sin(), self.cos())
    }

    pub fn lerp(self, rhs: Self, s: f32) -> Self {
        self + (rhs - self) * s
    }

    pub fn lerp_vec(self, rhs: Self, s: Self) -> Self {
        self + (rhs - self) * s
    }

    pub fn is_nan(self) -> bool {
        self.x.is_nan() || self.y.is_nan() || self.z.is_nan()
    }

    pub fn is_infinite(self) -> bool {
        self.x.is_infinite() || self.y.is_infinite() || self.z.is_infinite()
    }

    pub fn is_finite(self) -> bool {
        self.x.is_finite() && self.y.is_finite() && self.z.is_finite()
    }

    pub fn recip(self) -> Self {
        Self::new(self.x.recip(), self.y.recip(), self.z.recip())
    }

    pub fn max(self, rhs: Self) -> Self {
        Self::new(self.x.max(rhs.x), self.y.max(rhs.y), self.z.max(rhs.z))
    }

    pub fn min(self, rhs: Self) -> Self {
        Self::new(self.x.min(rhs.x), self.y.min(rhs.y), self.z.min(rhs.z))
    }

    pub fn clamp(self, min: Self, max: Self) -> Self {
        ruby_assert!(min.x <= max.x);
        ruby_assert!(min.y <= max.y);
        ruby_assert!(min.z <= max.z);

        self.min(max).max(min)
    }

    pub fn saturate(self) -> Self {
        self.clamp(Self::zero(), Self::one())
    }

    pub fn min_element(self) -> f32 {
        self.x.min(self.y).min(self.z)
    }

    pub fn max_element(self) -> f32 {
        self.x.max(self.y).max(self.z)
    }
}

impl Vec3f {
    pub fn dot(self, rhs: Self) -> f32 {
        self.x * rhs.x + self.y * rhs.y + self.z * rhs.z
    }

    pub fn cross(self, rhs: Self) -> Self {
        Self::new(
            self.y * rhs.z - self.z * rhs.y,
            self.z * rhs.x - self.x * rhs.z,
            self.x * rhs.y - self.y * rhs.x,
        )
    }

    pub fn length(self) -> f32 {
        self.dot(self).sqrt()
    }

    pub fn length_squared(self) -> f32 {
        self.dot(self)
    }

    pub fn length_recip(self) -> f32 {
        self.length().recip()
    }

    pub fn distance(self, rhs: Self) -> f32 {
        (rhs - self).length()
    }

    pub fn distance_squared(self, rhs: Self) -> f32 {
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
        (self.length_squared() - 1.0f32).abs() < f32::EPSILON
    }

    pub fn angle_between(self, rhs: Self) -> f32 {
        self.dot(rhs)
            .div(self.length_squared().mul(rhs.length_squared()).sqrt())
            .acos()
    }
}

impl Vec3f {
    pub fn to_array(self) -> [f32; 3] {
        [self.x, self.y, self.z]
    }

    pub fn to_tuple(self) -> (f32, f32, f32) {
        (self.x, self.y, self.z)
    }

    pub fn as_vec3d(self) -> Vec3d {
        vec3d(self.x as f64, self.y as f64, self.z as f64)
    }

    pub fn to_homogeneous_coord_point(self) -> Vec4f {
        Vec4f::from((self, 1.0))
    }

    pub fn to_homogeneous_coord_vector(self) -> Vec4f {
        Vec4f::from((self, 0.0))
    }
}

impl Vec3f {
    pub fn x(self) -> f32 {
        self.x
    }

    pub fn y(self) -> f32 {
        self.y
    }

    pub fn z(self) -> f32 {
        self.z
    }

    pub fn xx(self) -> Vec2f {
        Vec2f::new(self.x, self.x)
    }

    pub fn xy(self) -> Vec2f {
        Vec2f::new(self.x, self.y)
    }

    pub fn xz(self) -> Vec2f {
        Vec2f::new(self.x, self.z)
    }

    pub fn yx(self) -> Vec2f {
        Vec2f::new(self.y, self.x)
    }

    pub fn yy(self) -> Vec2f {
        Vec2f::new(self.y, self.y)
    }

    pub fn yz(self) -> Vec2f {
        Vec2f::new(self.y, self.z)
    }

    pub fn zx(self) -> Vec2f {
        Vec2f::new(self.z, self.x)
    }

    pub fn zy(self) -> Vec2f {
        Vec2f::new(self.z, self.y)
    }

    pub fn zz(self) -> Vec2f {
        Vec2f::new(self.z, self.z)
    }

    pub fn xxx(self) -> Self {
        Self::new(self.x, self.x, self.x)
    }

    pub fn xxy(self) -> Self {
        Self::new(self.x, self.x, self.y)
    }

    pub fn xxz(self) -> Self {
        Self::new(self.x, self.x, self.z)
    }

    pub fn xyx(self) -> Self {
        Self::new(self.x, self.y, self.x)
    }

    pub fn xyy(self) -> Self {
        Self::new(self.x, self.y, self.y)
    }

    pub fn xyz(self) -> Self {
        Self::new(self.x, self.y, self.z)
    }

    pub fn xzx(self) -> Self {
        Self::new(self.x, self.z, self.x)
    }

    pub fn xzy(self) -> Self {
        Self::new(self.x, self.z, self.y)
    }

    pub fn xzz(self) -> Self {
        Self::new(self.x, self.z, self.z)
    }

    pub fn yxx(self) -> Self {
        Self::new(self.y, self.x, self.x)
    }

    pub fn yxy(self) -> Self {
        Self::new(self.y, self.x, self.y)
    }

    pub fn yxz(self) -> Self {
        Self::new(self.y, self.x, self.z)
    }

    pub fn yyx(self) -> Self {
        Self::new(self.y, self.y, self.x)
    }

    pub fn yyy(self) -> Self {
        Self::new(self.y, self.y, self.y)
    }

    pub fn yyz(self) -> Self {
        Self::new(self.y, self.y, self.z)
    }

    pub fn yzx(self) -> Self {
        Self::new(self.y, self.z, self.x)
    }

    pub fn yzy(self) -> Self {
        Self::new(self.y, self.z, self.y)
    }

    pub fn yzz(self) -> Self {
        Self::new(self.y, self.z, self.z)
    }

    pub fn zxx(self) -> Self {
        Self::new(self.z, self.x, self.x)
    }

    pub fn zxy(self) -> Self {
        Self::new(self.z, self.x, self.y)
    }

    pub fn zxz(self) -> Self {
        Self::new(self.z, self.x, self.z)
    }

    pub fn zyx(self) -> Self {
        Self::new(self.z, self.y, self.x)
    }

    pub fn zyy(self) -> Self {
        Self::new(self.z, self.y, self.y)
    }

    pub fn zyz(self) -> Self {
        Self::new(self.z, self.y, self.z)
    }

    pub fn zzx(self) -> Self {
        Self::new(self.z, self.z, self.x)
    }

    pub fn zzy(self) -> Self {
        Self::new(self.z, self.z, self.y)
    }

    pub fn zzz(self) -> Self {
        Self::new(self.z, self.z, self.z)
    }

    pub fn xxxx(self) -> Vec4f {
        Vec4f::new(self.x, self.x, self.x, self.x)
    }

    pub fn xxxy(self) -> Vec4f {
        Vec4f::new(self.x, self.x, self.x, self.y)
    }

    pub fn xxxz(self) -> Vec4f {
        Vec4f::new(self.x, self.x, self.x, self.z)
    }

    pub fn xxyx(self) -> Vec4f {
        Vec4f::new(self.x, self.x, self.y, self.x)
    }

    pub fn xxyy(self) -> Vec4f {
        Vec4f::new(self.x, self.x, self.y, self.y)
    }

    pub fn xxyz(self) -> Vec4f {
        Vec4f::new(self.x, self.x, self.y, self.z)
    }

    pub fn xxzx(self) -> Vec4f {
        Vec4f::new(self.x, self.x, self.z, self.x)
    }

    pub fn xxzy(self) -> Vec4f {
        Vec4f::new(self.x, self.x, self.z, self.y)
    }

    pub fn xxzz(self) -> Vec4f {
        Vec4f::new(self.x, self.x, self.z, self.z)
    }

    pub fn xyxx(self) -> Vec4f {
        Vec4f::new(self.x, self.y, self.x, self.x)
    }

    pub fn xyxy(self) -> Vec4f {
        Vec4f::new(self.x, self.y, self.x, self.y)
    }

    pub fn xyxz(self) -> Vec4f {
        Vec4f::new(self.x, self.y, self.x, self.z)
    }

    pub fn xyyx(self) -> Vec4f {
        Vec4f::new(self.x, self.y, self.y, self.x)
    }

    pub fn xyyy(self) -> Vec4f {
        Vec4f::new(self.x, self.y, self.y, self.y)
    }

    pub fn xyyz(self) -> Vec4f {
        Vec4f::new(self.x, self.y, self.y, self.z)
    }

    pub fn xyzx(self) -> Vec4f {
        Vec4f::new(self.x, self.y, self.z, self.x)
    }

    pub fn xyzy(self) -> Vec4f {
        Vec4f::new(self.x, self.y, self.z, self.y)
    }

    pub fn xyzz(self) -> Vec4f {
        Vec4f::new(self.x, self.y, self.z, self.z)
    }

    pub fn xzxx(self) -> Vec4f {
        Vec4f::new(self.x, self.z, self.x, self.x)
    }

    pub fn xzxy(self) -> Vec4f {
        Vec4f::new(self.x, self.z, self.x, self.y)
    }

    pub fn xzxz(self) -> Vec4f {
        Vec4f::new(self.x, self.z, self.x, self.z)
    }

    pub fn xzyx(self) -> Vec4f {
        Vec4f::new(self.x, self.z, self.y, self.x)
    }

    pub fn xzyy(self) -> Vec4f {
        Vec4f::new(self.x, self.z, self.y, self.y)
    }

    pub fn xzyz(self) -> Vec4f {
        Vec4f::new(self.x, self.z, self.y, self.z)
    }

    pub fn xzzx(self) -> Vec4f {
        Vec4f::new(self.x, self.z, self.z, self.x)
    }

    pub fn xzzy(self) -> Vec4f {
        Vec4f::new(self.x, self.z, self.z, self.y)
    }

    pub fn xzzz(self) -> Vec4f {
        Vec4f::new(self.x, self.z, self.z, self.z)
    }

    pub fn yxxx(self) -> Vec4f {
        Vec4f::new(self.y, self.x, self.x, self.x)
    }

    pub fn yxxy(self) -> Vec4f {
        Vec4f::new(self.y, self.x, self.x, self.y)
    }

    pub fn yxxz(self) -> Vec4f {
        Vec4f::new(self.y, self.x, self.x, self.z)
    }

    pub fn yxyx(self) -> Vec4f {
        Vec4f::new(self.y, self.x, self.y, self.x)
    }

    pub fn yxyy(self) -> Vec4f {
        Vec4f::new(self.y, self.x, self.y, self.y)
    }

    pub fn yxyz(self) -> Vec4f {
        Vec4f::new(self.y, self.x, self.y, self.z)
    }

    pub fn yxzx(self) -> Vec4f {
        Vec4f::new(self.y, self.x, self.z, self.x)
    }

    pub fn yxzy(self) -> Vec4f {
        Vec4f::new(self.y, self.x, self.z, self.y)
    }

    pub fn yxzz(self) -> Vec4f {
        Vec4f::new(self.y, self.x, self.z, self.z)
    }

    pub fn yyxx(self) -> Vec4f {
        Vec4f::new(self.y, self.y, self.x, self.x)
    }

    pub fn yyxy(self) -> Vec4f {
        Vec4f::new(self.y, self.y, self.x, self.y)
    }

    pub fn yyxz(self) -> Vec4f {
        Vec4f::new(self.y, self.y, self.x, self.z)
    }

    pub fn yyyx(self) -> Vec4f {
        Vec4f::new(self.y, self.y, self.y, self.x)
    }

    pub fn yyyy(self) -> Vec4f {
        Vec4f::new(self.y, self.y, self.y, self.y)
    }

    pub fn yyyz(self) -> Vec4f {
        Vec4f::new(self.y, self.y, self.y, self.z)
    }

    pub fn yyzx(self) -> Vec4f {
        Vec4f::new(self.y, self.y, self.z, self.x)
    }

    pub fn yyzy(self) -> Vec4f {
        Vec4f::new(self.y, self.y, self.z, self.y)
    }

    pub fn yyzz(self) -> Vec4f {
        Vec4f::new(self.y, self.y, self.z, self.z)
    }

    pub fn yzxx(self) -> Vec4f {
        Vec4f::new(self.y, self.z, self.x, self.x)
    }

    pub fn yzxy(self) -> Vec4f {
        Vec4f::new(self.y, self.z, self.x, self.y)
    }

    pub fn yzxz(self) -> Vec4f {
        Vec4f::new(self.y, self.z, self.x, self.z)
    }

    pub fn yzyx(self) -> Vec4f {
        Vec4f::new(self.y, self.z, self.y, self.x)
    }

    pub fn yzyy(self) -> Vec4f {
        Vec4f::new(self.y, self.z, self.y, self.y)
    }

    pub fn yzyz(self) -> Vec4f {
        Vec4f::new(self.y, self.z, self.y, self.z)
    }

    pub fn yzzx(self) -> Vec4f {
        Vec4f::new(self.y, self.z, self.z, self.x)
    }

    pub fn yzzy(self) -> Vec4f {
        Vec4f::new(self.y, self.z, self.z, self.y)
    }

    pub fn yzzz(self) -> Vec4f {
        Vec4f::new(self.y, self.z, self.z, self.z)
    }

    pub fn zxxx(self) -> Vec4f {
        Vec4f::new(self.z, self.x, self.x, self.x)
    }

    pub fn zxxy(self) -> Vec4f {
        Vec4f::new(self.z, self.x, self.x, self.y)
    }

    pub fn zxxz(self) -> Vec4f {
        Vec4f::new(self.z, self.x, self.x, self.z)
    }

    pub fn zxyx(self) -> Vec4f {
        Vec4f::new(self.z, self.x, self.y, self.x)
    }

    pub fn zxyy(self) -> Vec4f {
        Vec4f::new(self.z, self.x, self.y, self.y)
    }

    pub fn zxyz(self) -> Vec4f {
        Vec4f::new(self.z, self.x, self.y, self.z)
    }

    pub fn zxzx(self) -> Vec4f {
        Vec4f::new(self.z, self.x, self.z, self.x)
    }

    pub fn zxzy(self) -> Vec4f {
        Vec4f::new(self.z, self.x, self.z, self.y)
    }

    pub fn zxzz(self) -> Vec4f {
        Vec4f::new(self.z, self.x, self.z, self.z)
    }

    pub fn zyxx(self) -> Vec4f {
        Vec4f::new(self.z, self.y, self.x, self.x)
    }

    pub fn zyxy(self) -> Vec4f {
        Vec4f::new(self.z, self.y, self.x, self.y)
    }

    pub fn zyxz(self) -> Vec4f {
        Vec4f::new(self.z, self.y, self.x, self.z)
    }

    pub fn zyyx(self) -> Vec4f {
        Vec4f::new(self.z, self.y, self.y, self.x)
    }

    pub fn zyyy(self) -> Vec4f {
        Vec4f::new(self.z, self.y, self.y, self.y)
    }

    pub fn zyyz(self) -> Vec4f {
        Vec4f::new(self.z, self.y, self.y, self.z)
    }

    pub fn zyzx(self) -> Vec4f {
        Vec4f::new(self.z, self.y, self.z, self.x)
    }

    pub fn zyzy(self) -> Vec4f {
        Vec4f::new(self.z, self.y, self.z, self.y)
    }

    pub fn zyzz(self) -> Vec4f {
        Vec4f::new(self.z, self.y, self.z, self.z)
    }

    pub fn zzxx(self) -> Vec4f {
        Vec4f::new(self.z, self.z, self.x, self.x)
    }

    pub fn zzxy(self) -> Vec4f {
        Vec4f::new(self.z, self.z, self.x, self.y)
    }

    pub fn zzxz(self) -> Vec4f {
        Vec4f::new(self.z, self.z, self.x, self.z)
    }

    pub fn zzyx(self) -> Vec4f {
        Vec4f::new(self.z, self.z, self.y, self.x)
    }

    pub fn zzyy(self) -> Vec4f {
        Vec4f::new(self.z, self.z, self.y, self.y)
    }

    pub fn zzyz(self) -> Vec4f {
        Vec4f::new(self.z, self.z, self.y, self.z)
    }

    pub fn zzzx(self) -> Vec4f {
        Vec4f::new(self.z, self.z, self.z, self.x)
    }

    pub fn zzzy(self) -> Vec4f {
        Vec4f::new(self.z, self.z, self.z, self.y)
    }

    pub fn zzzz(self) -> Vec4f {
        Vec4f::new(self.z, self.z, self.z, self.z)
    }
}
