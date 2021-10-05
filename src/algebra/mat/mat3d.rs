#![allow(dead_code)]

use std::{
    fmt::Display,
    ops::{Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Sub, SubAssign},
};

use crate::algebra::{vec3d, Mat2d, Mat3f, Mat4d, Vec3d};

#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Mat3d {
    x_axis: Vec3d,
    y_axis: Vec3d,
    z_axis: Vec3d,
}

pub fn mat3d(
    m00: f64,
    m01: f64,
    m02: f64,
    m10: f64,
    m11: f64,
    m12: f64,
    m20: f64,
    m21: f64,
    m22: f64,
) -> Mat3d {
    Mat3d::from_array([m00, m01, m02, m10, m11, m12, m20, m21, m22])
}

impl Display for Mat3d {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let (m00, m01, m02) = self.x_axis.to_tuple();
        let (m10, m11, m12) = self.y_axis.to_tuple();
        let (m20, m21, m22) = self.z_axis.to_tuple();
        write!(
            f,
            "Mat3d({}, {}, {} | {}, {}, {} | {}, {}, {})",
            m00, m01, m02, m10, m11, m12, m20, m21, m22
        )
    }
}

impl Default for Mat3d {
    fn default() -> Self {
        Self {
            x_axis: Vec3d::default(),
            y_axis: Vec3d::default(),
            z_axis: Vec3d::default(),
        }
    }
}

impl Add<Mat3d> for Mat3d {
    type Output = Mat3d;

    fn add(self, rhs: Mat3d) -> Self::Output {
        Mat3d::new(
            self.x_axis + rhs.x_axis,
            self.y_axis + rhs.y_axis,
            self.z_axis + rhs.z_axis,
        )
    }
}

impl Add<f64> for Mat3d {
    type Output = Mat3d;

    fn add(self, rhs: f64) -> Self::Output {
        Mat3d::new(self.x_axis + rhs, self.y_axis + rhs, self.z_axis + rhs)
    }
}

impl Add<Mat3d> for f64 {
    type Output = Mat3d;

    fn add(self, rhs: Mat3d) -> Self::Output {
        Mat3d::new(self + rhs.x_axis, self + rhs.y_axis, self + rhs.z_axis)
    }
}

impl AddAssign<Mat3d> for Mat3d {
    fn add_assign(&mut self, rhs: Mat3d) {
        *self = *self + rhs;
    }
}

impl AddAssign<f64> for Mat3d {
    fn add_assign(&mut self, rhs: f64) {
        *self = *self + rhs;
    }
}

impl Sub<Mat3d> for Mat3d {
    type Output = Mat3d;

    fn sub(self, rhs: Mat3d) -> Self::Output {
        Mat3d::new(
            self.x_axis - rhs.x_axis,
            self.y_axis - rhs.y_axis,
            self.z_axis - rhs.z_axis,
        )
    }
}

impl Sub<f64> for Mat3d {
    type Output = Mat3d;

    fn sub(self, rhs: f64) -> Self::Output {
        Mat3d::new(self.x_axis - rhs, self.y_axis - rhs, self.z_axis - rhs)
    }
}

impl Sub<Mat3d> for f64 {
    type Output = Mat3d;

    fn sub(self, rhs: Mat3d) -> Self::Output {
        Mat3d::new(self - rhs.x_axis, self - rhs.y_axis, self - rhs.z_axis)
    }
}

impl SubAssign<Mat3d> for Mat3d {
    fn sub_assign(&mut self, rhs: Mat3d) {
        *self = *self - rhs;
    }
}

impl SubAssign<f64> for Mat3d {
    fn sub_assign(&mut self, rhs: f64) {
        *self = *self - rhs;
    }
}

impl Mul<Mat3d> for Mat3d {
    type Output = Mat3d;

    fn mul(self, rhs: Mat3d) -> Self::Output {
        let m00 = self.x_axis.dot(vec3d(rhs[0][0], rhs[1][0], rhs[2][0]));
        let m01 = self.x_axis.dot(vec3d(rhs[0][1], rhs[1][1], rhs[2][1]));
        let m02 = self.x_axis.dot(vec3d(rhs[0][2], rhs[1][2], rhs[2][2]));

        let m10 = self.y_axis.dot(vec3d(rhs[0][0], rhs[1][0], rhs[2][0]));
        let m11 = self.y_axis.dot(vec3d(rhs[0][1], rhs[1][1], rhs[2][1]));
        let m12 = self.y_axis.dot(vec3d(rhs[0][2], rhs[1][2], rhs[2][2]));

        let m20 = self.z_axis.dot(vec3d(rhs[0][0], rhs[1][0], rhs[2][0]));
        let m21 = self.z_axis.dot(vec3d(rhs[0][1], rhs[1][1], rhs[2][1]));
        let m22 = self.z_axis.dot(vec3d(rhs[0][2], rhs[1][2], rhs[2][2]));

        Mat3d::new(
            vec3d(m00, m01, m02),
            vec3d(m10, m11, m12),
            vec3d(m20, m21, m22),
        )
    }
}

impl Mul<Vec3d> for Mat3d {
    type Output = Vec3d;

    fn mul(self, rhs: Vec3d) -> Self::Output {
        let v0 = self.x_axis.dot(rhs);
        let v1 = self.y_axis.dot(rhs);
        let v2 = self.z_axis.dot(rhs);
        vec3d(v0, v1, v2)
    }
}

impl Mul<f64> for Mat3d {
    type Output = Mat3d;

    fn mul(self, rhs: f64) -> Self::Output {
        Mat3d::new(self.x_axis * rhs, self.y_axis * rhs, self.z_axis * rhs)
    }
}

impl Mul<Mat3d> for f64 {
    type Output = Mat3d;

    fn mul(self, rhs: Mat3d) -> Self::Output {
        Mat3d::new(self * rhs.x_axis, self * rhs.y_axis, self * rhs.z_axis)
    }
}

impl MulAssign<Mat3d> for Mat3d {
    fn mul_assign(&mut self, rhs: Mat3d) {
        *self = *self * rhs;
    }
}

impl MulAssign<f64> for Mat3d {
    fn mul_assign(&mut self, rhs: f64) {
        *self = *self * rhs;
    }
}

impl Div<f64> for Mat3d {
    type Output = Mat3d;

    fn div(self, rhs: f64) -> Self::Output {
        Mat3d::new(self.x_axis / rhs, self.y_axis / rhs, self.z_axis / rhs)
    }
}

impl DivAssign<f64> for Mat3d {
    fn div_assign(&mut self, rhs: f64) {
        *self = *self / rhs;
    }
}

impl Index<usize> for Mat3d {
    type Output = Vec3d;

    fn index(&self, index: usize) -> &Self::Output {
        match index {
            0 => &self.x_axis,
            1 => &self.y_axis,
            2 => &self.z_axis,
            _ => panic!("`rmath::algebra::Mat3d::index`: index out of bounds."),
        }
    }
}

impl IndexMut<usize> for Mat3d {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        match index {
            0 => &mut self.x_axis,
            1 => &mut self.y_axis,
            2 => &mut self.z_axis,
            _ => panic!("`rmath::algebra::Mat3d::index_mut`: index out of bounds."),
        }
    }
}

impl Mat3d {
    pub fn new(x_axis: Vec3d, y_axis: Vec3d, z_axis: Vec3d) -> Self {
        Self {
            x_axis,
            y_axis,
            z_axis,
        }
    }

    pub fn zero() -> Self {
        Self::new(
            vec3d(0.0, 0.0, 0.0),
            vec3d(0.0, 0.0, 0.0),
            vec3d(0.0, 0.0, 0.0),
        )
    }

    pub fn identity() -> Self {
        Self::new(
            vec3d(1.0, 0.0, 0.0),
            vec3d(0.0, 1.0, 0.0),
            vec3d(0.0, 0.0, 1.0),
        )
    }
}

impl Mat3d {
    pub fn col(&self, index: usize) -> Vec3d {
        match index {
            0 => vec3d(self[0][0], self[1][0], self[2][0]),
            1 => vec3d(self[0][1], self[1][1], self[2][1]),
            2 => vec3d(self[0][2], self[1][2], self[2][2]),
            _ => panic!("`rmath::algebra::Mat3d::col`: index out of bounds."),
        }
    }

    pub fn row(&self, index: usize) -> Vec3d {
        match index {
            0 => self.x_axis,
            1 => self.y_axis,
            2 => self.z_axis,
            _ => panic!("`rmath::algebra::Mat3d::row`: index out of bounds."),
        }
    }

    pub fn is_nan(&self) -> bool {
        self.x_axis.is_nan() || self.y_axis.is_nan() || self.z_axis.is_nan()
    }

    pub fn is_infinite(&self) -> bool {
        self.x_axis.is_infinite() || self.y_axis.is_infinite() || self.z_axis.is_infinite()
    }

    pub fn is_finite(&self) -> bool {
        self.x_axis.is_finite() && self.y_axis.is_finite() && self.z_axis.is_finite()
    }

    pub fn transpose(&self) -> Self {
        let (m00, m01, m02) = self.x_axis.to_tuple();
        let (m10, m11, m12) = self.y_axis.to_tuple();
        let (m20, m21, m22) = self.z_axis.to_tuple();
        Self::new(
            vec3d(m00, m10, m20),
            vec3d(m01, m11, m21),
            vec3d(m02, m12, m22),
        )
    }

    pub fn determinant(&self) -> f64 {
        let (m00, m01, m02) = self.x_axis.to_tuple();
        let (m10, m11, m12) = self.y_axis.to_tuple();
        let (m20, m21, m22) = self.z_axis.to_tuple();

        m00 * m11 * m22 + m01 * m12 * m20 + m02 * m10 * m21
            - m00 * m12 * m21
            - m01 * m10 * m22
            - m02 * m11 * m20
    }

    pub fn inverse(&self) -> Self {
        let det = self.determinant();
        ruby_assert!(!(det.abs() < f64::EPSILON));

        let inv_det = det.recip();
        let (m00, m01, m02) = self.x_axis.to_tuple();
        let (m10, m11, m12) = self.y_axis.to_tuple();
        let (m20, m21, m22) = self.z_axis.to_tuple();

        let a00 = m11 * m22 - m12 * m21;
        let a01 = -(m10 * m22 - m12 * m20);
        let a02 = m10 * m21 - m11 * m20;

        let a10 = -(m01 * m22 - m02 * m21);
        let a11 = m00 * m22 - m02 * m20;
        let a12 = -(m00 * m21 - m01 * m20);

        let a20 = m01 * m12 - m02 * m11;
        let a21 = -(m00 * m12 - m02 * m10);
        let a22 = m00 * m11 - m01 * m10;

        Self::new(
            vec3d(a00, a10, a20),
            vec3d(a01, a11, a21),
            vec3d(a02, a12, a22),
        )
        .mul(inv_det)
    }

    pub fn try_inverse(&self) -> Option<Self> {
        let det = self.determinant();
        if det.abs() < f64::EPSILON {
            return None;
        }

        let inv_det = det.recip();
        let (m00, m01, m02) = self.x_axis.to_tuple();
        let (m10, m11, m12) = self.y_axis.to_tuple();
        let (m20, m21, m22) = self.z_axis.to_tuple();

        let a00 = m11 * m22 - m12 * m21;
        let a01 = -(m10 * m22 - m12 * m20);
        let a02 = m10 * m21 - m11 * m20;

        let a10 = -(m01 * m22 - m02 * m21);
        let a11 = m00 * m22 - m02 * m20;
        let a12 = -(m00 * m21 - m01 * m20);

        let a20 = m01 * m12 - m02 * m11;
        let a21 = -(m00 * m12 - m02 * m10);
        let a22 = m00 * m11 - m01 * m10;

        Some(
            Self::new(
                vec3d(a00, a10, a20),
                vec3d(a01, a11, a21),
                vec3d(a02, a12, a22),
            )
            .mul(inv_det),
        )
    }
}

impl From<Mat2d> for Mat3d {
    fn from(m: Mat2d) -> Self {
        mat3d(m[0][0], m[0][1], 0.0, m[1][0], m[1][1], 0.0, 0.0, 0.0, 1.0)
    }
}

impl From<Mat4d> for Mat3d {
    fn from(m: Mat4d) -> Self {
        mat3d(
            m[0][0], m[0][1], m[0][2], m[1][0], m[1][1], m[1][2], m[2][0], m[2][1], m[2][2],
        )
    }
}

impl Mat3d {
    pub fn from_array(m: [f64; 9]) -> Self {
        Self::new(
            vec3d(m[0], m[1], m[2]),
            vec3d(m[3], m[4], m[5]),
            vec3d(m[6], m[7], m[8]),
        )
    }

    pub fn from_array_2d(m: [[f64; 3]; 3]) -> Self {
        Self::new(
            vec3d(m[0][0], m[0][1], m[0][2]),
            vec3d(m[1][0], m[1][1], m[1][2]),
            vec3d(m[2][0], m[2][1], m[2][2]),
        )
    }

    pub fn from_diagonal(diagonal: Vec3d) -> Self {
        Self::new(
            vec3d(diagonal.x(), 0.0, 0.0),
            vec3d(0.0, diagonal.y(), 0.0),
            vec3d(0.0, 0.0, diagonal.z()),
        )
    }
}

impl Mat3d {
    pub fn to_array(self) -> [f64; 9] {
        [
            self[0][0], self[0][1], self[0][2], self[1][0], self[1][1], self[1][2], self[2][0],
            self[2][1], self[2][2],
        ]
    }

    pub fn to_array_2d(self) -> [[f64; 3]; 3] {
        [
            [self[0][0], self[0][1], self[0][2]],
            [self[1][0], self[1][1], self[1][2]],
            [self[2][0], self[2][1], self[2][2]],
        ]
    }

    pub fn to_mat2d(self) -> Mat2d {
        Mat2d::from(self)
    }

    pub fn to_mat4d(self) -> Mat4d {
        Mat4d::from(self)
    }

    pub fn to_mat3f(self) -> Mat3f {
        Mat3f::new(
            self.x_axis.to_vec3f(),
            self.y_axis.to_vec3f(),
            self.z_axis.to_vec3f(),
        )
    }
}
