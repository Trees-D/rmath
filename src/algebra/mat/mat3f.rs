#![allow(dead_code)]

use std::{
    fmt::Display,
    ops::{Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Sub, SubAssign},
};

use crate::algebra::{vec3f, Mat2f, Mat3d, Mat4f, Vec3f};

#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Mat3f {
    x_axis: Vec3f,
    y_axis: Vec3f,
    z_axis: Vec3f,
}

pub fn mat3f(
    m00: f32,
    m01: f32,
    m02: f32,
    m10: f32,
    m11: f32,
    m12: f32,
    m20: f32,
    m21: f32,
    m22: f32,
) -> Mat3f {
    Mat3f::from_array([m00, m01, m02, m10, m11, m12, m20, m21, m22])
}

impl Display for Mat3f {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let (m00, m01, m02) = self.x_axis.to_tuple();
        let (m10, m11, m12) = self.y_axis.to_tuple();
        let (m20, m21, m22) = self.z_axis.to_tuple();
        write!(
            f,
            "Mat3f({}, {}, {} | {}, {}, {} | {}, {}, {})",
            m00, m01, m02, m10, m11, m12, m20, m21, m22
        )
    }
}

impl Default for Mat3f {
    fn default() -> Self {
        Self {
            x_axis: Vec3f::default(),
            y_axis: Vec3f::default(),
            z_axis: Vec3f::default(),
        }
    }
}

impl Add<Mat3f> for Mat3f {
    type Output = Mat3f;

    fn add(self, rhs: Mat3f) -> Self::Output {
        Mat3f::new(
            self.x_axis + rhs.x_axis,
            self.y_axis + rhs.y_axis,
            self.z_axis + rhs.z_axis,
        )
    }
}

impl Add<f32> for Mat3f {
    type Output = Mat3f;

    fn add(self, rhs: f32) -> Self::Output {
        Mat3f::new(self.x_axis + rhs, self.y_axis + rhs, self.z_axis + rhs)
    }
}

impl Add<Mat3f> for f32 {
    type Output = Mat3f;

    fn add(self, rhs: Mat3f) -> Self::Output {
        Mat3f::new(self + rhs.x_axis, self + rhs.y_axis, self + rhs.z_axis)
    }
}

impl AddAssign<Mat3f> for Mat3f {
    fn add_assign(&mut self, rhs: Mat3f) {
        *self = *self + rhs;
    }
}

impl AddAssign<f32> for Mat3f {
    fn add_assign(&mut self, rhs: f32) {
        *self = *self + rhs;
    }
}

impl Sub<Mat3f> for Mat3f {
    type Output = Mat3f;

    fn sub(self, rhs: Mat3f) -> Self::Output {
        Mat3f::new(
            self.x_axis - rhs.x_axis,
            self.y_axis - rhs.y_axis,
            self.z_axis - rhs.z_axis,
        )
    }
}

impl Sub<f32> for Mat3f {
    type Output = Mat3f;

    fn sub(self, rhs: f32) -> Self::Output {
        Mat3f::new(self.x_axis - rhs, self.y_axis - rhs, self.z_axis - rhs)
    }
}

impl Sub<Mat3f> for f32 {
    type Output = Mat3f;

    fn sub(self, rhs: Mat3f) -> Self::Output {
        Mat3f::new(self - rhs.x_axis, self - rhs.y_axis, self - rhs.z_axis)
    }
}

impl SubAssign<Mat3f> for Mat3f {
    fn sub_assign(&mut self, rhs: Mat3f) {
        *self = *self - rhs;
    }
}

impl SubAssign<f32> for Mat3f {
    fn sub_assign(&mut self, rhs: f32) {
        *self = *self - rhs;
    }
}

impl Mul<Mat3f> for Mat3f {
    type Output = Mat3f;

    fn mul(self, rhs: Mat3f) -> Self::Output {
        let m00 = self.x_axis.dot(vec3f(rhs[0][0], rhs[1][0], rhs[2][0]));
        let m01 = self.x_axis.dot(vec3f(rhs[0][1], rhs[1][1], rhs[2][1]));
        let m02 = self.x_axis.dot(vec3f(rhs[0][2], rhs[1][2], rhs[2][2]));

        let m10 = self.y_axis.dot(vec3f(rhs[0][0], rhs[1][0], rhs[2][0]));
        let m11 = self.y_axis.dot(vec3f(rhs[0][1], rhs[1][1], rhs[2][1]));
        let m12 = self.y_axis.dot(vec3f(rhs[0][2], rhs[1][2], rhs[2][2]));

        let m20 = self.z_axis.dot(vec3f(rhs[0][0], rhs[1][0], rhs[2][0]));
        let m21 = self.z_axis.dot(vec3f(rhs[0][1], rhs[1][1], rhs[2][1]));
        let m22 = self.z_axis.dot(vec3f(rhs[0][2], rhs[1][2], rhs[2][2]));

        Mat3f::new(
            vec3f(m00, m01, m02),
            vec3f(m10, m11, m12),
            vec3f(m20, m21, m22),
        )
    }
}

impl Mul<Vec3f> for Mat3f {
    type Output = Vec3f;

    fn mul(self, rhs: Vec3f) -> Self::Output {
        let v0 = self.x_axis.dot(rhs);
        let v1 = self.y_axis.dot(rhs);
        let v2 = self.z_axis.dot(rhs);
        vec3f(v0, v1, v2)
    }
}

impl Mul<f32> for Mat3f {
    type Output = Mat3f;

    fn mul(self, rhs: f32) -> Self::Output {
        Mat3f::new(self.x_axis * rhs, self.y_axis * rhs, self.z_axis * rhs)
    }
}

impl Mul<Mat3f> for f32 {
    type Output = Mat3f;

    fn mul(self, rhs: Mat3f) -> Self::Output {
        Mat3f::new(self * rhs.x_axis, self * rhs.y_axis, self * rhs.z_axis)
    }
}

impl MulAssign<Mat3f> for Mat3f {
    fn mul_assign(&mut self, rhs: Mat3f) {
        *self = *self * rhs;
    }
}

impl MulAssign<f32> for Mat3f {
    fn mul_assign(&mut self, rhs: f32) {
        *self = *self * rhs;
    }
}

impl Div<f32> for Mat3f {
    type Output = Mat3f;

    fn div(self, rhs: f32) -> Self::Output {
        Mat3f::new(self.x_axis / rhs, self.y_axis / rhs, self.z_axis / rhs)
    }
}

impl DivAssign<f32> for Mat3f {
    fn div_assign(&mut self, rhs: f32) {
        *self = *self / rhs;
    }
}

impl Index<usize> for Mat3f {
    type Output = Vec3f;

    fn index(&self, index: usize) -> &Self::Output {
        match index {
            0 => &self.x_axis,
            1 => &self.y_axis,
            2 => &self.z_axis,
            _ => panic!("`rmath::algebra::Mat3f::index`: index out of bounds."),
        }
    }
}

impl IndexMut<usize> for Mat3f {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        match index {
            0 => &mut self.x_axis,
            1 => &mut self.y_axis,
            2 => &mut self.z_axis,
            _ => panic!("`rmath::algebra::Mat3f::index_mut`: index out of bounds."),
        }
    }
}

impl Mat3f {
    pub fn new(x_axis: Vec3f, y_axis: Vec3f, z_axis: Vec3f) -> Self {
        Self {
            x_axis,
            y_axis,
            z_axis,
        }
    }

    pub fn zero() -> Self {
        Self::new(
            vec3f(0.0, 0.0, 0.0),
            vec3f(0.0, 0.0, 0.0),
            vec3f(0.0, 0.0, 0.0),
        )
    }

    pub fn identity() -> Self {
        Self::new(
            vec3f(1.0, 0.0, 0.0),
            vec3f(0.0, 1.0, 0.0),
            vec3f(0.0, 0.0, 1.0),
        )
    }
}

impl Mat3f {
    pub fn col(&self, index: usize) -> Vec3f {
        match index {
            0 => vec3f(self[0][0], self[1][0], self[2][0]),
            1 => vec3f(self[0][1], self[1][1], self[2][1]),
            2 => vec3f(self[0][2], self[1][2], self[2][2]),
            _ => panic!("`rmath::algebra::Mat3f::col`: index out of bounds."),
        }
    }

    pub fn row(&self, index: usize) -> Vec3f {
        match index {
            0 => self.x_axis,
            1 => self.y_axis,
            2 => self.z_axis,
            _ => panic!("`rmath::algebra::Mat3f::row`: index out of bounds."),
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
            vec3f(m00, m10, m20),
            vec3f(m01, m11, m21),
            vec3f(m02, m12, m22),
        )
    }

    pub fn determinant(&self) -> f32 {
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
        ruby_assert!(!(det.abs() < f32::EPSILON));

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
            vec3f(a00, a10, a20),
            vec3f(a01, a11, a21),
            vec3f(a02, a12, a22),
        )
        .mul(inv_det)
    }

    pub fn tray_inverse(&self) -> Option<Self> {
        let det = self.determinant();
        if det.abs() < f32::EPSILON {
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
                vec3f(a00, a10, a20),
                vec3f(a01, a11, a21),
                vec3f(a02, a12, a22),
            )
            .mul(inv_det),
        )
    }
}

impl From<Mat2f> for Mat3f {
    fn from(m: Mat2f) -> Self {
        mat3f(m[0][0], m[0][1], 0.0, m[1][0], m[1][1], 0.0, 0.0, 0.0, 1.0)
    }
}

impl From<Mat4f> for Mat3f {
    fn from(m: Mat4f) -> Self {
        mat3f(
            m[0][0], m[0][1], m[0][2], m[1][0], m[1][1], m[1][2], m[2][0], m[2][1], m[2][2],
        )
    }
}

impl Mat3f {
    pub fn from_array(m: [f32; 9]) -> Self {
        Self::new(
            vec3f(m[0], m[1], m[2]),
            vec3f(m[3], m[4], m[5]),
            vec3f(m[6], m[7], m[8]),
        )
    }

    pub fn from_array_2d(m: [[f32; 3]; 3]) -> Self {
        Self::new(
            vec3f(m[0][0], m[0][1], m[0][2]),
            vec3f(m[1][0], m[1][1], m[1][2]),
            vec3f(m[2][0], m[2][1], m[2][2]),
        )
    }

    pub fn from_diagonal(diagonal: Vec3f) -> Self {
        Self::new(
            vec3f(diagonal.x(), 0.0, 0.0),
            vec3f(0.0, diagonal.y(), 0.0),
            vec3f(0.0, 0.0, diagonal.z()),
        )
    }
}

impl Mat3f {
    pub fn to_array(self) -> [f32; 9] {
        [
            self[0][0], self[0][1], self[0][2], self[1][0], self[1][1], self[1][2], self[2][0],
            self[2][1], self[2][2],
        ]
    }

    pub fn to_array_2d(self) -> [[f32; 3]; 3] {
        [
            [self[0][0], self[0][1], self[0][2]],
            [self[1][0], self[1][1], self[1][2]],
            [self[2][0], self[2][1], self[2][2]],
        ]
    }

    pub fn to_mat2f(self) -> Mat2f {
        Mat2f::from(self)
    }

    pub fn to_mat4f(self) -> Mat4f {
        Mat4f::from(self)
    }

    pub fn to_mat3d(self) -> Mat3d {
        Mat3d::new(
            self.x_axis.to_vec3d(),
            self.y_axis.to_vec3d(),
            self.z_axis.to_vec3d(),
        )
    }
}
