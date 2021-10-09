use std::{fmt::Display, ops::Mul};

use super::{Mat4d, Mat4f, Vec3d, Vec3f, Vec4d};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Transform {
    matrix: Mat4d,
}

pub trait Transformable {
    fn apply(&self, transform: Transform) -> Self;
    fn apply_matrix4d(&self, matrix: Mat4d) -> Self;
}

impl Default for Transform {
    fn default() -> Self {
        Self {
            matrix: Mat4d::identity(),
        }
    }
}

impl Display for Transform {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let m = self.matrix.to_array();
        write!(
            f,
            "Transform(matrix: ({} {} {} {} | {} {} {} {} | {} {} {} {} | {} {} {} {}))",
            m[0],
            m[1],
            m[2],
            m[3],
            m[4],
            m[5],
            m[6],
            m[7],
            m[8],
            m[9],
            m[10],
            m[11],
            m[12],
            m[13],
            m[14],
            m[15]
        )
    }
}

impl Mul<Transform> for Transform {
    type Output = Transform;

    fn mul(self, rhs: Transform) -> Self::Output {
        self.apply(rhs)
    }
}

impl Transform {
    pub fn from_matrix4d(matrix: Mat4d) -> Self {
        Self { matrix }
    }

    pub fn from_matrix4f(matrix: Mat4f) -> Self {
        Self {
            matrix: matrix.to_mat4d(),
        }
    }

    pub fn matrix(&self) -> Mat4d {
        self.matrix
    }

    pub fn inverse(&self) -> Self {
        Self {
            matrix: self.matrix.inverse(),
        }
    }
}

impl Transform {
    pub fn from_translation(translate: Vec3d) -> Self {
        Self::from_matrix4d(Mat4d::translation(translate))
    }

    pub fn from_rotation_x(angle: f64) -> Self {
        Self::from_matrix4d(Mat4d::rotation_x(angle))
    }

    pub fn from_rotation_y(angle: f64) -> Self {
        Self::from_matrix4d(Mat4d::rotation_y(angle))
    }

    pub fn from_rotation_z(angle: f64) -> Self {
        Self::from_matrix4d(Mat4d::rotation_z(angle))
    }

    pub fn from_rotation_axis(axis: Vec3d, angle: f64) -> Self {
        Self::from_matrix4d(Mat4d::rotation_axis(axis, angle))
    }

    pub fn from_scale(scale: Vec3d) -> Self {
        Self::from_matrix4d(Mat4d::scale(scale))
    }
}

impl Transform {
    pub fn translate(&self, translate: Vec3d) -> Self {
        self.apply_matrix4d(Mat4d::translation(translate))
    }

    pub fn rotate_x(&self, angle: f64) -> Self {
        self.apply_matrix4d(Mat4d::rotation_x(angle))
    }

    pub fn rotate_y(&self, angle: f64) -> Self {
        self.apply_matrix4d(Mat4d::rotation_y(angle))
    }

    pub fn rotate_z(&self, angle: f64) -> Self {
        self.apply_matrix4d(Mat4d::rotation_z(angle))
    }

    pub fn rotate_axis(&self, axis: Vec3d, angle: f64) -> Self {
        self.apply_matrix4d(Mat4d::rotation_axis(axis, angle))
    }

    pub fn scale(&self, scale: Vec3d) -> Self {
        self.apply_matrix4d(Mat4d::scale(scale))
    }
}

impl Transform {
    pub fn transform_homogeneous_coord(&self, coord: Vec4d) -> Vec4d {
        self.matrix * coord
    }

    pub fn transform_point3f(&self, point: Vec3f) -> Vec3f {
        let p = point.to_homogeneous_coord_point().to_vec4d();
        let p = self.transform_homogeneous_coord(p);
        if p.w().abs() > f64::EPSILON {
            (p.xyz() / p.w()).to_vec3f()
        } else {
            p.xyz().to_vec3f()
        }
    }

    pub fn transform_vector3f(&self, vector: Vec3f) -> Vec3f {
        let v = vector.to_homogeneous_coord_vector().to_vec4d();
        let v = self.transform_homogeneous_coord(v);
        if v.w().abs() > f64::EPSILON {
            (v.xyz() / v.w()).to_vec3f()
        } else {
            v.xyz().to_vec3f()
        }
    }

    pub fn transform_normal3f(&self, normal: Vec3f) -> Vec3f {
        let m = self.matrix.inverse().transpose();
        let n = normal.to_homogeneous_coord_vector().to_vec4d();
        let n = m * n;
        n.xyz().normalize().to_vec3f()
    }

    pub fn transform_point3d(&self, point: Vec3d) -> Vec3d {
        let p = point.to_homogeneous_coord_point();
        let p = self.transform_homogeneous_coord(p);
        if p.w().abs() > f64::EPSILON {
            p.xyz() / p.w()
        } else {
            p.xyz()
        }
    }

    pub fn transform_vector3d(&self, vector: Vec3d) -> Vec3d {
        let v = vector.to_homogeneous_coord_vector();
        let v = self.transform_homogeneous_coord(v);
        if v.w().abs() > f64::EPSILON {
            v.xyz() / v.w()
        } else {
            v.xyz()
        }
    }

    pub fn transform_normal3d(&self, normal: Vec3d) -> Vec3d {
        let m = self.matrix.inverse().transpose();
        let n = normal.to_homogeneous_coord_vector();
        let n = m * n;
        n.xyz().normalize()
    }

    pub fn transform<T: Transformable>(&self, object: T) -> T {
        object.apply(*self)
    }
}

impl Transformable for Transform {
    fn apply(&self, transform: Transform) -> Self {
        self.apply_matrix4d(transform.matrix)
    }

    fn apply_matrix4d(&self, matrix: Mat4d) -> Self {
        Self::from_matrix4d(matrix * self.matrix)
    }
}
