use crate::algebra::{vec2d, vec3d, Transform, Vec2d, Vec3d};

mod rng;
pub use rng::{rng_pcg32, Pcg32, Rng};

pub fn concentric_sample_disk(u: Vec2d) -> Vec2d {
    let u = u * 2.0 - 1.0;

    if u.x() == 0.0 && u.y() == 0.0 {
        return vec2d(0.0, 0.0);
    }

    if u.x().abs() > u.y().abs() {
        let r = u.x();
        let theta = std::f64::consts::FRAC_PI_4 * (u.y() / u.x());
        vec2d(theta.cos(), theta.sin()) * r
    } else {
        let r = u.y();
        let theta = std::f64::consts::FRAC_PI_2 - std::f64::consts::FRAC_PI_4 * (u.x() / u.y());
        vec2d(theta.cos(), theta.sin()) * r
    }
}

pub fn uniform_sample_disk(u: Vec2d) -> Vec2d {
    let (u1, u2) = u.to_tuple();
    let r = u1.sqrt();
    let theta = 2.0 * std::f64::consts::PI * u2;

    vec2d(theta.cos() * r, theta.sin() * r)
}

pub fn uniform_sample_sphere(u: Vec2d) -> (Vec3d, f64) {
    let (u1, u2) = u.to_tuple();
    let z = 1.0 - 2.0 * u1;
    let r = 0f64.max(1.0 - z * z).sqrt();
    let phi = std::f64::consts::PI * 2.0 * u2;

    (
        vec3d(phi.cos() * r, phi.sin() * r, z),
        1.0 / (4.0 * std::f64::consts::PI),
    )
}

pub fn uniform_sample_hemisphere(n: Vec3d, u: Vec2d) -> (Vec3d, f64) {
    let (mut result, _) = uniform_sample_sphere(u);
    if result.dot(n) < 0.0 {
        result = -result;
    }

    (result, 1.0 / (2.0 * std::f64::consts::PI))
}

pub fn cosine_sample_hemisphere(n: Vec3d, u: Vec2d) -> (Vec3d, f64) {
    let d = concentric_sample_disk(u);
    let z = 0.0f64.max(1.0 - d.x() * d.x() - d.y() * d.y()).sqrt();
    let pdf = z * std::f64::consts::FRAC_1_PI;
    if vec3d(0.0, 0.0, -1.0).eq(&n) {
        return (vec3d(d.x(), d.y(), -z), pdf);
    }
    let d = vec3d(d.x(), d.y(), z);
    let n0 = vec3d(0.0, 0.0, 1.0);
    let axis = n0.cross(n).normalize();
    let angle = n0.angle_between(n).to_degrees();
    let t = Transform::from_rotation_axis(axis, angle);
    let d = t.transform_vector3d(d);
    (d.normalize(), pdf)
}
