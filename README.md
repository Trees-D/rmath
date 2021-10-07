# rmath

A math library implemented by Rust.

---

## algebra

- `f32` Type
  - `Vec2f`, `Vec3f`, `Vec4f`
  - `Mat2f`, `Mat3f`, `Mat4f`
- `f64` Type
  - `Vec2d`, `Vec3d`, `Vec4d`
  - `Mat2d`, `Mat3d`, `Mat4d`
- transform
  - `Transform`
  - trait `Transformable`

## color

- `f32` Type
  - `RGBf`, `RGBAf`
- `f64` Type
  - `RGB`, `RGBA`
- `u8` Type
  - `RGB24`, `RGB32`

## geometry

- bounds
  - `Bounds3`
- ray
  - `Ray`

## sampling

- rng
  - `Pcg32`
- method
  - `concentric_sample_disk`
  - `uniform_sample_disk`
  - `uniform_sample_sphere`
  - `uniform_sample_hemisphere`
