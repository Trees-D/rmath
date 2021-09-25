#[cfg(any(
    all(debug_assertions, feature = "debug-ruby-assert"),
    feature = "ruby-assert"
))]
macro_rules! ruby_assert {
    ($($arg: tt)*) => ( assert!($($arg)*); )
}

#[cfg(not(any(
    all(debug_assertions, feature = "debug-ruby-assert"),
    feature = "ruby-assert"
)))]
macro_rules! ruby_assert {
    ($($arg: tt)*) => {};
}
