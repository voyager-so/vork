//! Admits the tests that read fixtures the public tree does not carry.
fn main() {
    // Explicit opt-in: directory layout must not change test admission.
    println!("cargo:rustc-check-cfg=cfg(vork_private_tests)");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=VORK_PRIVATE_TESTS");
    if std::env::var("VORK_PRIVATE_TESTS").as_deref() == Ok("1") {
        println!("cargo:rustc-cfg=vork_private_tests");
    }
}
