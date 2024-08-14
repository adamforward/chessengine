use std::env;

fn main() {
    println!("cargo:rustc-link-search=native=/usr/local/lib/libtorch/lib");
    println!("cargo:rustc-link-lib=torch");
    println!("cargo:rustc-link-lib=c10");
    println!("cargo:rustc-link-lib=c++");

    // Add rpath to ensure the program can find the shared library at runtime
    println!("cargo:rustc-link-arg=-Wl,-rpath,/usr/local/lib/libtorch/lib");

    // Set the environment variable for ABI if needed
    println!("cargo:rustc-env=LIBTORCH_CXX11_ABI=1");

    // Re-run the build script if this file or certain environment variables change
    println!("cargo:rerun-if-env-changed=build.rs");
    println!("cargo:rerun-if-env-changed=LIBTORCH");
    println!("cargo:rerun-if-env-changed=LIBTORCH_CXX11_ABI");
}