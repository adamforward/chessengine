use std::env;

fn main() {
    let libtorch_path = env::var("LIBTORCH").unwrap_or_else(|_| "/usr/local/libtorch".to_string());

    println!("Using LibTorch path: {}", libtorch_path);

    println!("cargo:rustc-link-search=native={}/lib", libtorch_path);

    println!("cargo:rustc-link-lib=torch");
    println!("cargo:rustc-link-lib=torch_cpu");
    println!("cargo:rustc-link-lib=c10");

    println!("cargo:rustc-link-lib=gomp");
    println!("cargo:rustc-link-lib=openblas");
    println!("cargo:rustc-link-lib=lapack");
    println!("cargo:rustc-link-lib=gfortran");

    println!("cargo:rustc-env=LIBTORCH_CXX11_ABI=0");

    println!("cargo:rerun-if-env-changed=LIBTORCH");
    println!("cargo:rerun-if-env-changed=LIBTORCH_CXX11_ABI");
}

