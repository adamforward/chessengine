use std::env;

fn main() {
    // Retrieve the LIBTORCH environment variable, which should point to the root directory of your libtorch installation
    let libtorch_path = env::var("LIBTORCH").expect("LIBTORCH environment variable not set");

    // Add the libtorch library path to the Rust compiler's search paths
    println!("cargo:rustc-link-search=native={}/lib", libtorch_path);

    // Link against the necessary libtorch libraries
    println!("cargo:rustc-link-lib=torch");
    println!("cargo:rustc-link-lib=c10");
    println!("cargo:rustc-link-lib=c++");

    // Link against LLVM's libomp (OpenMP library) if using OpenMP for parallel processing
    println!("cargo:rustc-link-search=native=/opt/homebrew/opt/llvm/lib");
    println!("cargo:rustc-link-lib=omp");

    // Add runtime library paths (rpath) to ensure the executable can find the shared libraries at runtime
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}/lib", libtorch_path);
    println!("cargo:rustc-link-arg=-Wl,-rpath,/opt/homebrew/opt/llvm/lib");

    // Set the environment variable for ABI compatibility with C++11 if needed
    println!("cargo:rustc-env=LIBTORCH_CXX11_ABI=1");

    // Re-run the build script if this file or certain environment variables change
    println!("cargo:rerun-if-env-changed=build.rs");
    println!("cargo:rerun-if-env-changed=LIBTORCH");
    println!("cargo:rerun-if-env-changed=LIBTORCH_CXX11_ABI");
}
