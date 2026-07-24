Here is the final, fully refined `prompt.md` containing all your specifications. It integrates the multiple-arena lifetime strategy alongside the data-oriented constraints and offline mathematical verification requirements.

---

# Technical Prompt: VIO Pipeline Port to C++17 (Data-Oriented Design)

## Context & Objective

We are porting a Visual-Inertial Odometry (VIO) pipeline from Python to high-performance C++17. The current pipeline features sliding-window hovering conditions handled via LIFO/FIFO buffer queuing techniques.

The primary objective of this phase is to:

1. Generate the master `CMakeLists.txt` configuring a vendorized Eigen dependency.
2. Look into the entire source code and technical documentation within `docs/` to extract mathematical matrices, state definitions, and pipeline mechanics.
3. Port every functional Python source directory (`core`, `initialize`, `msckf`, `type`) into equivalent high-performance `.hpp` and `.cpp` implementation pairs adhering to **strict Data-Oriented Design (DOD)** rules.

---

## Architectural Guardrails (Non-Negotiable)

### 1. Data-Oriented Design (DOD) Over OOP

* **No Inheritance or Classes:** Do not write keyword `class`, virtual functions, or inheritance hierarchies.
* **Plain Old Data (POD):** Represent data exclusively using flat `struct` collections containing raw primitive arrays, standard scalars, or Eigen types.
* **No Internal Allocation:** Structs must possess a fixed memory footprint. They must never internally allocate heap space.

### 2. Zero Dynamic Memory Allocation

* **Banned Containers:** Do not use `std::vector`, `std::list`, `std::map`, `std::unordered_map`, or `std::string`.
* **Fixed-Size Contiguous Arrays:** Use plain C-style arrays (`T arr[MAX_SIZE]`) or `std::array<T, N>` wrapped within your structs. Use explicit capacity counters to keep track of active elements:
```cpp
struct FeatureDatabase {
    Feature features[2048]; 
    std::size_t count{0};
};

```


* **Eigen Integration:** To handle linear algebra without dynamic allocations, enforce fixed-size matrices (e.g., `Eigen::Matrix<double, 3, 3>`) or wrap arena-allocated raw buffers using `Eigen::Map`:
```cpp
// Mapping pre-allocated raw memory into an operational Eigen block
Eigen::Map<Eigen::MatrixXd> dynamic_matrix(raw_buffer, rows, cols);

```



### 3. Arena Memory Management & Multiple Arena Strategy

You must strictly utilize the `ArenaAllocator` implementation available in `arena.cpp`. Memory allocation should ideally occur at the starting of program launch.

#### Available Arena API Reference:

```cpp
class ArenaAllocator {
public:
    ArenaAllocator(std::size_t size);
    void* allocate(std::size_t size, std::size_t alignment = alignof(std::max_align_t));
    
    template <typename T, typename... Args> 
    T* allocate(Args&&... args);
    
    void reset();
};

```

* **Multiple Arenas Allowed:** To make lifetime management significantly easier, you are explicitly allowed to create and use **more than one instance of `ArenaAllocator**`.
* **Lifetime Segmentation Example:**
* A `GlobalArena` for persistent states that last the lifetime of the program execution (e.g., persistent Feature Database pools).
* A `FrameArena` or `WindowArena` that can be quickly wiped clean via `arena.reset()` at the end of every frame or sliding window update cycle to manage LIFO/FIFO sliding state steps effortlessly.


* Objects placed in any Arena **must be trivially destructible**.

---

## Phase 1: Build System Setup (`CMakeLists.txt`)

Create a robust `CMakeLists.txt` file targeting C++17. Set up a local `vendor/` directory to include a fixed version of the Eigen header-only library.

```cmake
cmake_minimum_required(VERSION 3.14)
project(vio_pipeline_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Add local vendor directory for header-only Eigen dependencies
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/vendor/eigen)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# Project sources
set(SOURCES
    arena.cpp
    type/type.cpp
    core/cam_base.cpp
    core/feature_database.cpp
    # Add other mapped module sources here...
)

add_executable(vio_pipeline ${SOURCES} main.cpp)

```

---

## Phase 2: Structural Mappings

Translate each Python directory component into functional, standalone C++ algorithmic processing functions that act directly on the structures populated within your pre-allocated Arenas.

### 1. `type/` (Core Geometric Representations)

* Convert `JPLQuat.py`, `PoseJPL.py`, `Landmark.py`, and `quat_ops.py` into inline operations on flat structural representations.
* Do not use complex objects; utilize flat `Eigen::Matrix` or fixed vector arrays.

### 2. `core/` (Camera Models & Feature Database)

* Flatten `cam_base.py`, `CamEqui.py`, and `cam_radtan.py` into a single configuration struct switch or function pointers passing direct distortion coefficients (`RadTan` vs `Equidistant`).
* Convert `feature_database.py` into a bounded structural lookup engine relying strictly on fixed max limits (e.g., maximum active tracking constraints).

### 3. `initialize/` & `msckf/` (Math Engines)

* Map all prediction, zero-velocity updates, and propagation models directly to clean functional equations.
* Keep intermediate calculation spaces allocated via temporary, stack-based small Eigen matrices or isolated local frame sub-arenas to avoid blowing out stack constraints.

---

## Math Verification & Testing Protocol

Since the deployment machine has no ROS runtime system tools installed, all validation must focus strictly on **algorithmic math equivalence** against the reference Python implementation.

1. **Deterministic Test Vectors:** You are required to create a testing harness (`tests/verify_math.cpp`) that reads a static series of synthetic matrix states, IMU acceleration values, and landmark pixel coords.
2. **Comparison Matrix:** Run the original Python pipeline over these identical numeric values to print out state vectors, covariance matrix elements, and updating steps.
3. **Assertions:** Write explicit assertions validating that the C++ outputs match the Python mathematical outputs to within a floating-point tolerance of $10^{-7}$.
4. **Execution Verification:** Ensure the compiled binary can be successfully verified using standard test invocations without any system runtime environmental prerequisites (like `uv` or `ROS`).

---

## Next Steps for Agent Implementation:

1. Examine the contents of the `docs/` folder (such as `VIO_PIPELINE_V3_TECHNICAL_DOCUMENTATION.md` and `state-and-types.md`) to extract the analytical matrices, state dimension limits, and mathematical models.
2. Write out the operational `CMakeLists.txt`.
3. Progress systematically directory-by-directory (`type/` $\rightarrow$ `core/` $\rightarrow$ `initialize/` $\rightarrow$ `msckf/`), crafting the paired `.hpp` and `.cpp` definitions according to the specified DOD design constraints.

Begin executing step 1 immediately.
