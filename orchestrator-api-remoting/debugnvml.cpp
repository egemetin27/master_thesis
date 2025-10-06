// debugnvml.cpp
// Build: g++ -std=c++17 debugnvml.cpp -o debugnvml -lnvidia-ml -lpthread
// Usage: ./debugnvml [deviceIndex=0] [--keep]

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <nvml.h>
#include <string>

static void die(int code, const char *where, nvmlReturn_t st) {
    std::fprintf(stderr, "[%s] %s\n", where, nvmlErrorString(st));
    std::exit(code);
}

int main(int argc, char **argv) {
    int devIndex = 0;
    bool keep = false;
    if (argc >= 2)
        devIndex = std::atoi(argv[1]);
    if (argc >= 3 && std::string(argv[2]) == "--keep")
        keep = true;

    nvmlReturn_t st = nvmlInit_v2();
    if (st != NVML_SUCCESS)
        die(1, "nvmlInit_v2", st);

    nvmlDevice_t dev{};
    st = nvmlDeviceGetHandleByIndex_v2(devIndex, &dev);
    if (st != NVML_SUCCESS)
        die(1, "nvmlDeviceGetHandleByIndex_v2", st);

    unsigned int migCur = 0, migPend = 0;
    st = nvmlDeviceGetMigMode(dev, &migCur, &migPend);
    if (st != NVML_SUCCESS)
        die(1, "nvmlDeviceGetMigMode", st);
    if (migCur != NVML_DEVICE_MIG_ENABLE) {
        std::fprintf(stderr, "MIG not enabled on device %d (current=%u, pending=%u)\n", devIndex, migCur, migPend);
        nvmlShutdown();
        return 2;
    }

    // --- GPU Instance: 1-slice (≈ 1g.12gb on H100 94GB) ---
    nvmlGpuInstanceProfileInfo_t giInfo{};
    st = nvmlDeviceGetGpuInstanceProfileInfo(dev, NVML_GPU_INSTANCE_PROFILE_1_SLICE, &giInfo);
    if (st != NVML_SUCCESS)
        die(1, "nvmlDeviceGetGpuInstanceProfileInfo(1-slice)", st);

    std::printf("GI profile: id=%u, slices=%u, memory=%llu MiB\n", giInfo.id, giInfo.sliceCount,
                (unsigned long long)giInfo.memorySizeMB);

    nvmlGpuInstance_t gi{};
    st = nvmlDeviceCreateGpuInstance(dev, giInfo.id, &gi);
    if (st == NVML_ERROR_NO_PERMISSION) {
        std::fprintf(stderr, "No permission to create GPU Instance.\n");
        nvmlShutdown();
        return 3;
    }
    if (st != NVML_SUCCESS)
        die(1, "nvmlDeviceCreateGpuInstance", st);
    std::puts("Created GPU Instance OK.");

    // --- Compute Instance: 1-slice, shared engines ---
    nvmlComputeInstanceProfileInfo_t ciInfo{};
    st = nvmlGpuInstanceGetComputeInstanceProfileInfo(gi, NVML_COMPUTE_INSTANCE_PROFILE_1_SLICE,
                                                      NVML_COMPUTE_INSTANCE_ENGINE_PROFILE_SHARED, &ciInfo);
    if (st != NVML_SUCCESS)
        die(1, "nvmlGpuInstanceGetComputeInstanceProfileInfo(1-slice,shared)", st);

    std::printf("CI profile: id=%u\n", ciInfo.id);

    nvmlComputeInstance_t ci{};
    st = nvmlGpuInstanceCreateComputeInstance(gi, ciInfo.id, &ci);
    if (st == NVML_ERROR_NO_PERMISSION) {
        std::fprintf(stderr, "No permission to create Compute Instance.\n");
        // best-effort cleanup
        nvmlReturn_t dst = nvmlGpuInstanceDestroy(gi);
        if (dst != NVML_SUCCESS)
            std::fprintf(stderr, "[cleanup gi] %s\n", nvmlErrorString(dst));
        nvmlShutdown();
        return 3;
    }
    if (st != NVML_SUCCESS)
        die(1, "nvmlGpuInstanceCreateComputeInstance", st);
    std::puts("Created Compute Instance OK.");

    if (keep) {
        std::puts("--keep specified, leaving GI/CI in place.");
    } else {
        st = nvmlComputeInstanceDestroy(ci);
        if (st != NVML_SUCCESS)
            die(1, "nvmlComputeInstanceDestroy", st);
        st = nvmlGpuInstanceDestroy(gi);
        if (st != NVML_SUCCESS)
            die(1, "nvmlGpuInstanceDestroy", st);
        std::puts("Destroyed CI and GI (cleanup done).");
    }

    nvmlShutdown();
    std::puts("Done.");
    return 0;
}
