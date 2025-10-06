// g++ -std=c++20 orchestrator.cpp -o orchestrator -lpthread /usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1
// (or install libnvidia-ml-dev and link with -lnvidia-ml)
//
// Requires: NVML headers (from NVIDIA driver or CUDA toolkit) and nlohmann/json (header-only).
//   sudo apt-get install -y nlohmann-json3-dev
//
// Run (dev): sudo -E ./orchestrator

#include "json.hpp"
#include <arpa/inet.h>
#include <chrono>
#include <cstdint>
#include <errno.h>
#include <fcntl.h>
#include <memory>
#include <mutex>
#include <nvml.h>
#include <signal.h>
#include <spawn.h>
#include <stdarg.h>
#include <stdio.h>
#include <string.h>
#include <string>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>
#include <unordered_map>

#define PATH_BASE_CUDA_SHM "/dev/shm/shared_mem_"
#define PATH_BASE_CUDA_PIN "/dev/shm/cuda_pin_"
#define GPU_INDEX 0 // which physical GPU to use for MIG slices

using json = nlohmann::json;

static void logf(const char *lvl, const char *fmt, ...) {
    char buf[4096];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    fprintf(stderr, "[%s] %s\n", lvl, buf);
}
#define LOGI(...) logf("INFO", __VA_ARGS__)
#define LOGE(...) logf("ERR", __VA_ARGS__)
// trace logging (opt-in via --trace)
static bool g_trace = false;
// skip file creation (opt-in via --no-createFiles)
static bool g_no_create_files = false;
#define LOGD(...)                                                                                                           \
    do {                                                                                                                    \
        if (g_trace)                                                                                                        \
            logf("TRACE", __VA_ARGS__);                                                                                     \
    } while (0)
#define TRACE_FN() LOGD("enter %s", __func__)

// ---------- config ------------------------------------------------------------
struct VmConfig {
    std::string vm_id;         // e.g., "vm-123"
    std::string uds_path_base; // e.g., "/u/home/mege/workspace/tmp/v.sock"
    uint32_t control_port = 1110;
    uint32_t data_port_base = 1200;
};

struct ProcHandle {
    pid_t pid = -1;
};

struct Assignment {
    std::string mig_uuid;
    uint32_t data_port;
    ProcHandle server;
};

static std::vector<VmConfig> VMS = {
        {"vm1", "/u/home/mege/workspace/tmp/v1.sock", 1110, 1111}, {"vm2", "/u/home/mege/workspace/tmp/v2.sock", 1110, 1111},
        {"vm3", "/u/home/mege/workspace/tmp/v3.sock", 1110, 1111}, {"vm4", "/u/home/mege/workspace/tmp/v4.sock", 1110, 1111},
        {"vm5", "/u/home/mege/workspace/tmp/v5.sock", 1110, 1111}, {"vm6", "/u/home/mege/workspace/tmp/v6.sock", 1110, 1111},
        {"vm7", "/u/home/mege/workspace/tmp/v7.sock", 1110, 1111},
        // Each VM has: one control socket and one data socket
        // vm1: control=v1.sock_1110, data=v1.sock_1111
        // vm2: control=v2.sock_1110, data=v2.sock_1111
        // etc. (port numbers can be the same since socket paths are different)
};

// ---------- Firecracker UDS SHM CUDA_PIN Helpers ------------------------------------------
static inline std::string uds_for(const VmConfig &vm, uint32_t port) {
    return vm.uds_path_base + "_" + std::to_string(port);
}

static inline std::string shm_for(const VmConfig &vm) {
    // parse vm.uds_path_base to get the number of the vm, e.g. v1.sock -> 1
    size_t pos1 = vm.uds_path_base.rfind('v');
    size_t pos2 = vm.uds_path_base.rfind('.');
    if (pos1 == std::string::npos || pos2 == std::string::npos || pos2 <= pos1 + 1) {
        throw std::runtime_error("Invalid uds_path_base format: " + vm.uds_path_base);
    }
    std::string id_str = vm.uds_path_base.substr(pos1 + 1, pos2 - pos1 - 1);
    int identifier = std::stoi(id_str);
    return PATH_BASE_CUDA_SHM + std::to_string(identifier);
}

static inline std::string pin_for(const VmConfig &vm) {
    // parse vm.uds_path_base to get the number of the vm, e.g. v1.sock -> 1
    size_t pos1 = vm.uds_path_base.rfind('v');
    size_t pos2 = vm.uds_path_base.rfind('.');
    if (pos1 == std::string::npos || pos2 == std::string::npos || pos2 <= pos1 + 1) {
        throw std::runtime_error("Invalid uds_path_base format: " + vm.uds_path_base);
    }
    std::string id_str = vm.uds_path_base.substr(pos1 + 1, pos2 - pos1 - 1);
    int identifier = std::stoi(id_str);
    return PATH_BASE_CUDA_PIN + std::to_string(identifier);
}

static void createShmFiles() {
    TRACE_FN();
    for (uint32_t i = 1; i <= 7; ++i) {
        // delete old ones if they exist
        std::string old_path = PATH_BASE_CUDA_SHM + std::to_string(i);
        ::unlink(old_path.c_str());
        // truncate -s 10M /dev/shm/shared_mem1
        // chmod 600 /dev/shm/shared_mem1
        std::string path = PATH_BASE_CUDA_SHM + std::to_string(i);
        int fd = ::open(path.c_str(), O_RDWR | O_CREAT, 0600);
        if (fd >= 0) {
            ::ftruncate(fd, 10 * 1024 * 1024); // 10 MiB
            ::close(fd);
        }
    }
}

static void createPinFiles() {
    TRACE_FN();
    for (uint32_t i = 1; i <= 7; ++i) {
        // delete old ones if they exist
        std::string old_path = PATH_BASE_CUDA_PIN + std::to_string(i);
        ::unlink(old_path.c_str());
        // truncate -s 50M /dev/shm/cuda_pin1
        // chmod 666 /dev/shm/cuda_pin1
        std::string path = PATH_BASE_CUDA_PIN + std::to_string(i);
        int fd = ::open(path.c_str(), O_RDWR | O_CREAT, 0666);
        if (fd >= 0) {
            ::ftruncate(fd, 50 * 1024 * 1024); // 50 MiB
            ::close(fd);
        }
    }
}

// ---------- UNIX socket server ------------------------------------------------
static int make_unix_server(const std::string &path) {
    TRACE_FN();
    ::unlink(path.c_str());
    int fd = ::socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
    if (fd < 0)
        return -1;

    sockaddr_un addr{};
    addr.sun_family = AF_UNIX;
    if (path.size() >= sizeof(addr.sun_path)) {
        errno = ENAMETOOLONG;
        ::close(fd);
        return -1;
    }
    strncpy(addr.sun_path, path.c_str(), sizeof(addr.sun_path) - 1);
    if (::bind(fd, (sockaddr *)&addr, sizeof(addr)) < 0) {
        ::close(fd);
        return -1;
    }
    if (::listen(fd, 64) < 0) {
        ::close(fd);
        return -1;
    }
    ::chmod(path.c_str(), 0600);
    return fd;
}

static int accept_client(int server_fd) {
    TRACE_FN();
    return ::accept4(server_fd, nullptr, nullptr, SOCK_CLOEXEC);
}

static void reap_children() {
    TRACE_FN();
    int status = 0;
    pid_t p;
    while ((p = ::waitpid(-1, &status, WNOHANG)) > 0) {
        LOGI("reaped pid=%d status=%d", p, status);
    }
}

// ---------- NVML MIG wrapper (H100 94GB tuned) --------------------------------
struct NvmlMig {
    NvmlMig() {
        TRACE_FN();
        NVML_CHECK(nvmlInit_v2());
        unsigned int count = 0;
        NVML_CHECK(nvmlDeviceGetCount_v2(&count));
        if (count == 0)
            throw std::runtime_error("No NVIDIA devices found");
    }
    ~NvmlMig() {
        TRACE_FN();
        nvmlShutdown();
    }

    void enableMigOnGpu(unsigned gpuIndex) {
        TRACE_FN();
        nvmlDevice_t dev{};
        NVML_CHECK(nvmlDeviceGetHandleByIndex_v2(gpuIndex, &dev));
        NVML_CHECK(nvmlDeviceSetMigMode(dev, NVML_DEVICE_MIG_ENABLE, NVML_DEVICE_MIG_DISABLE));
        LOGI("Enabled MIG on GPU %u", gpuIndex);
    }

    std::pair<nvmlGpuInstance_t, nvmlComputeInstance_t> createGiCi(nvmlDevice_t dev, unsigned giProfileId,
                                                                   unsigned ciProfileId) {
        TRACE_FN();
        nvmlGpuInstance_t gi{};
        nvmlComputeInstance_t ci{};

        nvmlGpuInstanceProfileInfo_t giInfo{};
        NVML_CHECK(nvmlDeviceGetGpuInstanceProfileInfo(dev, giProfileId, &giInfo));
        NVML_CHECK(nvmlDeviceCreateGpuInstance(dev, giInfo.id, &gi));

        nvmlComputeInstanceProfileInfo_t ciInfo{};
        NVML_CHECK(nvmlGpuInstanceGetComputeInstanceProfileInfo(gi, ciProfileId, NVML_COMPUTE_INSTANCE_ENGINE_PROFILE_SHARED,
                                                                &ciInfo));
        NVML_CHECK(nvmlGpuInstanceCreateComputeInstance(gi, ciInfo.id, &ci));
        return {gi, ci};
    }

    std::string migUuidFor(nvmlDevice_t dev, nvmlGpuInstance_t gi, nvmlComputeInstance_t ci) {
        TRACE_FN();
        // Resolve GI/CI pair to a MIG device UUID
        nvmlGpuInstanceInfo_t giInfo{};
        NVML_CHECK(nvmlGpuInstanceGetInfo(gi, &giInfo)); // ✅ struct out
        unsigned int giId = giInfo.id;                   // use field

        nvmlComputeInstanceInfo_t ciInfo{};
        NVML_CHECK(nvmlComputeInstanceGetInfo(ci, &ciInfo)); // ✅ struct out
        unsigned int ciId = ciInfo.id;                       // use field

        char uuid[120] = {0};

        // Iterate all MIG devices to find match
        unsigned migCount = 0;
        NVML_CHECK(nvmlDeviceGetMaxMigDeviceCount(dev, &migCount));
        bool found = false;
        for (unsigned i = 0; i < migCount; ++i) {
            nvmlDevice_t migDev{};
            if (nvmlDeviceGetMigDeviceHandleByIndex(dev, i, &migDev) != NVML_SUCCESS)
                continue;
            unsigned gId = 0, cId = 0;
            if (nvmlDeviceGetGpuInstanceId(migDev, &gId) != NVML_SUCCESS)
                continue;
            if (nvmlDeviceGetComputeInstanceId(migDev, &cId) != NVML_SUCCESS)
                continue;
            if (gId == giId && cId == ciId) {
                NVML_CHECK(nvmlDeviceGetUUID(migDev, uuid, sizeof(uuid)));
                found = true;
            }
        }
        if (!found) {
            // Best-effort: clean up and fail
            nvmlComputeInstanceDestroy(ci);
            nvmlGpuInstanceDestroy(gi);
            throw std::runtime_error("Failed to resolve MIG UUID for GI/CI");
        }
        return std::string(uuid);
    }

    std::string allocateMig(unsigned gpuIndex, unsigned reqGB, nvmlGpuInstance_t *outGi, nvmlComputeInstance_t *outCi) {
        try {
            TRACE_FN();
            nvmlDevice_t dev{};
            NVML_CHECK(nvmlDeviceGetHandleByIndex_v2(gpuIndex, &dev));

            const auto &p = chooseProfile(reqGB);
            auto [gi, ci] = createGiCi(dev, p.giProfileId, p.ciProfileId);
            auto uuid = migUuidFor(dev, gi, ci);
            *outGi = gi;
            *outCi = ci;
            LOGI("Allocated MIG %s for request %u GB (profile %s)", uuid.c_str(), reqGB, p.name);
            return uuid;
        } catch (const std::exception &e) {
            // release on error
            if (outGi && *outGi) {
                nvmlGpuInstanceDestroy(*outGi);
                LOGE("Destroyed GI on error");
            }
            if (outCi && *outCi) {
                nvmlComputeInstanceDestroy(*outCi);
                LOGE("Destroyed CI on error");
            }
            throw;
        }
    }

    void destroyMig(nvmlGpuInstance_t gi, nvmlComputeInstance_t ci) {
        TRACE_FN();
        if (ci)
            nvmlComputeInstanceDestroy(ci);
        if (gi)
            nvmlGpuInstanceDestroy(gi);
    }

    struct Profile {
        unsigned slices;      // 1,2,3,4,7
        unsigned memGB_label; // 12,24,47,94 (for H100 94GB PCIe)
        int giProfileId;      // NVML_GPU_INSTANCE_PROFILE_*_SLICE
        int ciProfileId;      // NVML_COMPUTE_INSTANCE_PROFILE_*_SLICE
        const char *name;     // "1g.12gb", etc.
    };

    static const std::vector<Profile> &h100_94gb_profiles() {
        TRACE_FN();
        static const std::vector<Profile> k = {
                {1, 12, NVML_GPU_INSTANCE_PROFILE_1_SLICE, NVML_COMPUTE_INSTANCE_PROFILE_1_SLICE, "1g.12gb"},
                {2, 24, NVML_GPU_INSTANCE_PROFILE_2_SLICE, NVML_COMPUTE_INSTANCE_PROFILE_2_SLICE, "2g.24gb"},
                {3, 47, NVML_GPU_INSTANCE_PROFILE_3_SLICE, NVML_COMPUTE_INSTANCE_PROFILE_3_SLICE, "3g.47gb"},
                {4, 47, NVML_GPU_INSTANCE_PROFILE_4_SLICE, NVML_COMPUTE_INSTANCE_PROFILE_4_SLICE, "4g.47gb"},
                {7, 94, NVML_GPU_INSTANCE_PROFILE_7_SLICE, NVML_COMPUTE_INSTANCE_PROFILE_7_SLICE, "7g.94gb"},
        };
        return k;
    }

    // Pick the smallest profile whose labeled size >= reqGB.
    const Profile &chooseProfile(unsigned reqGB) const {
        TRACE_FN();
        auto &cat = h100_94gb_profiles();
        // default to the largest (7g.94gb)
        size_t best = cat.size() - 1;
        for (size_t i = 0; i < cat.size(); ++i) {
            if (cat[i].memGB_label >= reqGB) {
                best = i;
                break;
            }
        }
        LOGD("Chose profile %s for request %u GB", cat[best].name, reqGB);
        return cat[best];
    }

  private:
    static void NVML_CHECK(nvmlReturn_t st) {
        if (st != NVML_SUCCESS) {
            throw std::runtime_error(std::string("NVML error: ") + nvmlErrorString(st));
        }
    }
};

// ---------- process launch (cuda_server) -------------------------------------
static ProcHandle spawn_cuda_server(const VmConfig &vm, uint32_t data_port, const std::string &mig_uuid) {
    TRACE_FN();

    std::string udsPath = uds_for(vm, data_port);
    std::string shmPath = shm_for(vm);
    std::string pinPath = pin_for(vm);

    std::string exe = "./cuda_server";
    std::string argDebug = "--debug"; // (optional)
    std::string argUds = std::string("--uds=") + udsPath;
    std::string argShm = std::string("--shm=") + shmPath;
    std::string argPin = std::string("--pin=") + pinPath;
    std::string argMig = std::string("--migID=") + mig_uuid;

    std::vector<char *> argv;
    argv.push_back(const_cast<char *>(exe.c_str()));
    argv.push_back(const_cast<char *>(argDebug.c_str()));
    argv.push_back(const_cast<char *>(argUds.c_str()));
    argv.push_back(const_cast<char *>(argShm.c_str()));
    argv.push_back(const_cast<char *>(argPin.c_str()));
    argv.push_back(const_cast<char *>(argMig.c_str()));
    argv.push_back(nullptr);

    // Inherit parent environment and inject CUDA_VISIBLE_DEVICES=MIG-<UUID>
    std::vector<std::string> envKVs;

    // Copy all parent environment variables except CUDA_VISIBLE_DEVICES
    extern char **environ;
    for (char **env = environ; *env != nullptr; ++env) {
        std::string envVar(*env);
        if (envVar.find("CUDA_VISIBLE_DEVICES=") != 0) {
            envKVs.emplace_back(envVar);
        }
    }

    // Add our specific CUDA_VISIBLE_DEVICES
    envKVs.emplace_back("CUDA_VISIBLE_DEVICES=" + mig_uuid);

    std::vector<char *> envp;
    for (auto &s : envKVs)
        envp.push_back(const_cast<char *>(s.c_str()));
    envp.push_back(nullptr);

    posix_spawn_file_actions_t fa;
    posix_spawn_file_actions_init(&fa);

    // Redirect stderr to a log file for debugging (optional)
    std::string logPath = "/tmp/cuda_server_" + std::to_string(data_port) + ".log";
    posix_spawn_file_actions_addopen(&fa, STDERR_FILENO, logPath.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);

    pid_t pid = -1;
    int rc = posix_spawnp(&pid, exe.c_str(), &fa, /*attr*/ nullptr, argv.data(), envp.data());
    posix_spawn_file_actions_destroy(&fa);
    if (rc != 0) {
        LOGE("posix_spawn cuda_server failed: %s", strerror(rc));
        return {};
    }

    // Give the process a moment to start, then check if it's still alive
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    int status;
    pid_t result = waitpid(pid, &status, WNOHANG);
    if (result == pid) {
        // Process already exited
        LOGE("cuda_server pid=%d exited immediately with status=%d, check %s", pid, status, logPath.c_str());
        return {};
    } else if (result == -1) {
        LOGE("waitpid failed for cuda_server pid=%d: %s", pid, strerror(errno));
        return {};
    }

    LOGI("spawned cuda_server pid=%d MIG=%s uds=%s shm=%s pin=%s (stderr: %s)", pid, mig_uuid.c_str(), udsPath.c_str(),
         shmPath.c_str(), pinPath.c_str(), logPath.c_str());
    return {.pid = pid};
}

static void kill_if_alive(ProcHandle &p) {
    TRACE_FN();
    if (p.pid > 0) {
        kill(p.pid, SIGTERM);
        p.pid = -1;
    }
    reap_children();
}

static void reap_children_async() {
    TRACE_FN();
    int status = 0;
    pid_t p;
    while ((p = ::waitpid(-1, &status, WNOHANG)) > 0) {
        LOGI("reaped pid=%d status=%d", p, status);
    }
}

// ---------- per-VM state & control handling ----------------------------------
struct VmState {
    VmConfig cfg;
    std::mutex mu;
    std::unordered_map<int, Assignment> assignments; // key: control connection fd
    std::unordered_map<int, std::pair<nvmlGpuInstance_t, nvmlComputeInstance_t>> mig_handles;
    bool data_port_in_use = false; // single data port per VM
};

static unsigned normalize_req_gb(const json &req) {
    // Accept either "size_gb" (number) or a "profile" string like "1g.12gb"
    if (req.contains("profile")) {
        std::string p = req["profile"].get<std::string>();
        if (p.rfind("1g", 0) == 0)
            return 12;
        else if (p.rfind("2g", 0) == 0)
            return 24;
        else if (p.rfind("3g", 0) == 0)
            return 47;
        else if (p.rfind("4g", 0) == 0)
            return 47;
        else if (p.rfind("7g", 0) == 0)
            return 94;
    }
    // numeric path: accept bytes / MiB / GiB heuristically
    unsigned long long v = req.value("size_gb", 12ull); // get as 64-bit
    if (v > 1024ull * 1024ull) {                        // looks like bytes
        v = (v + (1ull << 30) - 1) >> 30;               // -> GiB (ceil)
    } else if (v > 1024ull) {                           // looks like MiB
        v = (v + 1023ull) / 1024ull;                    // -> GiB (ceil)
    }
    return static_cast<unsigned>(v);
}

static void handle_vm_client(int cfd, VmState &vm, NvmlMig &mig) {
    TRACE_FN();
    FILE *fp = fdopen(cfd, "r+");
    if (!fp) {
        close(cfd);
        return;
    }

    int conn_id = cfd;
    char *line = nullptr;
    size_t cap = 0;

    nvmlGpuInstance_t gi{};
    nvmlComputeInstance_t ci{};

    while (getline(&line, &cap, fp) != -1) {
        try {
            json req = json::parse(line);
            const std::string op = req.value("op", "");
            if (op == "alloc") {
                unsigned sizeGB = normalize_req_gb(req);

                uint32_t data_port;
                {
                    std::scoped_lock lk(vm.mu);
                    if (vm.data_port_in_use) {
                        json resp = {{"status", "error"}, {"reason", "data_port_busy"}};
                        std::string s = resp.dump() + "\n";
                        fwrite(s.data(), 1, s.size(), fp);
                        fflush(fp);
                        continue;
                    }
                    data_port = vm.cfg.data_port_base; // Use the single data port
                    vm.data_port_in_use = true;
                }

                std::string mig_uuid = mig.allocateMig(/*gpu*/ GPU_INDEX, sizeGB, &gi, &ci);
                auto server = spawn_cuda_server(vm.cfg, data_port, mig_uuid);
                if (server.pid <= 0) {
                    // cuda_server failed to spawn, mark port as free
                    {
                        std::scoped_lock lk(vm.mu);
                        vm.data_port_in_use = false;
                    }
                    // Clean up the MIG allocation
                    mig.destroyMig(gi, ci);

                    json resp = {{"status", "error"}, {"reason", "failed_to_spawn_cuda_server"}};
                    std::string s = resp.dump() + "\n";
                    if (fwrite(s.data(), 1, s.size(), fp) != s.size() || fflush(fp) != 0) {
                        LOGE("Failed to write error response: %s", strerror(errno));
                    }
                    continue;
                }

                {
                    std::scoped_lock lk(vm.mu);
                    vm.assignments[conn_id] = Assignment{mig_uuid, data_port, server};
                    vm.mig_handles[conn_id] = {gi, ci};
                }

                json resp = {{"status", "ok"}, {"mig_uuid", mig_uuid}, {"data_port", data_port}};
                std::string s = resp.dump() + "\n";
                if (fwrite(s.data(), 1, s.size(), fp) != s.size() || fflush(fp) != 0) {
                    LOGE("Failed to write success response: %s", strerror(errno));
                }

            } else if (op == "release") {
                std::scoped_lock lk(vm.mu);
                auto it = vm.assignments.find(conn_id);
                if (it != vm.assignments.end()) {
                    if (it->second.server.pid > 0)
                        kill(it->second.server.pid, SIGTERM);
                    reap_children();
                    auto mh = vm.mig_handles[conn_id];
                    mig.destroyMig(mh.first, mh.second);
                    uint32_t freed_port = it->second.data_port;
                    vm.assignments.erase(it);
                    vm.mig_handles.erase(conn_id);
                    // mark port as free and unlink its UDS
                    vm.data_port_in_use = false;
                    ::unlink(uds_for(vm.cfg, freed_port).c_str());
                }
                json resp = {{"status", "ok"}};
                std::string s = resp.dump() + "\n";
                if (fwrite(s.data(), 1, s.size(), fp) != s.size() || fflush(fp) != 0) {
                    LOGE("Failed to write release response: %s", strerror(errno));
                }

            } else if (op == "resize") {
                unsigned sizeGB = req.value("size_gb", 24u);

                { // stop server
                    std::scoped_lock lk(vm.mu);
                    auto it = vm.assignments.find(conn_id);
                    if (it != vm.assignments.end() && it->second.server.pid > 0)
                        kill(it->second.server.pid, SIGTERM);
                }
                reap_children();

                { // destroy old MIG
                    std::scoped_lock lk(vm.mu);
                    auto mh = vm.mig_handles[conn_id];
                    mig.destroyMig(mh.first, mh.second);
                    uint32_t old_port = 0;
                    if (vm.assignments.count(conn_id))
                        old_port = vm.assignments[conn_id].data_port;
                    vm.assignments.erase(conn_id);
                    vm.mig_handles.erase(conn_id);
                    if (old_port) {
                        // port becomes free (but we'll reuse the same port)
                        vm.data_port_in_use = false;
                        ::unlink(uds_for(vm.cfg, old_port).c_str());
                    }
                }

                // allocate new MIG
                std::string mig_uuid = mig.allocateMig(/*gpu*/ GPU_INDEX, sizeGB, &gi, &ci);
                uint32_t data_port = vm.cfg.data_port_base; // Reuse the same data port
                {
                    std::scoped_lock lk(vm.mu);
                    vm.data_port_in_use = true;
                }
                auto server = spawn_cuda_server(vm.cfg, data_port, mig_uuid);
                {
                    std::scoped_lock lk(vm.mu);
                    vm.assignments[conn_id] = Assignment{mig_uuid, data_port, server};
                    vm.mig_handles[conn_id] = {gi, ci};
                }
                json resp = {{"status", "ok"}, {"mig_uuid", mig_uuid}, {"data_port", data_port}};
                std::string s = resp.dump() + "\n";
                fwrite(s.data(), 1, s.size(), fp);
                fflush(fp);

            } else if (op == "list_profiles") {
                // Report H100 94GB profiles this orchestrator supports.
                json arr = json::array();
                for (auto &p : NvmlMig::h100_94gb_profiles()) {
                    arr.push_back({{"name", p.name}, {"slices", p.slices}, {"label_gb", p.memGB_label}});
                }
                json resp = {{"status", "ok"}, {"profiles", arr}};
                std::string s = resp.dump() + "\n";
                fwrite(s.data(), 1, s.size(), fp);
                fflush(fp);

            } else {
                json resp = {{"status", "error"}, {"error", "unknown op"}};
                std::string s = resp.dump() + "\n";
                fwrite(s.data(), 1, s.size(), fp);
                fflush(fp);
            }
        } catch (const std::exception &e) {
            LOGE("client handler error: %s", e.what());
            break;
        }
    }

    // on disconnect: best-effort cleanup (like release)
    {
        std::scoped_lock lk(vm.mu);
        auto it = vm.assignments.find(conn_id);
        if (it != vm.assignments.end()) {
            if (it->second.server.pid > 0)
                kill(it->second.server.pid, SIGTERM);
            uint32_t freed_port = 0;
            if (it != vm.assignments.end())
                freed_port = it->second.data_port;
            vm.assignments.erase(it);
            if (freed_port) {
                vm.data_port_in_use = false; // mark port as free
                ::unlink(uds_for(vm.cfg, freed_port).c_str());
            }
        }
        reap_children();
        auto it2 = vm.mig_handles.find(conn_id);
        if (it2 != vm.mig_handles.end()) {
            try {
                NvmlMig().destroyMig(it2->second.first, it2->second.second);
            } catch (...) {
            }
            vm.mig_handles.erase(it2);
        }
    }
    if (line)
        free(line);
    fclose(fp);
}

static void control_listener_thread(VmState &vm, NvmlMig &mig) {
    TRACE_FN();
    // Each VM has one data port: data_port_base (e.g., 1111)
    // No port pool needed - just track if the single port is in use

    const std::string ctl_path = uds_for(vm.cfg, vm.cfg.control_port);
    int sfd = make_unix_server(ctl_path);
    if (sfd < 0) {
        LOGE("failed to bind control socket %s", ctl_path.c_str());
        return;
    }
    LOGI("[%s] control listening on %s (data port: %u)", vm.cfg.vm_id.c_str(), ctl_path.c_str(), vm.cfg.data_port_base);

    while (true) {
        int cfd = accept_client(sfd);
        if (cfd < 0) {
            if (errno == EINTR)
                continue;
            perror("accept");
            break;
        }
        std::thread(handle_vm_client, cfd, std::ref(vm), std::ref(mig)).detach();
    }
}

// ---------- main --------------------------------------------------------------
int main(int argc, char **argv) {
    TRACE_FN();
    // parse command line flags
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--trace") == 0) {
            g_trace = true;
            LOGI("trace logging enabled");
        } else if (strcmp(argv[i], "--no-createFiles") == 0) {
            g_no_create_files = true;
        }
    }
    signal(SIGCHLD, SIG_IGN);
    signal(SIGPIPE, SIG_IGN); // Ignore broken pipe signals

    try {
        NvmlMig mig;
        // Hopper: enable MIG dynamically (needs privileges). Re-run on each start.
        // mig.enableMigOnGpu(/*gpu_index*/ 0);

        // Create shared memory and pinned memory files unless --no-createFiles is specified
        if (!g_no_create_files) {
            createShmFiles();
            createPinFiles();
        } else {
            LOGI("skipping SHM/PIN file creation");
        }

        // Start one control listener per VM
        std::vector<std::unique_ptr<VmState>> states;
        std::vector<std::thread> threads;
        for (const auto &v : VMS) {
            auto st = std::make_unique<VmState>();
            st->cfg = v;
            threads.emplace_back(control_listener_thread, std::ref(*st), std::ref(mig));
            states.push_back(std::move(st));
        }
        for (auto &t : threads)
            t.join();

    } catch (const std::exception &e) {
        LOGE("fatal: %s", e.what());
        return 1;
    }
    return 0;
}
