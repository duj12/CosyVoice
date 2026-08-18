#!/usr/bin/env python3
"""llama_cpp/llama.py — llama.cpp C API 的 ctypes 绑定层（embedding 输入专用）。

设计（仿 Qwen3-TTS-X 的 qwen3_tts_gguf/inference/llama.py，针对 lam_tts 定制）：

- LlamaModel:   进程级单例，加载 GGUF 权重，全量卸载 GPU（n_gpu_layers=-1）
- LlamaContext: 一个推理会话（KV cache 独立），封装 prefill / 单步 decode
- LlamaBatch:   embedding 输入专用 batch 管理（float32 embedding + 手动 position）
- LlamaContextPool: 并发池（vc_concurrent 路），每路一个 context，共享同一 model

关键设计决策：
1. **输入是连续 embedding**（float32），不是 token ids —— 用 llama_batch.embd
2. **输出是 hidden states**（embeddings），不是 logits —— ctx_params.embeddings=true + llama_set_embeddings
3. **position 手动指定**：prefill 0..T-1，decode 从 T 递增（驱动 RoPE）
4. **KV cache**：llama.cpp 内部管理（新版 llama_memory API），context 复用
5. **线程安全**：llama_decode 必须串行（同 context 加锁；并发用池）
6. **libstdc++ 兼容**：llama.cpp 用 gcc-11 编译需 GLIBCXX_3.4.30，运行时需 LD_PRELOAD
   系统 libstdc++ 或本模块预加载

绑定 API 基于 llama.cpp commit 1269cb1ff (2026-08-04) 的 include/llama.h。
"""
import ctypes
import os
import shutil
import subprocess
import threading
from pathlib import Path
from typing import List, Optional

import numpy as np


# ---------------------------------------------------------------
# 库加载
# ---------------------------------------------------------------

# 自动构建（找不到 libllama.so 时 clone+编译）相关配置
_LLAMACPP_REPO_URL = "https://github.com/ggml-org/llama.cpp.git"
_LLAMACPP_COMMIT = "1269cb1ff"  # 与下方 ctypes 绑定层对齐的 commit（2026-08-04）
# 默认构建根目录：可用环境变量 LLAMA_CPP_BUILD_ROOT 覆盖
_BUILD_ROOT = os.environ.get("LLAMA_CPP_BUILD_ROOT") or os.path.join(
    str(Path.home()), ".cache", "llama.cpp")
# 是否允许自动 clone+编译。默认开；LLAMA_CPP_AUTO_BUILD=0 关闭（仅报错）
_AUTO_BUILD = os.environ.get("LLAMA_CPP_AUTO_BUILD", "1") != "0"


def _run(cmd, cwd=None, timeout=None) -> subprocess.CompletedProcess:
    """执行命令，失败抛 RuntimeError 并附上输出尾部，方便排错。"""
    print(f"[llama.cpp auto-build] $ {' '.join(cmd)}")
    try:
        proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)
    except FileNotFoundError:
        raise RuntimeError(f"[llama.cpp auto-build] command not found: {cmd[0]} "
                           "(need git/cmake/gcc in PATH)")
    if proc.returncode != 0:
        tail = lambda s: (s or "")[-3000:]
        raise RuntimeError(
            f"[llama.cpp auto-build] command failed ({proc.returncode}): {' '.join(cmd)}\n"
            f"--- stdout ---\n{tail(proc.stdout)}\n--- stderr ---\n{tail(proc.stderr)}")
    return proc


def _ensure_llama_built() -> str:
    """找不到 libllama.so 时，自动 clone llama.cpp + cmake 编译，返回 libllama.so 路径。

    优先级（决定源码/build 位置）：
    1. 环境变量 LLAMA_CPP_BUILD_ROOT（或 LLAMA_CPP_SRC_DIR）
    2. 当前仓库相邻的 llama.cpp（开发环境已有源码树时直接复用，不重复 clone）
    3. 默认 $HOME/.cache/llama.cpp（全新环境）

    用文件锁保证多进程并发部署时只构建一次。
    """
    if not _AUTO_BUILD:
        raise FileNotFoundError(
            "libllama.so not found and LLAMA_CPP_AUTO_BUILD=0 (auto-build disabled). "
            "Set LLAMA_CPP_LIB or install llama.cpp manually.")

    # 决定源码目录：环境变量 > 仓库相邻(开发环境复用) > 默认构建根目录
    env_src = os.environ.get("LLAMA_CPP_SRC_DIR")
    dev_src = str(Path(__file__).resolve().parent.parent.parent.parent.parent / "llama.cpp")
    if env_src and os.path.exists(os.path.join(env_src, "CMakeLists.txt")):
        src_dir = env_src
    elif os.path.exists(os.path.join(dev_src, "CMakeLists.txt")):
        src_dir = dev_src
    else:
        src_dir = os.path.join(_BUILD_ROOT, "llama.cpp")

    build_dir = os.path.join(src_dir, "build")
    lib_path = os.path.join(build_dir, "bin", "libllama.so")
    if os.path.exists(lib_path):
        return lib_path  # 已构建好，直接复用

    # 工具链检查
    for tool in ("git", "cmake"):
        if not shutil.which(tool):
            raise RuntimeError(f"[llama.cpp auto-build] '{tool}' not found in PATH. "
                               f"Install it first (e.g. apt install {tool}).")
    if not shutil.which("g++"):
        raise RuntimeError("[llama.cpp auto-build] 'g++' not found. Install gcc/g++ first.")

    os.makedirs(src_dir, exist_ok=True)
    lock_path = os.path.join(_BUILD_ROOT, ".llama_cpp_build.lock")
    os.makedirs(os.path.dirname(lock_path), exist_ok=True)

    import fcntl
    with open(lock_path, "w") as lf:
        fcntl.flock(lf.fileno(), fcntl.LOCK_EX)  # 阻塞等锁，串行构建
        if os.path.exists(lib_path):
            return lib_path  # 双检查：等锁期间别的进程已构建完

        # clone
        if not os.path.exists(os.path.join(src_dir, "CMakeLists.txt")):
            print(f"[llama.cpp auto-build] cloning {_LLAMACPP_REPO_URL} -> {src_dir}")
            _run(["git", "clone", _LLAMACPP_REPO_URL, src_dir], timeout=900)

        # 固定到绑定层对齐的 commit；checkout 失败（commit 被改写）则警告并用 HEAD
        try:
            _run(["git", "-C", src_dir, "checkout", _LLAMACPP_COMMIT], timeout=300)
        except RuntimeError as e:
            print(f"[llama.cpp auto-build] WARN: pin to {_LLAMACPP_COMMIT} failed:\n{e}\n"
                  "Using HEAD — ctypes struct bindings may not match, expect errors.")

        # CUDA：有 nvcc 才开
        has_nvcc = shutil.which("nvcc") is not None
        cuda_flag = "-DGGML_CUDA=ON" if has_nvcc else "-DGGML_CUDA=OFF"
        if not has_nvcc:
            print("[llama.cpp auto-build] WARN: nvcc not found, building CPU-only. "
                  "Install CUDA toolkit to enable GPU.")
        print(f"[llama.cpp auto-build] cmake configure ({cuda_flag}, Release)...")
        _run(["cmake", "-B", build_dir, "-DCMAKE_BUILD_TYPE=Release", cuda_flag],
             cwd=src_dir, timeout=600)
        print("[llama.cpp auto-build] building (this can take several minutes)...")
        _run(["cmake", "--build", build_dir, "--config", "Release", "-j"],
             cwd=src_dir, timeout=3600)

        if not os.path.exists(lib_path):
            raise FileNotFoundError(f"[llama.cpp auto-build] build finished but {lib_path} missing")
        print(f"[llama.cpp auto-build] OK: {lib_path}")
        return lib_path


def _preload_system_libstdcxx():
    """conda python 自带旧版 libstdc++，llama.cpp CUDA 用 gcc-11 编译需 GLIBCXX_3.4.30。
    预加载系统 libstdc++，避免 OSError: GLIBCXX_3.4.30 not found。"""
    sys_dir = "/usr/lib/x86_64-linux-gnu"
    if os.path.exists(os.path.join(sys_dir, "libstdc++.so.6")):
        cur = os.environ.get("LD_LIBRARY_PATH", "")
        if sys_dir not in cur:
            os.environ["LD_LIBRARY_PATH"] = sys_dir + os.pathsep + cur
    for p in (
        "/usr/lib/x86_64-linux-gnu/libstdc++.so.6",
        "/usr/lib/gcc/x86_64-linux-gnu/11/libstdc++.so",
    ):
        if os.path.exists(p):
            try:
                ctypes.CDLL(p, mode=ctypes.RTLD_GLOBAL)
                return
            except OSError:
                pass


def _find_llama_lib(preferred_dir: Optional[str] = None) -> str:
    """定位 libllama.so。查找优先级：
    1. 环境变量 LLAMA_CPP_LIB（指定完整 .so 路径，最优先）
    2. llama_cpp_config.llama_lib_dir（yaml 配置的库目录）
    3. 相对源码目录 llama.cpp/build/bin（开发环境）
    4. 系统标准安装位置（cmake --install 到 /usr/local 后）
    """
    _preload_system_libstdcxx()

    # 1) 显式指定完整路径
    explicit = [os.environ.get("LLAMA_CPP_LIB")]
    if preferred_dir:
        explicit.append(os.path.join(preferred_dir, "libllama.so"))
    for c in explicit:
        if c and os.path.exists(c):
            return c

    # 2) 相对源码目录（仓库里放 llama.cpp 源码树时）
    src_rel = str(
        Path(__file__).resolve().parent.parent.parent.parent.parent
        / "llama.cpp" / "build" / "bin" / "libllama.so"
    )

    # 3) 系统标准安装位置（cmake --install --prefix /usr/local）
    sys_paths = [
        "/usr/local/lib/libllama.so",
        "/usr/local/lib64/libllama.so",
        "/usr/lib/x86_64-linux-gnu/libllama.so",
    ]
    for c in [src_rel] + sys_paths:
        if c and os.path.exists(c):
            return c

    # 4) find_library 兜底（依赖 ldconfig 缓存）
    import ctypes.util
    name = ctypes.util.find_library("llama")
    if name:
        return name

    # 5) 全部失败 → 自动 clone + 编译（可用 LLAMA_CPP_AUTO_BUILD=0 关闭）
    try:
        return _ensure_llama_built()
    except Exception as e:
        raise FileNotFoundError(
            "libllama.so not found. Options: set LLAMA_CPP_LIB env var, add "
            "llama_lib_dir to llama_cpp_config, `cmake --install llama.cpp`, "
            "or enable auto-build (default on).\nDetails: %s" % e
        )


def _load_libs(preferred_dir: Optional[str] = None) -> "ctypes.CDLL":
    llama_so = _find_llama_lib(preferred_dir)
    lib_dir = os.path.dirname(llama_so)
    # 依赖库优先从 libllama 同目录加载；同时把该目录加进 LD_LIBRARY_PATH，
    # 让系统安装场景下 dlopen 能解析到同目录的 libggml*.so。
    cur = os.environ.get("LD_LIBRARY_PATH", "")
    if lib_dir not in cur:
        os.environ["LD_LIBRARY_PATH"] = lib_dir + os.pathsep + cur
    for dep in ("libggml-base.so", "libggml.so", "libggml-cpu.so", "libggml-cuda.so"):
        p = os.path.join(lib_dir, dep)
        if os.path.exists(p):
            try:
                ctypes.CDLL(p)
            except OSError as e:
                print(f"[warn] load {dep} failed: {e}")
    libllama = ctypes.CDLL(llama_so)
    _silence_ggml_debug(libllama)
    return libllama


_GGML_LOG_LEVEL = {
    "NONE": 0, "DEBUG": 1, "INFO": 2, "WARN": 3, "ERROR": 4, "CONT": 5,
}

# C 侧 ggml_log_set 会持有这个函数指针；必须存为模块级全局，
# 否则局部变量被 GC 后回调变成悬垂指针 → 下次 ggml 打日志时段错误。
_GGML_LOG_CB = None


def _silence_ggml_debug(libllama: "ctypes.CDLL") -> None:
    """llama.cpp/ggml 默认日志回调不过滤级别，CUDA Graph 复用时会打印 DEBUG 日志
    （`CUDA Graph id %zu reused`）刷屏。这里通过 ggml_log_set 挂一个只放行
    INFO 及以上的回调，把 DEBUG 静音（不影响 WARN/ERROR）。
    """
    global _GGML_LOG_CB
    log_cb_t = ctypes.CFUNCTYPE(
        None, ctypes.c_int, ctypes.c_char_p, ctypes.c_void_p)

    @log_cb_t
    def _log_cb(level: int, text: bytes, user_data) -> None:
        if level >= _GGML_LOG_LEVEL["INFO"]:
            import sys
            sys.stderr.write(text.decode("utf-8", "replace"))
            sys.stderr.flush()

    _GGML_LOG_CB = _log_cb  # 保持引用存活，避免悬垂指针

    try:
        ggml_log_set = getattr(libllama, "ggml_log_set")
        ggml_log_set.argtypes = [log_cb_t, ctypes.c_void_p]
        ggml_log_set.restype = None
        ggml_log_set(_log_cb, None)
    except (AttributeError, OSError) as e:
        print(f"[warn] ggml_log_set failed, CUDA Graph debug logs may be noisy: {e}")


# ---------------------------------------------------------------
# ctypes 结构体（与 include/llama.h 逐字段对齐，2026-08-04 版本）
# ---------------------------------------------------------------

class llama_model_params(ctypes.Structure):
    _fields_ = [
        ("devices", ctypes.c_void_p),
        ("tensor_buft_overrides", ctypes.c_void_p),
        ("n_gpu_layers", ctypes.c_int32),
        ("split_mode", ctypes.c_int32),
        ("load_mode", ctypes.c_int32),
        ("main_gpu", ctypes.c_int32),
        ("tensor_split", ctypes.c_void_p),
        ("progress_callback", ctypes.c_void_p),
        ("progress_callback_user_data", ctypes.c_void_p),
        ("kv_overrides", ctypes.c_void_p),
        ("vocab_only", ctypes.c_bool),
        ("check_tensors", ctypes.c_bool),
        ("use_extra_bufts", ctypes.c_bool),
        ("no_host", ctypes.c_bool),
        ("no_alloc", ctypes.c_bool),
        ("load_mtp", ctypes.c_bool),
    ]


class llama_context_params(ctypes.Structure):
    _fields_ = [
        ("n_ctx", ctypes.c_uint32),
        ("n_batch", ctypes.c_uint32),
        ("n_ubatch", ctypes.c_uint32),
        ("n_seq_max", ctypes.c_uint32),
        ("n_rs_seq", ctypes.c_uint32),
        ("n_outputs_max", ctypes.c_uint32),
        ("n_threads", ctypes.c_int32),
        ("n_threads_batch", ctypes.c_int32),
        ("ctx_type", ctypes.c_int32),
        ("rope_scaling_type", ctypes.c_int32),
        ("pooling_type", ctypes.c_int32),
        ("attention_type", ctypes.c_int32),
        ("flash_attn_type", ctypes.c_int32),
        ("rope_freq_base", ctypes.c_float),
        ("rope_freq_scale", ctypes.c_float),
        ("yarn_ext_factor", ctypes.c_float),
        ("yarn_attn_factor", ctypes.c_float),
        ("yarn_beta_fast", ctypes.c_float),
        ("yarn_beta_slow", ctypes.c_float),
        ("yarn_orig_ctx", ctypes.c_uint32),
        ("defrag_thold", ctypes.c_float),
        ("cb_eval", ctypes.c_void_p),
        ("cb_eval_user_data", ctypes.c_void_p),
        ("type_k", ctypes.c_int32),
        ("type_v", ctypes.c_int32),
        ("abort_callback", ctypes.c_void_p),
        ("abort_callback_data", ctypes.c_void_p),
        ("embeddings", ctypes.c_bool),
        ("offload_kqv", ctypes.c_bool),
        ("no_perf", ctypes.c_bool),
        ("op_offload", ctypes.c_bool),
        ("swa_full", ctypes.c_bool),
        ("kv_unified", ctypes.c_bool),
        ("samplers", ctypes.c_void_p),
        ("n_samplers", ctypes.c_size_t),
        ("ctx_other", ctypes.c_void_p),
    ]


class llama_batch(ctypes.Structure):
    _fields_ = [
        ("n_tokens", ctypes.c_int32),
        ("token", ctypes.POINTER(ctypes.c_int32)),
        ("embd", ctypes.POINTER(ctypes.c_float)),
        ("pos", ctypes.POINTER(ctypes.c_int32)),
        ("n_seq_id", ctypes.POINTER(ctypes.c_int32)),
        ("seq_id", ctypes.POINTER(ctypes.POINTER(ctypes.c_int32))),
        ("logits", ctypes.POINTER(ctypes.c_int8)),
    ]


class _API:
    """llama.cpp C API 绑定（模块级单例）。"""

    _lib: Optional["ctypes.CDLL"] = None
    _lock = threading.Lock()

    @classmethod
    def lib(cls, preferred_dir: Optional[str] = None) -> "ctypes.CDLL":
        if cls._lib is None:
            with cls._lock:
                if cls._lib is None:
                    cls._lib = _load_libs(preferred_dir)
                    cls._setup()
        return cls._lib

    @classmethod
    def _setup(cls):
        lib = cls._lib
        # 值传递返回结构体的函数
        lib.llama_model_default_params.restype = llama_model_params
        lib.llama_context_default_params.restype = llama_context_params
        lib.llama_batch_init.restype = llama_batch

        lib.llama_backend_init.argtypes = []
        lib.llama_backend_init.restype = None

        lib.llama_model_load_from_file.argtypes = [ctypes.c_char_p, llama_model_params]
        lib.llama_model_load_from_file.restype = ctypes.c_void_p

        # 注意：llama_init_from_model 的 params 是按值传递结构体，不是指针
        lib.llama_init_from_model.argtypes = [ctypes.c_void_p, llama_context_params]
        lib.llama_init_from_model.restype = ctypes.c_void_p

        lib.llama_decode.argtypes = [ctypes.c_void_p, llama_batch]
        lib.llama_decode.restype = ctypes.c_int32

        lib.llama_get_embeddings_ith.argtypes = [ctypes.c_void_p, ctypes.c_int32]
        lib.llama_get_embeddings_ith.restype = ctypes.POINTER(ctypes.c_float)

        lib.llama_model_n_embd.argtypes = [ctypes.c_void_p]
        lib.llama_model_n_embd.restype = ctypes.c_int32

        # 新版 KV cache API（替代旧 llama_kv_cache_clear）
        lib.llama_get_memory.argtypes = [ctypes.c_void_p]
        lib.llama_get_memory.restype = ctypes.c_void_p
        lib.llama_memory_clear.argtypes = [ctypes.c_void_p, ctypes.c_bool]
        lib.llama_memory_clear.restype = None

        lib.llama_batch_free.argtypes = [llama_batch]
        lib.llama_batch_free.restype = None

        lib.llama_set_embeddings.argtypes = [ctypes.c_void_p, ctypes.c_bool]
        lib.llama_set_embeddings.restype = None

        lib.llama_n_ctx.argtypes = [ctypes.c_void_p]
        lib.llama_n_ctx.restype = ctypes.c_uint32


# ---------------------------------------------------------------
# LlamaModel / LlamaContext / LlamaBatch / LlamaContextPool
# ---------------------------------------------------------------

class LlamaModel:
    """一个 GGUF 模型（进程级单例，权重只加载一次）。"""

    def __init__(self, gguf_path: str, n_gpu_layers: int = -1, lib_dir: Optional[str] = None):
        self.gguf_path = gguf_path
        self.n_gpu_layers = n_gpu_layers
        lib = _API.lib(preferred_dir=lib_dir)

        lib.llama_backend_init()
        params = lib.llama_model_default_params()
        params.n_gpu_layers = n_gpu_layers
        params.use_extra_bufts = False

        self.ptr = lib.llama_model_load_from_file(gguf_path.encode(), params)
        if not self.ptr:
            raise RuntimeError(f"llama_model_load_from_file failed: {gguf_path}")
        self.n_embd = lib.llama_model_n_embd(self.ptr)
        self._lock = threading.Lock()


class LlamaBatch:
    """embedding 输入专用 batch。管理 C 堆上的 llama_batch 及其内存。"""

    def __init__(self, lib, n_tokens: int, embd_dim: int, n_seq_max: int = 1):
        self._lib = lib
        self.embd_dim = embd_dim
        self.batch = lib.llama_batch_init(n_tokens, embd_dim, n_seq_max)

    def set_embeddings(self, emb_float32: np.ndarray, pos_start: int, logits_mask: List[int]):
        """填 embedding、position、seq_id、logits flags。
        emb_float32: [n_tokens, embd_dim] float32 连续数组。
        logits_mask: 长度 n_tokens，1=需要输出 hidden states。
        """
        n = emb_float32.shape[0]
        assert emb_float32.shape[1] == self.embd_dim
        assert emb_float32.dtype == np.float32
        emb = np.ascontiguousarray(emb_float32)
        ctypes.memmove(
            self.batch.embd,
            emb.ravel().ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            n * self.embd_dim * 4,
        )
        for i in range(n):
            self.batch.pos[i] = pos_start + i
            self.batch.n_seq_id[i] = 1
            self.batch.seq_id[i][0] = 0  # seq_id 已由 batch_init 分配
            self.batch.logits[i] = int(logits_mask[i])
        self.batch.n_tokens = n

    def free(self):
        self._lib.llama_batch_free(self.batch)


class LlamaContext:
    """一个推理会话（独立 KV cache）。所有 llama_decode 调用必须串行（同 context 加锁）。"""

    def __init__(self, model: LlamaModel, n_ctx: int = 2048, flash_attn: bool = True,
                 embeddings: bool = True, offload_kqv: bool = True):
        self.model = model
        lib = _API.lib()
        params = lib.llama_context_default_params()
        params.n_ctx = n_ctx
        params.embeddings = embeddings
        params.flash_attn_type = 1 if flash_attn else 0
        params.offload_kqv = offload_kqv
        params.no_perf = True

        self.ptr = lib.llama_init_from_model(model.ptr, params)
        if not self.ptr:
            raise RuntimeError("llama_init_from_model failed")
        lib.llama_set_embeddings(self.ptr, embeddings)
        self.n_ctx = lib.llama_n_ctx(self.ptr)
        self._lock = threading.Lock()
        self.n_past = 0

    def reset(self):
        """清空 KV cache，回到初始状态。"""
        with self._lock:
            lib = _API.lib()
            mem = lib.llama_get_memory(self.ptr)
            lib.llama_memory_clear(mem, True)
            self.n_past = 0

    def prefill(self, emb_float32: np.ndarray) -> np.ndarray:
        """prefill 一段 embedding，返回最后一个 token 的 hidden states。
        emb_float32: [T, n_embd] float32。"""
        lib = _API.lib()
        with self._lock:
            batch = LlamaBatch(lib, emb_float32.shape[0], self.model.n_embd)
            mask = [0] * emb_float32.shape[0]
            mask[-1] = 1
            batch.set_embeddings(emb_float32, 0, mask)
            try:
                ret = lib.llama_decode(self.ptr, batch.batch)
                if ret != 0:
                    raise RuntimeError(f"prefill decode failed: {ret}")
            finally:
                batch.free()
            self.n_past = emb_float32.shape[0]
            return self._get_last_hidden()

    def decode_step(self, emb_float32: np.ndarray) -> np.ndarray:
        """decode 单步（1 个 token），返回其 hidden states。"""
        lib = _API.lib()
        with self._lock:
            batch = LlamaBatch(lib, 1, self.model.n_embd)
            batch.set_embeddings(emb_float32.reshape(1, -1), self.n_past, [1])
            try:
                ret = lib.llama_decode(self.ptr, batch.batch)
                if ret != 0:
                    raise RuntimeError(f"decode failed: {ret}")
            finally:
                batch.free()
            self.n_past += 1
            return self._get_last_hidden()

    def get_last_hidden(self) -> np.ndarray:
        """取最近一次 decode 的最后一个 token 的 hidden states（不推进 KV）。"""
        return self._get_last_hidden()

    def _get_last_hidden(self) -> np.ndarray:
        lib = _API.lib()
        hid = lib.llama_get_embeddings_ith(self.ptr, -1)
        return np.ctypeslib.as_array(hid, shape=(self.model.n_embd,)).copy()


class LlamaContextPool:
    """并发池：预建 N 个 context，每路一个，共享同一 model。"""

    def __init__(self, model: LlamaModel, n_contexts: int, n_ctx: int = 2048,
                 flash_attn: bool = True):
        self.model = model
        self.n_contexts = n_contexts
        self._contexts: List[LlamaContext] = [
            LlamaContext(model, n_ctx=n_ctx, flash_attn=flash_attn) for _ in range(n_contexts)
        ]
        self._available: List[LlamaContext] = list(self._contexts)
        self._lock = threading.Lock()

    def acquire(self) -> LlamaContext:
        with self._lock:
            if not self._available:
                raise RuntimeError("LlamaContextPool exhausted")
            return self._available.pop()

    def release(self, ctx: LlamaContext):
        with self._lock:
            if ctx not in self._contexts:
                raise ValueError("ctx not from this pool")
            self._available.append(ctx)

    def reset_all(self):
        for ctx in self._contexts:
            ctx.reset()
