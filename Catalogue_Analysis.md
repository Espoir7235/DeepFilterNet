# DeepFilterNet Project

根目录包含各类全局配置文件（如 `Cargo.toml` 统筹 Rust 工作区，`pyproject.toml` 统筹 Python 环境设置），以及许可证、Readme 和构建/发布脚本 (`.sh`)。以下是核心子目录的说明：

```text
DeepFilterNet/                                # 项目根目录，实现基于深度学习的高性能音频降噪开源工程
├── Cargo.toml                                # Rust 工作空间(Workspace)主配置文件，管理各Rust子模块及全局依赖
├── cbindgen.toml                             # C/C++ 接口代码提取 (FFI) 生成器配置
├── clippy.toml                               # Rust 官方语法建议以及 Lint 工具 (Clippy) 配置文件
├── LICENSE / LICENSE-APACHE / LICENSE-MIT    # 该开源项目使用的许可协议以及权利声明文件
├── post-release.sh                           # 项目在发布新版本之后的自动清理、同步或其他处理脚本
├── pyproject.toml                            # 根目录的 Python 构建及依赖项目配置
├── README.md                                 # 项目根目录主介绍、安装及系统使用说明文档
├── release.sh                                # 用于执行自动化打包以及修改版本标记以发布新版本的工具
├── rustfmt.toml                              # Rust 语言官方指定的代码排版与格式化标准配置文件
│
├── assets/                                   # 存储训练数据集索引、模型用例和其他静态资源的配置目录
│   ├── clean.hdf5                            # 纯净(无环境噪声)语音特征或索引的缓存文件
│   ├── dataset.cfg                           # 音频数据路径配置与训练集组合相关配置文件
│   ├── noise_flac.hdf5                       # 面向 FLAC 格式编码噪声数据集加载优化的 HDF5 文件
│   ├── noise_vorbis.hdf5                     # 面向 Vorbis 格式噪声编码优化使用的缓存配置大纲
│   ├── noise.hdf5                            # 基本常见噪声混杂包引用的全局 HDF5 数据索引
│   └── README.md                             # assets 目录数据的相关内容介绍
│
├── DeepFilterNet/                            # 核心 Python 侧层：包含了声学神经网络组装、训练过程以及推理入口
│   ├── pyproject.toml / requirements*.txt    # 核心算法 Python 包依赖清单 (包括基础网络、评估模块及DNSMOS专属依赖)
│   ├── README.md / LICENSE*                  # 专注阐述 Python 模型训练这一块的相关知识协议文档
│   ├── df/                                   # DeepFilter 也就是算法相关的真正 Python 源代码包目录
│   │   ├── __init__.py                       # 模块包初始化
│   │   ├── checkpoint.py                     # 管理和实现神经网络模型运行权重 (Checkpoint) 的断点保存与复盘恢复机制
│   │   ├── config.py                         # 解析读取模型参数环境以及提取超参数设定类
│   │   ├── deepfilternet[2/3/mf].py          # 分别包含当前深度降噪的第 1、2、3 代架构或多帧架构(mf)的主干设计逻辑
│   │   ├── enhance.py                        # 对音频进行执行去噪及听觉增强验证的 CLI 脚本源码
│   │   ├── evaluation_utils.py               # 网络模型各项语音质量评测算法通用支撑工具
│   │   ├── io.py                             # 处理和读取不同格式的音频读写、张量(Tensor)流转操作模块
│   │   ├── logger.py                         # 记录控制面板信息输出规则控制及错误日志模块
│   │   ├── loss.py                           # 模型优化训练中所需要的误差损失计算(如谱损失，信噪比补偿等)算术类
│   │   ├── lr.py                             # 模型训练进程中控制学习率演进、衰减或循环计划(LR Scheduler)组件
│   │   ├── model.py                          # 高度抽象封装深度网络的基本流程及基础父类定义网络推演边界
│   │   ├── modules.py                        # 具体构筑神经网络的积木单元，例如卷积层与门控激活、下采样层代码
│   │   ├── multiframe.py                     # 时序与延迟约束下实现多帧跨频段运算的核心底层辅助块
│   │   ├── sepm.py / stoi.py                 # 分别对接并完成 PESQ / STOI 这种常用客观听觉评级的指标计算桥梁代码
│   │   ├── train.py                          # 项目内最关键环节：模型大规模迭代与训练总入口的主执行程序
│   │   ├── utils.py                          # 常规字符串修改、目录安全以及杂项运算的基础保障工具代码
│   │   ├── version.py                        # 通过脚本解析获取运行时刻该工程版本的定位
│   │   ├── visualization.py                  # 负责绘图，并在训练过程生成对应频谱对比图及图表来便于追踪训练质量 
│   │   └── scripts/                          # 一些围绕着流水线上下游所附加的脚本
│   │       ├── dnsmos_[...].py               # 不同 DNS 版本的 MOS 主观评价预测封装自动化接口 (DNSMOS评价体系)
│   │       ├── export.py                     # 推理部署阶段用来将 PyTorch 模型权重保存转换为通用 ONNX/Rust可运行数据的关键入口
│   │       ├── filter_dnsmos.py              # 利用上面提到的 DNSMOS 来衡量提取更纯正的音频实现数据过滤
│   │       └── fix_n_samples_hdf5.py         # 用来解决或排错 HDF5 中因为存储大小记录错误引发的重置处理脚本
│   └── tests/                                # df 子模块关联的 Python 单元及边界集成自动测试代码
│       └── test_dflib.py                     # 测试模型模块函数等内部 API 行内结果是否预期的单元脚本
│
├── demo/                                     # 利用 Rust 语言写的一套供开发演示的终端应用程序
│   ├── Cargo.toml / README.md                # 演示程序的入口编译包依赖配置项及其描述
│   └── src/                                  # 核心源码
│       ├── capture.rs                        # 涉及直接抓取桌面扬声器或特定硬件麦克风的捕获功能代码
│       ├── cmap.rs                           # 产生用在频谱等展示用途的热力图、颜色图映射逻辑 
│       └── main.rs                           # 演示案例应用的总命令行处理主入口逻辑程序
│
├── ladspa/                                   # 开源 LADSPA Linux 音频特效开发架构下的降噪底层适配插件系统
│   ├── Cargo.toml / LICENSE* / README*       # 生成宿主所依赖动态库配置和简短说明
│   ├── filter-chain-configs/                 # 提供对接外部宿主播放软件例如 PipeWire 对应的直接挂载管道配置
│   │   ├── deepfilter-mono-source.conf       # 单麦克风及单声道语音信道净化(麦克风除噪)处理链路配置  
│   │   └── deepfilter-stereo-sink.conf       # 双声道系统声音或外部喇叭监听链路(重放)降噪配置
│   └── src/lib.rs                            # 严格实现了标准 LADSPA API 注册监听函数的 Rust 本体库源文件
│
├── libDF/                                    # 构建整个深度系统在推理以及数据处理最高效地原生层（Rust底层库）
│   ├── Cargo.toml / LICENSE*                 # 专门针对于 libDF 底层计算库的配置管理信息
│   └── src/                                  # Rust 层面的内部数据与音频信号处理流水线实现 
│       ├── augmentations.rs                  # 读取数据时对声学流动态进行音频特征增强等泛化训练手法代码
│       ├── capi.rs                           # 把 Rust 的算法暴露导出成 C ABI 对外通用函数调用的定义层
│       ├── dataloader.rs / dataset.rs        # 将复杂的异步 HDF5 流高吞吐量快速装载为模型张量所需的管理集合
│       ├── hdf5_key_cache.rs                 # 在多线程并发加载大量音频过程中对 HDF5 文件访问建立的特征缓存体系
│       ├── lib.rs                            # 汇集该文件夹下所有内部逻辑向外的导出口模块集成文件
│       ├── logging.rs                        # 承担这个特定 Rust 原生包底层运行时刻终端文字信息输出控制
│       ├── tract.rs                          # 跟原生神经网络推理方案 `tract` 对接用来最终无需依靠 Python 在底层跑深度模型
│       ├── transforms.rs                     # 快速实现在线短时傅里叶变换 (STFT) 分析音频谱面与 ERB 特征倒谱类
│       ├── util.rs / wav_utils.rs            # 涵盖数字矩阵变换辅助及提供 WAV 标准音频无损读取工具的工具类
│       └── wasm.rs                           # 若向前端 Web 或是浏览器系统打包输出为 WebAssembly 需要借助的绑定入口
│
├── pyDF/                                     # 利用 Python C-Extensions 让 Python 直调高速的 Rust 推理引擎逻辑 (绑定层)
│   ├── Cargo.toml / pyproject.toml / setup.py# 配置如何构建打包给 pip 进行原生扩展支持所用到的全套信息声明
│   ├── libdf.pyi                             # 为在 Python 使用 pyDF 内部的方法暴露严格和清晰的 IDE 类型提示辅助
│   └── src/lib.rs                            # 用 PyO3 框架把 libDF 中的底层方法桥接入 Python 方法表的本体实现代码
│
├── pyDF-data/                                # 用于将高效原生并行的 libDF Dataloader 的流式管理逻辑桥接供给模型训练使用的绑定层
│   ├── Cargo.toml / pyproject.toml / setup.py# 构建原生模块所依赖的项目设定系统
│   ├── libdfdata/                            # Python 在模型调用端使用该封装加载工具入口
│   │   └── torch_dataloader.py               # 将原生的 Rust Load 机制与 PyTorch 原生的 DataLoader 执行特性实现无缝接合代码
│   └── src/lib.rs                            # 使用 PyO3 转出并行提取器使得其允许在 GIL 中使用以达到超高加载吞吐量实现
│
└── scripts/                                  # 整个项目通用的各式系统/性能评估辅助及操作脚本汇总目录
    ├── assert_close_npz.py / split_npz.py    # 用于校验 NumPy 打包数据的相似度验证或者提供数据块拆封的便捷脚本
    ├── build_wasm_package.sh                 # 单行指令打包并且汇集输出成 WASM 给终端页面调用的编译流程脚本
    ├── copy_datadir.py (以及 .sh)            # 方便服务器、环境之间迁移 HDF5 以及大型数据集资源的协助搬运代码
    ├── demo.py                               # 给用户演示怎么使用纯 Python 去驱动并启动语音清晰度的程序示例
    ├── download_process_dns4.sh              # 自动下载解压缩再批量按照模型策略预先处理国际大赛DNS4数据大包任务脚本
    ├── external_usage.py                     # 显示了其他项目如何在工程内引用其核心包提供极简单的降噪 API 调用示范代码
    ├── get_version.sh                        # 用于提供系统版本查询以及抽取以自动注入 CI CD 中各阶段的帮助类 shell
    ├── live_spectrum.py                      # 基于音频输入实时捕获后画出并且刷新时序频谱图像的图形化测试工具代码
    ├── perf_df_dec.sh / perf_enc.sh          # 多线程或者特定压降场景对部分解码器编码组件单独运行以取得性能跑分的 Shell
    ├── read_toml.py                          # 一个能够在终端利用纯 Python 打印查询当前 TOML 文件结构项值的快捷小命令
    ├── sbatch_test_voicebank.sh (和 train)   # 提供给超算 Slurm 集群自动调度分配测试、训练节点硬件所需的任务下发配属文件
    ├── set_batch_size.py                     # 帮助用户和脚本来集中全局查找并且修改训练所承载 Batch-Size 规模的更新文件
    ├── setup_env.sh                          # 对于刚刚 Clone 的环境一键解决关联支持拉取配置环境预装环境系统辅助文件
    ├── test_model_tract_cli.sh               # 使其在脱离 Python 外壳仅只用 Tract 命令执行下利用本地导出模型测算结果对比
    └── WAcc[_xx].py                          # 利用比如 Whisper 等系统做 ASR (语音识别)，对经过降噪或原生数据识别准确度计算校对用语
```
