# 直接启动 P / D

仓库根目录的 p_start.sh 和 d_start.sh 为独立、可编辑的部署入口，无需命令行参数或预先 export 变量。沿用既有 start_vllm_pd.sh 的原生校验和启动配置，不修改原 start_vllm_server.sh。

当前默认：模型 /home/models/glm5-mtp-4l，Python /usr/local/python3.12.13/bin/python3，KVAF binding /home/c00612283/KVAF-Devlop/build-pd-py312；P TP2/DP1 使用卡 2、3，HTTP 8100；D TP1/DP2 使用卡 4、5，HTTP 8200；K 位于同机 28997。只修改每个脚本顶部配置即可。

```bash
cd /home/c00612283/vllm-ascend-v0.30.0rc2
./d_start.sh
```

待 D 初始化就绪，另一终端执行：

```bash
cd /home/c00612283/vllm-ascend-v0.30.0rc2
./p_start.sh
```

保留前台终端，正常退出用 Ctrl+C，不用 Ctrl+Z。K 必须先启动，Proxy 在 P/D 健康后启动；此入口不后台拉起或终止其他服务。日志保存到仓库 pd-logs/p或d/时间_PID/server.log，plog 和 metrics 各实例隔离。

脚本使用 Python 标准库自动获取网卡 IPv4，不依赖 ip/ss。优先使用顶部 preferred_nic=enp34s0f1，有 IPv4 时使用该地址，否则取默认路由网卡；最终选择会在启动时打印。容器须提供正确的通信网卡/网络。多网卡环境请核对所选地址，需要时将 local_ip 填成实际地址（如 141.62.38.12）；指定地址必须属于容器可见网卡，不盲目硬编码。

仅打印命令、不加载模型或 SDK：

```bash
./d_start.sh --dry-run
./p_start.sh --dry-run
```

默认诊断关闭。需要 KV 日志及字节检查时，在两份脚本顶部均改 diagnostics_requests=2、verify_kv=true，按 [诊断说明](kvaf_pd_diagnostics.md) 执行；校验影响性能。D dp 改变时，同时调整 P decode_workers；此示例 D TP=1，端口从31000按worker递增。此配置用于单机验证，生产拓扑按原需求另配完整 D endpoint 和 K 映射。

保留 W4A4C8、MTP3、PP1、本地 prefix caching 关闭，以及 c8_enable_reshape_optim=false。不复制参考脚本的 PP2、42/36 层切分、Mooncake、MTP1 或 prefix caching 开启配置。FLASHCOMM1 等环境沿用当前已验证启动入口；如需 CANN/ATB 初始化，可将 env_file 设为现场环境脚本路径，不强制加载 /root/.bashrc。
