# 直接启动 P / D

仓库根目录的 p_start.sh 和 d_start.sh 为独立、可编辑的部署入口，无需命令行参数或预先 export 变量。按手动启动模板组织：顶部配置、原生JSON、环境变量、完整vLLM启动命令；不调用自动启动包装脚本，不修改原 start_vllm_server.sh。

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

IP、网卡均手动填写：local_ip=141.62.38.12、nic_name=enp34s0f1，不自动探测或切换网卡。P 的 decode_endpoints 是手动JSON列表；D worker数或地址变化时同步修改。K 地址由 D 的 k_address 设置。

语法检查（不启动服务）：

```bash
bash -n p_start.sh
bash -n d_start.sh
```

默认诊断关闭。需要 KV 日志及字节检查时，在两份脚本顶部均改 diagnostics_requests=2、verify_kv=true，按 [诊断说明](kvaf_pd_diagnostics.md) 执行；校验影响性能。D data_parallel_size 改变时，同时调整 P decode_endpoints；此示例 D TP=1，端口从31000按worker递增。此配置用于单机验证，生产拓扑按原需求另配完整 D endpoint 和 K 映射。

保留 W4A4C8、MTP3、PP1、本地 prefix caching 关闭，以及 c8_enable_reshape_optim=false。不复制参考脚本的 PP2、42/36 层切分、Mooncake、MTP1 或 prefix caching 开启配置。FLASHCOMM1 等环境保留当前已验证值，均在脚本中显式列出；如需 CANN/ATB 初始化，可将 env_file 设为现场环境脚本路径，不强制加载 /root/.bashrc。
