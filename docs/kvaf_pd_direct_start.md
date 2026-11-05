# 手动字段启动 P / D

当前现场（已有 vLLM、141.62.38.12、P TP2 / D DP2、K 28997）请优先使用[更新后的完整教程](kvaf_pd_existing_vllm_tutorial.md)，包含当天问题汇总、编译检查点和诊断流程。

仓库根目录 p_start.sh、d_start.sh 按标准启动模板编排：环境设置在前，完整原生 vLLM 命令与内联 JSON 在后。参数直接填在字段内，无 Python 配置生成、自动网卡识别、JSON变量或启动包装脚本；无需预先 export 或传参。

默认 P 为物理卡 2/3、TP2/DP1、HTTP 8100；D 为卡 4/5、TP1/DP2、HTTP 8200；模型 /home/models/glm5-mtp-4l，IP 141.62.38.12、网卡 enp34s0f1，K 端口 28997。

```bash
cd /home/c00612283/vllm-ascend-v0.30.0rc2
./d_start.sh
```

待 D 就绪，在另一终端进入同一目录执行 ./p_start.sh。K 必须先启动；P/D 健康后启动 Proxy。退出用 Ctrl+C，不用 Ctrl+Z。语法检查使用 bash -n p_start.sh 和 bash -n d_start.sh，不再提供 --dry-run。

后续直接编辑：

- 网卡/IP：nic_name、local_ip；P 的 decode_endpoints 和 D 的 cpu_address 是独立原生字段，换地址时一并核对。
- 卡号：ASCEND_RT_VISIBLE_DEVICES；TP/DP/HTTP：直接修改相应命令行参数。
- 模型：serve 后的路径；SDK：PYTHONPATH 和 kvaf_config.build_dir 两处路径保持一致，并匹配脚本使用的 Python。
- 诊断：P/D 的 JSON 内 diagnostics_requests=0、verify_kv=false 默认关闭；改为2/true开启，详见 [诊断说明](kvaf_pd_diagnostics.md)。
- 其他原生参数：直接在命令中增加或修改，无需在顶部再声明同名变量。

日志保存到当前工作目录 pd-logs/p或d/时间_PID/server.log，plog 与 metrics 分角色隔离。脚本不自动杀进程或启动其他服务。

保留当前已验证的环境值、W4A4C8、MTP3、PP1、关闭本地 prefix、c8_enable_reshape_optim=false，以及外部 PDKVoffloadConnector；不照搬参考模板中的 Mooncake、KVPP、DSA CP、MTP1 或 prefix 开启配置。CANN/ATB 环境沿用现有容器，可按需在环境区手动添加现场 source 命令。两份脚本依然固定使用同一个 Python 调用原生 vLLM CLI，避免 SDK ABI 混用。
