# 外部实现

本目录保存外部实现的本地缓存，除本说明外默认不纳入 Git。当前引入官方 openpi 和固定提交的 LeRobot，用于 π0.5 PyTorch 后端部署；来源、许可证、版本与环境补丁见[部署说明](../docs/pi05-deployment.md)及 `configs/backends/pi05-runtime/sources.json`。

使用外部工作时，在相关研究卡或工程决议中记录来源 URL、许可证、准确 commit/tag、用途与本地修改。优先使用可锁定依赖或可复现获取方式；需要 vendor 或 submodule 时，单独记录方案，避免嵌套仓库被意外提交。
