# 贡献指南

感谢你对 AirControl 项目的关注！我们欢迎任何形式的贡献，包括但不限于：

- 🐛 报告 Bug
- 💡 提出新功能建议
- 📝 改进文档
- 🔧 提交代码修复
- ✨ 添加新功能

## 如何贡献

### 1. 报告问题

如果你发现了 Bug 或有功能建议，请创建 Issue：

1. 点击 [Issues](https://github.com/enixz/AirControl/issues) 页面
2. 点击 "New Issue" 按钮
3. 选择合适的模板（Bug 报告或功能请求）
4. 填写详细信息，包括：
   - 问题描述
   - 复现步骤
   - 预期行为
   - 实际行为
   - 环境信息（操作系统、Python 版本等）

### 2. 提交代码

#### 2.1 Fork 项目

1. 点击项目右上角的 "Fork" 按钮
2. 克隆你的 Fork 到本地：
   ```bash
   git clone https://github.com/your-username/AirControl.git
   cd AirControl
   ```

#### 2.2 创建分支

```bash
# 创建并切换到新分支
git checkout -b feature/your-feature-name

# 或者修复 Bug
git checkout -b fix/your-bug-fix
```

#### 2.3 开发环境

```bash
# 创建虚拟环境
python -m venv .venv

# 激活虚拟环境（Windows）
.venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt

# 运行测试确保环境正常
python -m pytest tests/
```

#### 2.4 编写代码

- 遵循 PEP 8 代码规范
- 添加类型注解
- 编写清晰的注释
- 保持函数简洁（单一职责）

#### 2.5 编写测试

为你的代码编写单元测试：

```bash
# 运行所有测试
python -m pytest tests/

# 运行特定测试
python -m pytest tests/test_your_feature.py
```

#### 2.6 提交更改

```bash
# 添加更改
git add .

# 提交更改（使用语义化提交信息）
git commit -m "feat: add your feature description"

# 推送到你的 Fork
git push origin feature/your-feature-name
```

#### 2.7 创建 Pull Request

1. 点击 "Compare & pull request" 按钮
2. 填写 PR 描述，包括：
   - 更改内容
   - 相关 Issue 编号
   - 测试情况
3. 等待代码审查
4. 根据反馈进行修改

### 3. 提交信息规范

使用 [Conventional Commits](https://www.conventionalcommits.org/) 规范：

```
<type>(<scope>): <subject>

<body>

<footer>
```

**类型（type）**：
- `feat`: 新功能
- `fix`: Bug 修复
- `docs`: 文档更新
- `style`: 代码格式调整（不影响功能）
- `refactor`: 代码重构
- `perf`: 性能优化
- `test`: 测试相关
- `chore`: 构建/工具相关

**示例**：
```
feat(mouse): add edge acceleration feature

- Add non-linear cubic acceleration for screen edges
- Configure dead zones for Y-axis canvas
- Update UI settings panel

Closes #123
```

### 4. 代码规范

#### Python 风格

- 遵循 [PEP 8](https://peps.python.org/pep-0008/)
- 使用 4 空格缩进
- 行长度限制：88 字符（Black 默认）
- 使用类型注解

#### 命名规范

- 类名：`PascalCase`
- 函数/方法：`snake_case`
- 常量：`UPPER_SNAKE_CASE`
- 私有方法：`_leading_underscore`

#### 文档字符串

```python
def process_frame(frame: np.ndarray) -> dict:
    """处理单帧图像并返回手势识别结果。

    Args:
        frame: BGR 格式的图像数组

    Returns:
        包含手势信息的字典，格式如下：
        {
            "gesture": str,
            "confidence": float,
            "landmarks": list
        }

    Raises:
        ValueError: 当图像格式不正确时
    """
    pass
```

### 5. 开发流程

#### 5.1 手势识别开发

1. 在 `app/services/gesture_recognizer.py` 中添加新手势
2. 更新 `app/modes/` 中对应模式的处理逻辑
3. 在 `config.json` 中添加映射配置
4. 编写测试用例

#### 5.2 语音命令开发

1. 在 `app/voice_keywords/` 中添加关键词
2. 更新 `app/services/voice_command.py` 的命令映射
3. 测试离线和在线识别

#### 5.3 UI 开发

1. 在 `app/main_ui.py` 中修改界面
2. 更新 `app/config_manager.py` 的配置项
3. 确保响应式布局

### 6. 测试指南

#### 运行测试

```bash
# 运行所有测试
python -m pytest tests/

# 运行带覆盖率的测试
python -m pytest tests/ --cov=app

# 运行特定测试文件
python -m pytest tests/test_edge_map.py

# 运行特定测试用例
python -m pytest tests/test_edge_map.py::TestEdgeMap::test_cubic_acceleration
```

#### 编写测试

```python
import pytest
from app.services.mouse_controller import MouseController

class TestMouseController:
    """鼠标控制器测试。"""

    def test_edge_acceleration(self):
        """测试边缘加速功能。"""
        controller = MouseController()
        # 测试逻辑
        assert controller._edge_map(0.9) > controller._edge_map(0.5)

    @pytest.mark.parametrize("input_val,expected", [
        (0.0, 0.0),
        (0.5, 0.5),
        (1.0, 1.0),
    ])
    def test_edge_map_values(self, input_val, expected):
        """测试边缘映射的边界值。"""
        controller = MouseController()
        result = controller._edge_map(input_val)
        assert abs(result - expected) < 0.01
```

### 7. 文档贡献

#### README 更新

- 保持中英文双语
- 使用清晰的标题层级
- 添加代码示例
- 包含截图或 GIF（如果适用）

#### 代码注释

- 解释复杂的算法
- 标注 TODO 和 FIXME
- 记录重要的设计决策

### 8. 社区准则

- 尊重他人
- 建设性讨论
- 包容不同观点
- 避免人身攻击

### 9. 许可证

贡献的代码将采用与项目相同的 [Apache License 2.0](LICENSE) 许可证。

### 10. 联系方式

- GitHub Issues：[Issues 页面](https://github.com/enixz/AirControl/issues)
- 邮箱：[your-email@example.com]

---

感谢你的贡献！🎉
