# Python异常调试技巧

当Python程序报错时，使用以下技巧快速定位和解决问题。

## 1. 读懂错误信息

Python错误信息从下往上读：
```
Traceback (most recent call last):
  File "main.py", line 10, in <module>
    result = process_data(data)
  File "utils.py", line 5, in process_data
    return data["key"]
KeyError: 'key'
```
- 最后一行是错误类型和原因
- 往上是调用栈，找到你的代码位置

## 2. 使用pdb调试器

```python
# 在可疑位置插入断点
import pdb; pdb.set_trace()

# Python 3.7+ 更简洁的写法
breakpoint()
```

常用命令：
- `n`：执行下一行
- `s`：进入函数
- `p 变量名`：打印变量值
- `c`：继续执行

## 3. 使用try-except捕获详情

```python
import traceback

try:
    risky_operation()
except Exception as e:
    print(f"错误类型: {type(e).__name__}")
    print(f"错误信息: {e}")
    traceback.print_exc()  # 打印完整堆栈
```

## 4. 常见错误解决方案

| 错误类型 | 常见原因 | 解决方法 |
|----------|----------|----------|
| KeyError | 字典key不存在 | 使用.get()或先检查 |
| IndexError | 索引越界 | 检查列表长度 |
| TypeError | 类型不匹配 | 检查变量类型 |
| AttributeError | 属性不存在 | 检查对象类型和拼写 |

## 5. IDE调试功能

推荐使用VSCode或PyCharm的图形化调试器：
- 设置断点（点击行号左侧）
- 查看变量值
- 单步执行
- 条件断点
