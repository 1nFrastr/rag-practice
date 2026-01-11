# Python语法错误大全

本文收集整理了Python中常见的语法错误（SyntaxError），供开发者参考。

## 1. 缩进错误

```python
# 错误示例
def foo():
print("hello")  # IndentationError: expected an indented block

# 正确写法
def foo():
    print("hello")
```

## 2. 冒号缺失

```python
# 错误示例
if x > 0  # SyntaxError: invalid syntax
    print(x)

# 正确写法
if x > 0:
    print(x)
```

## 3. 括号不匹配

```python
# 错误示例
result = (1 + 2 * 3  # SyntaxError: unexpected EOF

# 正确写法
result = (1 + 2) * 3
```

## 4. 字符串引号不匹配

```python
# 错误示例
message = "Hello'  # SyntaxError: EOL while scanning string literal

# 正确写法
message = "Hello"
```

## 5. 使用保留字作为变量名

```python
# 错误示例
class = "Math"  # SyntaxError: invalid syntax

# 正确写法
class_name = "Math"
```

## 6. f-string语法错误

```python
# 错误示例
name = "Alice"
print(f"Hello, {name")  # SyntaxError

# 正确写法
print(f"Hello, {name}")
```

## 7. Python 2/3 兼容问题

```python
# Python 2 写法（Python 3 报错）
print "Hello"  # SyntaxError: Missing parentheses in call to 'print'

# Python 3 正确写法
print("Hello")
```

## 8. 赋值与比较混淆

```python
# 错误示例（在条件中使用 = 而非 ==）
if x = 5:  # SyntaxError: invalid syntax

# 正确写法
if x == 5:
```

## 9. 无效的函数参数

```python
# 错误示例
def foo(a, *args, b):  # Python 3.0之前会报错
    pass

# 注意：Python 3.0+ 这是合法的keyword-only参数
```

## 语法错误检查工具

- **pylint**：静态代码分析
- **pyflakes**：快速语法检查
- **mypy**：类型检查
