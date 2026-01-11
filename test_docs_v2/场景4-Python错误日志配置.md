# Python错误日志配置指南

## 概述

良好的日志配置有助于问题排查和系统监控。本文介绍Python logging模块的最佳实践配置。

## 基础配置

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)
```

## 高级配置（使用配置字典）

```python
LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        },
        'json': {
            'class': 'pythonjsonlogger.jsonlogger.JsonFormatter',
            'format': '%(asctime)s %(levelname)s %(name)s %(message)s'
        }
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'standard',
            'level': 'DEBUG'
        },
        'file': {
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': 'logs/app.log',
            'maxBytes': 10485760,  # 10MB
            'backupCount': 5,
            'formatter': 'json'
        }
    },
    'loggers': {
        '': {
            'handlers': ['console', 'file'],
            'level': 'INFO'
        }
    }
}
```

## 日志级别说明

| 级别 | 数值 | 使用场景 |
|------|------|----------|
| DEBUG | 10 | 调试信息 |
| INFO | 20 | 正常运行信息 |
| WARNING | 30 | 警告信息 |
| ERROR | 40 | 错误信息 |
| CRITICAL | 50 | 严重错误 |

## 与ELK集成

配置Filebeat收集日志，发送到Elasticsearch进行集中管理和检索。
