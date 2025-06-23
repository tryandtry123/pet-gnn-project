"""
改进的日志系统
提供统一的日志配置、轮转、结构化日志等功能
"""

import logging
import logging.handlers
import os
import json
from datetime import datetime
from pathlib import Path
import traceback


class StructuredFormatter(logging.Formatter):
    """结构化日志格式器"""
    
    def format(self, record):
        # 创建结构化日志数据
        log_data = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
        }
        
        # 添加异常信息
        if record.exc_info:
            log_data['exception'] = {
                'type': record.exc_info[0].__name__ if record.exc_info[0] else None,
                'message': str(record.exc_info[1]) if record.exc_info[1] else None,
                'traceback': traceback.format_exception(*record.exc_info)
            }
        
        # 添加额外字段
        if hasattr(record, 'extra_fields'):
            log_data.update(record.extra_fields)
        
        return json.dumps(log_data, ensure_ascii=False, default=str)


class ColoredFormatter(logging.Formatter):
    """彩色控制台日志格式器"""
    
    # 颜色代码
    COLORS = {
        'DEBUG': '\033[36m',    # 青色
        'INFO': '\033[32m',     # 绿色
        'WARNING': '\033[33m',  # 黄色
        'ERROR': '\033[31m',    # 红色
        'CRITICAL': '\033[35m', # 紫色
    }
    RESET = '\033[0m'
    
    def format(self, record):
        # 添加颜色
        color = self.COLORS.get(record.levelname, '')
        record.levelname = f"{color}{record.levelname}{self.RESET}"
        
        # 格式化消息
        formatted = super().format(record)
        return formatted


def setup_logging(
    log_level='INFO',
    log_dir='logs',
    app_name='pet_gnn',
    enable_console=True,
    enable_file=True,
    enable_structured=False,
    max_file_size=10*1024*1024,  # 10MB
    backup_count=5
):
    """
    设置完整的日志系统
    
    Args:
        log_level: 日志级别
        log_dir: 日志目录
        app_name: 应用名称
        enable_console: 是否启用控制台输出
        enable_file: 是否启用文件输出
        enable_structured: 是否启用结构化日志
        max_file_size: 单个日志文件最大大小
        backup_count: 备份文件数量
    """
    
    # 创建日志目录
    log_dir = Path(log_dir)
    log_dir.mkdir(exist_ok=True)
    
    # 获取根日志器
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # 清除现有的处理器
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    handlers = []
    
    # 控制台处理器
    if enable_console:
        console_handler = logging.StreamHandler()
        if os.name == 'nt':  # Windows系统
            console_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        else:  # Unix系统支持颜色
            console_formatter = ColoredFormatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        console_handler.setFormatter(console_formatter)
        handlers.append(console_handler)
    
    # 文件处理器
    if enable_file:
        # 普通日志文件
        file_path = log_dir / f'{app_name}.log'
        file_handler = logging.handlers.RotatingFileHandler(
            file_path,
            maxBytes=max_file_size,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        handlers.append(file_handler)
        
        # 错误日志文件
        error_file_path = log_dir / f'{app_name}_error.log'
        error_handler = logging.handlers.RotatingFileHandler(
            error_file_path,
            maxBytes=max_file_size,
            backupCount=backup_count,
            encoding='utf-8'
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(file_formatter)
        handlers.append(error_handler)
        
        # 结构化日志文件
        if enable_structured:
            struct_file_path = log_dir / f'{app_name}_structured.log'
            struct_handler = logging.handlers.RotatingFileHandler(
                struct_file_path,
                maxBytes=max_file_size,
                backupCount=backup_count,
                encoding='utf-8'
            )
            struct_handler.setFormatter(StructuredFormatter())
            handlers.append(struct_handler)
    
    # 添加所有处理器
    for handler in handlers:
        root_logger.addHandler(handler)
    
    # 配置特定模块的日志级别
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    
    # 记录配置信息
    logger = logging.getLogger(__name__)
    logger.info(f"日志系统初始化完成")
    logger.info(f"  日志级别: {log_level}")
    logger.info(f"  日志目录: {log_dir.absolute()}")
    logger.info(f"  控制台输出: {enable_console}")
    logger.info(f"  文件输出: {enable_file}")
    logger.info(f"  结构化日志: {enable_structured}")
    
    return logger


def get_logger(name, extra_fields=None):
    """
    获取带有额外字段的日志器
    
    Args:
        name: 日志器名称
        extra_fields: 额外字段字典
    
    Returns:
        logging.Logger: 配置好的日志器
    """
    logger = logging.getLogger(name)
    
    if extra_fields:
        # 创建适配器来添加额外字段
        class ExtraFieldsAdapter(logging.LoggerAdapter):
            def process(self, msg, kwargs):
                if 'extra' not in kwargs:
                    kwargs['extra'] = {}
                kwargs['extra']['extra_fields'] = self.extra
                return msg, kwargs
        
        return ExtraFieldsAdapter(logger, extra_fields)
    
    return logger


class PerformanceLogger:
    """性能监控日志器"""
    
    def __init__(self, logger_name='performance'):
        self.logger = get_logger(logger_name)
    
    def __enter__(self):
        self.start_time = datetime.now()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = (datetime.now() - self.start_time).total_seconds()
        if exc_type:
            self.logger.error(f"操作失败，耗时: {duration:.3f}秒")
        else:
            self.logger.info(f"操作完成，耗时: {duration:.3f}秒")
    
    def log_step(self, step_name, start_time=None):
        """记录步骤耗时"""
        if start_time:
            duration = (datetime.now() - start_time).total_seconds()
            self.logger.info(f"{step_name} 完成，耗时: {duration:.3f}秒")
        else:
            self.logger.info(f"{step_name} 开始")
            return datetime.now()


def log_function_call(logger=None):
    """函数调用日志装饰器"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            func_logger = logger or logging.getLogger(func.__module__)
            func_logger.debug(f"调用函数 {func.__name__}，参数: args={args}, kwargs={kwargs}")
            
            try:
                result = func(*args, **kwargs)
                func_logger.debug(f"函数 {func.__name__} 执行成功")
                return result
            except Exception as e:
                func_logger.error(f"函数 {func.__name__} 执行失败: {e}")
                raise
        
        return wrapper
    return decorator


def log_performance(logger=None):
    """性能监控装饰器"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            func_logger = logger or logging.getLogger(func.__module__)
            start_time = datetime.now()
            
            try:
                result = func(*args, **kwargs)
                duration = (datetime.now() - start_time).total_seconds()
                func_logger.info(f"函数 {func.__name__} 执行完成，耗时: {duration:.3f}秒")
                return result
            except Exception as e:
                duration = (datetime.now() - start_time).total_seconds()
                func_logger.error(f"函数 {func.__name__} 执行失败，耗时: {duration:.3f}秒，错误: {e}")
                raise
        
        return wrapper
    return decorator


# 预配置的日志器实例
class LoggerFactory:
    """日志器工厂类"""
    
    _initialized = False
    
    @classmethod
    def setup(cls, **kwargs):
        """初始化日志系统"""
        if not cls._initialized:
            setup_logging(**kwargs)
            cls._initialized = True
    
    @classmethod
    def get_logger(cls, name, **kwargs):
        """获取日志器"""
        if not cls._initialized:
            cls.setup()
        return get_logger(name, **kwargs)
    
    @classmethod
    def get_performance_logger(cls, name='performance'):
        """获取性能日志器"""
        if not cls._initialized:
            cls.setup()
        return PerformanceLogger(name)


# 默认初始化
if not hasattr(setup_logging, '_called'):
    try:
        setup_logging(
            log_level='INFO',
            log_dir='logs',
            app_name='pet_gnn',
            enable_console=True,
            enable_file=True,
            enable_structured=False
        )
        setup_logging._called = True
    except Exception as e:
        print(f"日志系统初始化失败: {e}")


# 便捷接口
def debug(msg, **kwargs):
    """调试日志"""
    logging.getLogger('pet_gnn').debug(msg, **kwargs)

def info(msg, **kwargs):
    """信息日志"""
    logging.getLogger('pet_gnn').info(msg, **kwargs)

def warning(msg, **kwargs):
    """警告日志"""
    logging.getLogger('pet_gnn').warning(msg, **kwargs)

def error(msg, **kwargs):
    """错误日志"""
    logging.getLogger('pet_gnn').error(msg, **kwargs)

def critical(msg, **kwargs):
    """严重错误日志"""
    logging.getLogger('pet_gnn').critical(msg, **kwargs)


# 向后兼容函数
def setup_logger(name=None, level=logging.INFO, log_file=None):
    """
    向后兼容的日志设置函数
    保持与旧代码的兼容性
    
    Args:
        name: 记录器名称
        level: 日志级别
        log_file: 日志文件路径 (可选)
    
    Returns:
        logger: 配置好的记录器
    """
    # 如果还没有初始化过主日志系统，先初始化
    if not hasattr(setup_logging, '_called'):
        try:
            setup_logging(
                log_level=logging.getLevelName(level),
                log_dir='logs',
                app_name='pet_gnn',
                enable_console=True,
                enable_file=True
            )
        except:
            pass  # 忽略重复初始化错误
    
    # 返回指定名称的日志器
    logger_name = name or 'pet_gnn'
    return get_logger(logger_name) 