#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
股票分析系统 - 快速使用示例
演示如何使用系统的主要功能
"""

import sys
sys.path.append('/home/claude/src/stock_analyzer')

from main import StockAnalyzer


def example_basic_usage():
    """示例1: 基本使用"""
    print("\n" + "="*60)
    print("示例1: 基本使用 - 下载单只股票日线数据")
    print("="*60)
    
    # 初始化系统（使用AKShare数据源）
    analyzer = StockAnalyzer(data_source='akshare')
    
    # 下载平安银行(000001)最近1年的日线数据
    analyzer.download_stock_daily(
        stock_code='000001',
        start_date='2023-01-01',
        end_date='2024-02-15'
    )
    
    # 查看统计信息
    analyzer.show_stats('000001')
    
    # 关闭系统
    analyzer.close()


def example_batch_download():
    """示例2: 批量下载多只股票"""
    print("\n" + "="*60)
    print("示例2: 批量下载多只股票的日线数据")
    print("="*60)
    
    analyzer = StockAnalyzer(data_source='akshare')
    
    # 批量下载银行股
    stock_list = [
        '000001',  # 平安银行
        '600000',  # 浦发银行
        '600036',  # 招商银行
        '601398',  # 工商银行
    ]
    
    results = analyzer.batch_download_daily(
        stock_codes=stock_list,
        start_date='2023-01-01'
    )
    
    print(f"\n下载结果: 成功 {results['success']} 只，失败 {results['failed']} 只")
    
    analyzer.close()


def example_download_minute():
    """示例3: 下载分钟线数据"""
    print("\n" + "="*60)
    print("示例3: 下载分钟线数据")
    print("="*60)
    
    analyzer = StockAnalyzer(data_source='akshare')
    
    # 下载1分钟线数据
    analyzer.download_stock_minute(
        stock_code='000001',
        period='1',  # 1分钟
        start_date='2024-02-01',
        end_date='2024-02-15'
    )
    
    # 下载5分钟线数据
    analyzer.download_stock_minute(
        stock_code='000001',
        period='5',  # 5分钟
        start_date='2024-02-01',
        end_date='2024-02-15'
    )
    
    # 查看统计
    analyzer.show_stats('000001')
    
    analyzer.close()


def example_incremental_update():
    """示例4: 增量更新"""
    print("\n" + "="*60)
    print("示例4: 增量更新数据")
    print("="*60)
    
    analyzer = StockAnalyzer(data_source='akshare')
    
    # 首次下载（如果数据库中没有数据）
    print("\n第一步: 首次下载历史数据...")
    analyzer.download_stock_daily(
        stock_code='000001',
        start_date='2024-01-01'
    )
    
    # 增量更新（只下载最新的数据）
    print("\n第二步: 增量更新最新数据...")
    analyzer.update_stock_daily('000001')
    
    # 查看更新后的统计
    analyzer.show_stats('000001')
    
    analyzer.close()


def example_batch_update():
    """示例5: 批量增量更新"""
    print("\n" + "="*60)
    print("示例5: 批量增量更新多只股票")
    print("="*60)
    
    analyzer = StockAnalyzer(data_source='akshare')
    
    # 批量更新
    stock_list = ['000001', '600000', '600036', '601398']
    
    results = analyzer.batch_update_daily(stock_list)
    
    print(f"\n更新结果: 成功 {results['success']} 只，失败 {results['failed']} 只")
    
    # 查看每只股票的统计
    for stock_code in stock_list:
        analyzer.show_stats(stock_code)
    
    analyzer.close()


def example_all_features():
    """示例6: 完整工作流程"""
    print("\n" + "="*60)
    print("示例6: 完整工作流程演示")
    print("="*60)
    
    analyzer = StockAnalyzer(data_source='akshare')
    
    stock_code = '000001'
    
    # 步骤1: 下载历史日线数据
    print("\n步骤1: 下载历史日线数据...")
    analyzer.download_stock_daily(
        stock_code=stock_code,
        start_date='2023-01-01',
        end_date='2024-02-15'
    )
    
    # 步骤2: 下载分钟线数据
    print("\n步骤2: 下载分钟线数据...")
    analyzer.download_stock_minute(
        stock_code=stock_code,
        period='5',
        start_date='2024-02-01',
        end_date='2024-02-15'
    )
    
    # 步骤3: 查看统计
    print("\n步骤3: 查看数据统计...")
    analyzer.show_stats(stock_code)
    
    # 步骤4: 增量更新
    print("\n步骤4: 增量更新到最新...")
    analyzer.update_stock_daily(stock_code)
    analyzer.update_stock_minute(stock_code, period='5')
    
    # 步骤5: 再次查看统计
    print("\n步骤5: 查看更新后的统计...")
    analyzer.show_stats(stock_code)
    
    analyzer.close()


if __name__ == '__main__':
    print("\n" + "="*60)
    print("股票分析系统 - 使用示例")
    print("="*60)
    
    # 运行示例（可以选择运行其中一个或多个示例）
    
    # 示例1: 基本使用
    example_basic_usage()
    
    # 示例2: 批量下载（注释掉以节省时间）
    # example_batch_download()
    
    # 示例3: 下载分钟线（注释掉以节省时间）
    # example_download_minute()
    
    # 示例4: 增量更新
    # example_incremental_update()
    
    # 示例5: 批量增量更新（注释掉以节省时间）
    # example_batch_update()
    
    # 示例6: 完整工作流程（注释掉以节省时间）
    # example_all_features()
    
    print("\n" + "="*60)
    print("示例运行完成！")
    print("="*60)
