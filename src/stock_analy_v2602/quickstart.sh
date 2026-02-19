#!/bin/bash
# 股票分析系统 - 快速启动脚本

echo "========================================"
echo "股票分析系统 - 快速启动"
echo "========================================"

# 进入项目目录
cd /home/claude/src/stock_analyzer

# 检查是否已安装依赖
echo ""
echo "检查依赖包..."

if python3 -c "import akshare" 2>/dev/null; then
    echo "✓ 依赖包已安装"
else
    echo "✗ 依赖包未安装，开始安装..."
    pip install -r requirements.txt --break-system-packages
fi

echo ""
echo "========================================"
echo "系统已准备就绪！"
echo "========================================"
echo ""
echo "使用方法："
echo ""
echo "1. 修改配置文件 config/config.py（设置MySQL密码）"
echo ""
echo "2. 命令行使用："
echo "   # 下载日线数据"
echo "   python main.py --action download_daily --stock 000001"
echo ""
echo "   # 批量下载"
echo "   python main.py --action download_daily --stocks '000001,600000,600036'"
echo ""
echo "   # 增量更新"
echo "   python main.py --action update_daily --stock 000001"
echo ""
echo "   # 下载分钟线"
echo "   python main.py --action download_minute --stock 000001 --period 5"
echo ""
echo "   # 查看统计"
echo "   python main.py --action stats --stock 000001"
echo ""
echo "3. Python代码使用："
echo "   python examples.py"
echo ""
echo "详细文档请查看 README.md"
echo "========================================"
