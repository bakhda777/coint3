"""Модуль визуализации результатов торговой стратегии."""

from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import seaborn as sns
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np

# Настройка стиля
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


def create_performance_report(
    equity_curve: pd.Series,
    pnl_series: pd.Series,
    metrics: Dict[str, float],
    pair_counts: List[Tuple[str, int]],
    results_dir: Path,
    strategy_name: str = "CointegrationStrategy"
) -> None:
    """Создает комплексный отчет о производительности стратегии."""
    
    # Создаем директорию для результатов
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Создаем фигуру с 8 подграфиками (4x2)
    fig = plt.figure(figsize=(20, 20))
    fig.suptitle(f'{strategy_name} - Отчет о производительности', fontsize=16, fontweight='bold')
    
    # 1. Кривая капитала
    ax1 = plt.subplot(4, 2, 1)
    if not equity_curve.empty:
        equity_curve.plot(ax=ax1, linewidth=3, color='#2E86AB')
        ax1.fill_between(equity_curve.index, equity_curve.values, alpha=0.2, color='#2E86AB')
        
        # Добавляем аннотации для ключевых точек
        max_val = equity_curve.max()
        min_val = equity_curve.min()
        start_val = equity_curve.iloc[0]
        end_val = equity_curve.iloc[-1]
        
        ax1.scatter([equity_curve.index[0]], [start_val], color='green', s=100, zorder=5, label=f'Старт: {start_val:,.0f}')
        ax1.scatter([equity_curve.index[-1]], [end_val], color='red', s=100, zorder=5, label=f'Финиш: {end_val:,.0f}')
        
        if equity_curve.idxmax() != equity_curve.index[0] and equity_curve.idxmax() != equity_curve.index[-1]:
            ax1.scatter([equity_curve.idxmax()], [max_val], color='gold', s=100, zorder=5, label=f'Пик: {max_val:,.0f}')
        
        ax1.set_title('Кривая капитала (Equity Curve)', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Время')
        ax1.set_ylabel('Капитал ($)')
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=10)
        
        # Форматируем оси
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
    
    # 2. Накопленные P&L
    ax2 = plt.subplot(4, 2, 2)
    if not pnl_series.empty:
        cumulative_pnl = pnl_series.cumsum()
        cumulative_pnl.plot(ax=ax2, linewidth=3, color='#28A745')
        ax2.fill_between(cumulative_pnl.index, cumulative_pnl.values, alpha=0.2, color='#28A745')
        ax2.set_title('Накопленные P&L', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Время')
        ax2.set_ylabel('Накопленный P&L ($)')
        ax2.grid(True, alpha=0.3)
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
    
    # 3. Дневные P&L
    ax3 = plt.subplot(4, 2, 3)
    if not pnl_series.empty:
        colors = ['green' if x > 0 else 'red' for x in pnl_series.values]
        pnl_series.plot(ax=ax3, kind='bar', color=colors, alpha=0.7, width=0.8)
        ax3.set_title('Дневные P&L', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Время')
        ax3.set_ylabel('P&L ($)')
        ax3.grid(True, alpha=0.3)
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)
    
    # 4. Распределение P&L
    ax4 = plt.subplot(4, 2, 4)
    if not pnl_series.empty:
        pnl_series.hist(bins=30, ax=ax4, alpha=0.7, color='#A23B72', edgecolor='black')
        ax4.axvline(pnl_series.mean(), color='red', linestyle='--', linewidth=2, 
                   label=f'Среднее: ${pnl_series.mean():,.0f}')
        ax4.axvline(pnl_series.median(), color='blue', linestyle='--', linewidth=2, 
                   label=f'Медиана: ${pnl_series.median():,.0f}')
        ax4.set_title('Распределение дневных P&L', fontsize=14, fontweight='bold')
        ax4.set_xlabel('P&L ($)')
        ax4.set_ylabel('Частота')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    # 5. Drawdown
    ax5 = plt.subplot(4, 2, 5)
    if not equity_curve.empty:
        running_max = equity_curve.cummax()
        drawdown = (equity_curve - running_max) / running_max * 100
        drawdown.plot(ax=ax5, color='red', linewidth=2)
        ax5.fill_between(drawdown.index, drawdown.values, 0, alpha=0.3, color='red')
        ax5.set_title('Drawdown (%)', fontsize=14, fontweight='bold')
        ax5.set_xlabel('Время')
        ax5.set_ylabel('Drawdown (%)')
        ax5.grid(True, alpha=0.3)
        plt.setp(ax5.xaxis.get_majorticklabels(), rotation=45)
        
        # Отмечаем максимальную просадку
        max_dd_idx = drawdown.idxmin()
        max_dd_val = drawdown.min()
        ax5.scatter([max_dd_idx], [max_dd_val], color='darkred', s=100, zorder=5, 
                   label=f'Макс. просадка: {max_dd_val:.1f}%')
        ax5.legend()
    
    # 6. Количество пар по периодам
    ax6 = plt.subplot(4, 2, 6)
    if pair_counts:
        periods, counts = zip(*pair_counts)
        bars = ax6.bar(range(len(periods)), counts, color='#F18F01', alpha=0.7, edgecolor='black')
        ax6.set_title('Количество торгуемых пар по периодам', fontsize=14, fontweight='bold')
        ax6.set_xlabel('Период')
        ax6.set_ylabel('Количество пар')
        ax6.set_xticks(range(len(periods)))
        ax6.set_xticklabels([f'Период {i+1}' for i in range(len(periods))], rotation=45)
        ax6.grid(True, alpha=0.3)
        
        # Добавляем значения на столбцы
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{count}', ha='center', va='bottom', fontweight='bold')
    
    # 7. Таблица метрик (часть 1)
    ax7 = plt.subplot(4, 2, 7)
    ax7.axis('off')
    
    # Форматируем основные метрики
    main_metrics = []
    for key, value in metrics.items():
        if key in ['sharpe_ratio', 'max_drawdown', 'total_pnl', 'win_rate', 'avg_daily_pnl']:
            if key == 'sharpe_ratio':
                main_metrics.append(['Коэффициент Шарпа', f'{value:.4f}'])
            elif key == 'max_drawdown':
                main_metrics.append(['Максимальная просадка', f'${value:,.0f}'])
            elif key == 'total_pnl':
                main_metrics.append(['Общий P&L', f'${value:,.0f}'])
            elif key == 'win_rate':
                main_metrics.append(['% прибыльных дней', f'{value:.2%}'])
            elif key == 'avg_daily_pnl':
                main_metrics.append(['Средний дневной P&L', f'${value:,.0f}'])
    
    # Создаем таблицу основных метрик
    if main_metrics:
        table1 = ax7.table(cellText=main_metrics,
                          colLabels=['Основные метрики', 'Значение'],
                          cellLoc='left',
                          loc='center')
        table1.auto_set_font_size(False)
        table1.set_fontsize(10)
        table1.scale(1.2, 1.8)
    
    ax7.set_title('Основные метрики', fontsize=14, fontweight='bold')
    
    # 8. Таблица метрик по сделкам
    ax8 = plt.subplot(4, 2, 8)
    ax8.axis('off')
    
    # Форматируем метрики по сделкам
    trade_metrics = []
    for key, value in metrics.items():
        if key in ['total_trades', 'total_pairs_traded', 'avg_trades_per_pair', 'win_rate_trades', 'total_costs']:
            if key == 'total_trades':
                trade_metrics.append(['Всего сделок', f'{value:,}'])
            elif key == 'total_pairs_traded':
                trade_metrics.append(['Торгуемых пар', f'{value:,}'])
            elif key == 'avg_trades_per_pair':
                trade_metrics.append(['Сделок на пару', f'{value:.1f}'])
            elif key == 'win_rate_trades':
                trade_metrics.append(['Успешность сделок', f'{value:.2%}'])
            elif key == 'total_costs':
                trade_metrics.append(['Общие расходы', f'${value:,.0f}'])
    
    # Создаем таблицу метрик по сделкам
    if trade_metrics:
        table2 = ax8.table(cellText=trade_metrics,
                          colLabels=['Метрики сделок', 'Значение'],
                          cellLoc='left',
                          loc='center')
        table2.auto_set_font_size(False)
        table2.set_fontsize(10)
        table2.scale(1.2, 1.8)
    
    ax8.set_title('Статистика сделок', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(results_dir / f'{strategy_name}_performance_report.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"📊 Отчет сохранен: {results_dir / f'{strategy_name}_performance_report.png'}")


def create_interactive_report(
    equity_curve: pd.Series,
    pnl_series: pd.Series,
    results_dir: Path,
    strategy_name: str = "CointegrationStrategy"
) -> None:
    """Создает интерактивный HTML отчет с использованием Plotly."""
    
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        
        # Создаем субплоты
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Кривая капитала', 'Дневные P&L', 'Накопленные P&L', 'Drawdown'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Кривая капитала
        if not equity_curve.empty:
            fig.add_trace(
                go.Scatter(x=equity_curve.index, y=equity_curve.values,
                          mode='lines', name='Капитал', line=dict(color='blue', width=2)),
                row=1, col=1
            )
        
        # Дневные P&L
        if not pnl_series.empty:
            colors = ['green' if x > 0 else 'red' for x in pnl_series.values]
            fig.add_trace(
                go.Bar(x=pnl_series.index, y=pnl_series.values,
                      marker_color=colors, name='Дневной P&L', opacity=0.7),
                row=1, col=2
            )
            
            # Накопленные P&L
            cumulative_pnl = pnl_series.cumsum()
            fig.add_trace(
                go.Scatter(x=cumulative_pnl.index, y=cumulative_pnl.values,
                          mode='lines', name='Накопленный P&L', line=dict(color='green', width=2)),
                row=2, col=1
            )
        
        # Drawdown
        if not equity_curve.empty:
            running_max = equity_curve.cummax()
            drawdown = (equity_curve - running_max) / running_max * 100
            fig.add_trace(
                go.Scatter(x=drawdown.index, y=drawdown.values,
                          mode='lines', name='Drawdown (%)', 
                          line=dict(color='red', width=2), fill='tozeroy'),
                row=2, col=2
            )
        
        # Обновляем макет
        fig.update_layout(
            title=f'{strategy_name} - Интерактивный отчет',
            height=800,
            showlegend=True,
            template='plotly_white'
        )
        
        # Сохраняем HTML
        html_file = results_dir / f'{strategy_name}_interactive_report.html'
        fig.write_html(str(html_file))
        
        print(f"🌐 Интерактивный отчет сохранен: {html_file}")
        
    except ImportError:
        print("⚠️  Plotly не установлен, интерактивный отчет пропущен")


def format_metrics_summary(metrics: Dict[str, Any]) -> str:
    """Форматирует метрики в красивую строку для вывода."""
    
    lines = [
        "🎯 " + "="*60,
        "📊 ИТОГИ ТОРГОВОЙ СТРАТЕГИИ",
        "🎯 " + "="*60,
        "",
        "💰 ФИНАНСОВЫЕ РЕЗУЛЬТАТЫ:",
        f"   • Общий P&L:              {metrics.get('total_pnl', 0):>15,.0f}",
        f"   • Максимальная просадка:  {metrics.get('max_drawdown', 0):>15,.0f}",
        f"   • Средний дневной P&L:    {metrics.get('avg_daily_pnl', 0):>15,.0f}",
        "",
        "📈 МЕТРИКИ РИСКА:",
        f"   • Коэффициент Шарпа:      {metrics.get('sharpe_ratio', 0):>15.4f}",
        f"   • Волатильность P&L:      {metrics.get('volatility', 0):>15.2f}",
        f"   • Процент прибыльных дней: {metrics.get('win_rate', 0):>14.2%}",
        "",
        "🔄 ТОРГОВАЯ АКТИВНОСТЬ:",
        f"   • Всего торговых дней:    {metrics.get('total_days', 0):>15,}",
        f"   • Всего сделок:           {metrics.get('total_trades', 0):>15,}",
        f"   • Торгуемых пар:          {metrics.get('total_pairs_traded', 0):>15,}",
        f"   • Сделок на пару:         {metrics.get('avg_trades_per_pair', 0):>15.1f}",
        f"   • Успешность сделок:      {metrics.get('win_rate_trades', 0):>14.2%}",
        "",
        "💸 РАСХОДЫ И ЭКСТРЕМУМЫ:",
        f"   • Общие расходы:          {metrics.get('total_costs', 0):>15,.0f}",
        f"   • Максимальная прибыль:   {metrics.get('max_single_gain', 0):>15,.0f}",
        f"   • Максимальный убыток:    {metrics.get('max_single_loss', 0):>15,.0f}",
        f"   • Лучшая пара (P&L):      {metrics.get('best_pair_pnl', 0):>15,.0f}",
        f"   • Худшая пара (P&L):      {metrics.get('worst_pair_pnl', 0):>15,.0f}",
        "",
        "📊 ОЦЕНКА СТРАТЕГИИ:",
    ]
    
    # Добавляем оценку
    sharpe = metrics.get('sharpe_ratio', 0)
    total_pnl = metrics.get('total_pnl', 0)
    
    if sharpe > 0.5 and total_pnl > 0:
        lines.append("   ✅ ОТЛИЧНАЯ стратегия! Высокий Шарп и положительный результат")
    elif sharpe > 0.2 and total_pnl > 0:
        lines.append("   ✅ ХОРОШАЯ стратегия! Положительный результат с разумным риском")
    elif total_pnl > 0:
        lines.append("   ⚠️  ПРИЕМЛЕМАЯ стратегия. Прибыльна, но высокий риск")
    else:
        lines.append("   ❌ УБЫТОЧНАЯ стратегия. Требует доработки")
    
    lines.extend([
        "",
        "🎯 " + "="*60,
        ""
    ])
    
    return "\n".join(lines)


def calculate_extended_metrics(pnl_series: pd.Series, equity_curve: pd.Series) -> Dict[str, Any]:
    """Рассчитывает расширенные метрики для анализа."""
    
    if pnl_series.empty:
        return {}
    
    metrics = {}
    
    # Базовые метрики
    metrics['total_days'] = len(pnl_series)
    metrics['avg_daily_pnl'] = pnl_series.mean()
    metrics['volatility'] = pnl_series.std()
    metrics['win_rate'] = (pnl_series > 0).mean()
    metrics['max_single_gain'] = pnl_series.max()
    metrics['max_single_loss'] = pnl_series.min()
    
    return metrics 