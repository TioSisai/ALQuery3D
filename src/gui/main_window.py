#!/usr/bin/env python3
"""
ALQuery3D 主窗口 - 深色主题的embedding生成器界面
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QLabel, QComboBox, QSpinBox, QSlider,
                             QDoubleSpinBox, QPushButton, QScrollArea, QFrame,
                             QGridLayout, QGroupBox, QSplitter, QMessageBox)
from PyQt5.QtCore import Qt, pyqtSignal, QThread
from PyQt5.QtGui import QPalette, QColor, QFont
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D

from src.data.embedding_generator import EmbeddingGenerator


class EmbeddingGeneratorThread(QThread):
    """后台生成embedding的线程"""
    finished = pyqtSignal(np.ndarray, np.ndarray)  # embeddings, labels
    error = pyqtSignal(str)

    def __init__(self, generator_params):
        super().__init__()
        self.generator_params = generator_params

    def run(self):
        try:
            generator = EmbeddingGenerator(embedding_dim=128, random_state=42)
            embeddings, labels = generator.generate_clustered_embeddings(**self.generator_params)

            # 降维到3D
            reduced_3d = generator.reduce_dimensions(n_components=3, method='pca')

            self.finished.emit(reduced_3d, labels)
        except Exception as e:
            self.error.emit(str(e))


class ClassParameterWidget(QWidget):
    """单个类别参数设置组件"""

    def __init__(self, class_id):
        super().__init__()
        self.class_id = class_id
        self.setupUI()

    def setupUI(self):
        layout = QVBoxLayout()

        # 类别标题
        title = QLabel(f"Class {self.class_id}")
        title.setFont(QFont("Arial", 12, QFont.Bold))
        title.setStyleSheet("color: #64FFDA; margin-bottom: 10px;")
        layout.addWidget(title)

        # 创建参数控制网格
        grid = QGridLayout()

        # 样本数量
        grid.addWidget(QLabel("Samples:"), 0, 0)
        self.samples_spinbox = QSpinBox()
        self.samples_spinbox.setRange(10, 500)
        self.samples_spinbox.setValue(100)
        self.samples_spinbox.setStyleSheet(self.get_spinbox_style())
        grid.addWidget(self.samples_spinbox, 0, 1)

        # 分散度
        self.dispersion_input, self.dispersion_slider = self.create_parameter_control(
            "Dispersion:", 0.0, 1.0, 0.5, 1
        )
        grid.addWidget(QLabel("Dispersion:"), 1, 0)
        grid.addWidget(self.dispersion_input, 1, 1)
        grid.addWidget(self.dispersion_slider, 1, 2)

        # 曲度
        self.curvature_input, self.curvature_slider = self.create_parameter_control(
            "Curvature:", 0.0, 1.0, 0.2, 2
        )
        grid.addWidget(QLabel("Curvature:"), 2, 0)
        grid.addWidget(self.curvature_input, 2, 1)
        grid.addWidget(self.curvature_slider, 2, 2)

        # 扁平度
        self.flatness_input, self.flatness_slider = self.create_parameter_control(
            "Flatness:", 0.0, 1.0, 0.7, 3
        )
        grid.addWidget(QLabel("Flatness:"), 3, 0)
        grid.addWidget(self.flatness_input, 3, 1)
        grid.addWidget(self.flatness_slider, 3, 2)

        # 类内相关性
        self.correlation_input, self.correlation_slider = self.create_parameter_control(
            "Correlation:", 0.0, 1.0, 0.3, 4
        )
        grid.addWidget(QLabel("Correlation:"), 4, 0)
        grid.addWidget(self.correlation_input, 4, 1)
        grid.addWidget(self.correlation_slider, 4, 2)

        layout.addLayout(grid)

        # 设置整体样式
        self.setLayout(layout)
        self.setStyleSheet("""
            QWidget {
                background-color: #2B2B2B;
                border: 1px solid #555555;
                border-radius: 8px;
                padding: 10px;
                margin: 5px;
            }
            QLabel {
                color: #FFFFFF;
                border: none;
                margin: 2px;
            }
        """)

    def create_parameter_control(self, name, min_val, max_val, default_val, row):
        """创建参数输入框和滑块的联动控制"""
        # 输入框
        input_box = QDoubleSpinBox()
        input_box.setRange(min_val, max_val)
        input_box.setSingleStep(0.01)
        input_box.setDecimals(2)
        input_box.setValue(default_val)
        input_box.setStyleSheet(self.get_spinbox_style())

        # 滑块
        slider = QSlider(Qt.Horizontal)
        slider.setRange(int(min_val * 100), int(max_val * 100))
        slider.setValue(int(default_val * 100))
        slider.setStyleSheet(self.get_slider_style())

        # 联动
        input_box.valueChanged.connect(lambda v: slider.setValue(int(v * 100)))
        slider.valueChanged.connect(lambda v: input_box.setValue(v / 100.0))

        return input_box, slider

    def get_spinbox_style(self):
        return """
            QSpinBox, QDoubleSpinBox {
                background-color: #3C3C3C;
                border: 1px solid #555555;
                border-radius: 4px;
                color: #FFFFFF;
                padding: 5px;
                min-width: 80px;
            }
            QSpinBox::up-button, QDoubleSpinBox::up-button {
                background-color: #555555;
                border: 1px solid #777777;
            }
            QSpinBox::down-button, QDoubleSpinBox::down-button {
                background-color: #555555;
                border: 1px solid #777777;
            }
        """

    def get_slider_style(self):
        return """
            QSlider::groove:horizontal {
                border: 1px solid #555555;
                height: 8px;
                background: #3C3C3C;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #64FFDA;
                border: 1px solid #555555;
                width: 18px;
                margin: -5px 0;
                border-radius: 9px;
            }
            QSlider::sub-page:horizontal {
                background: #64FFDA;
                border: 1px solid #555555;
                height: 8px;
                border-radius: 4px;
            }
        """

    def get_parameters(self):
        """获取当前参数值"""
        return {
            'n_samples': self.samples_spinbox.value(),
            'dispersion': self.dispersion_input.value(),
            'curvature': self.curvature_input.value(),
            'flatness': self.flatness_input.value(),
            'correlation': self.correlation_input.value()
        }


class EmbeddingVisualizationWidget(QWidget):
    """3D可视化组件"""

    def __init__(self):
        super().__init__()
        self.setupUI()

    def setupUI(self):
        layout = QVBoxLayout()

        # 创建matplotlib图形
        self.figure = Figure(figsize=(10, 8), facecolor='#2B2B2B')
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setStyleSheet("background-color: #2B2B2B;")

        layout.addWidget(self.canvas)
        self.setLayout(layout)

        # 初始化空图
        self.init_empty_plot()

    def init_empty_plot(self):
        """初始化空的3D图"""
        self.figure.clear()
        ax = self.figure.add_subplot(111, projection='3d', facecolor='#2B2B2B')
        ax.set_xlabel('Component 1', color='white')
        ax.set_ylabel('Component 2', color='white')
        ax.set_zlabel('Component 3', color='white')
        ax.set_title('3D Embeddings Visualization', color='white', fontsize=14)

        # 设置坐标轴颜色
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.zaxis.label.set_color('white')
        ax.tick_params(colors='white')

        # 设置背景
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('gray')
        ax.yaxis.pane.set_edgecolor('gray')
        ax.zaxis.pane.set_edgecolor('gray')

        self.canvas.draw()

    def update_plot(self, embeddings_3d, labels):
        """更新3D图"""
        self.figure.clear()
        ax = self.figure.add_subplot(111, projection='3d', facecolor='#2B2B2B')

        # 为每个类别使用不同颜色
        colors = plt.cm.tab10(np.linspace(0, 1, len(np.unique(labels))))

        for i, class_id in enumerate(np.unique(labels)):
            mask = labels == class_id
            ax.scatter(embeddings_3d[mask, 0],
                       embeddings_3d[mask, 1],
                       embeddings_3d[mask, 2],
                       c=[colors[i]],
                       label=f'Class {class_id}',
                       alpha=0.7,
                       s=30)

        ax.set_xlabel('Component 1', color='white')
        ax.set_ylabel('Component 2', color='white')
        ax.set_zlabel('Component 3', color='white')
        ax.set_title('3D Embeddings Visualization', color='white', fontsize=14)
        ax.legend()

        # 设置坐标轴颜色
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.zaxis.label.set_color('white')
        ax.tick_params(colors='white')

        # 设置背景
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('gray')
        ax.yaxis.pane.set_edgecolor('gray')
        ax.zaxis.pane.set_edgecolor('gray')

        self.canvas.draw()


class MainWindow(QMainWindow):
    """主窗口"""

    def __init__(self):
        super().__init__()
        self.class_widgets = []
        self.setupUI()
        self.setDarkTheme()

    def setupUI(self):
        self.setWindowTitle("ALQuery3D - High-Dimensional Embeddings Generator")
        self.setGeometry(100, 100, 1400, 900)

        # 创建中央组件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # 主布局 - 水平分割
        main_layout = QHBoxLayout()

        # 左侧控制面板
        control_panel = self.create_control_panel()

        # 右侧可视化面板
        self.visualization_widget = EmbeddingVisualizationWidget()

        # 使用分割器
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(control_panel)
        splitter.addWidget(self.visualization_widget)
        splitter.setSizes([500, 900])  # 设置初始大小比例

        main_layout.addWidget(splitter)
        central_widget.setLayout(main_layout)

    def create_control_panel(self):
        """创建左侧控制面板"""
        panel = QWidget()
        panel.setMaximumWidth(550)
        layout = QVBoxLayout()

        # 标题
        title = QLabel("Embedding Generator")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        title.setStyleSheet("color: #64FFDA; margin: 10px; text-align: center;")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        # 类别数量选择
        class_layout = QHBoxLayout()
        class_layout.addWidget(QLabel("Number of Classes:"))
        self.class_combo = QComboBox()
        self.class_combo.addItems([str(i) for i in range(1, 11)])
        self.class_combo.setCurrentText("3")
        self.class_combo.setStyleSheet(self.get_combo_style())
        self.class_combo.currentTextChanged.connect(self.update_class_widgets)
        class_layout.addWidget(self.class_combo)
        class_layout.addStretch()
        layout.addLayout(class_layout)

        # 类别参数区域（可滚动）
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setStyleSheet("QScrollArea { border: none; background-color: #1E1E1E; }")
        self.scroll_widget = QWidget()
        self.scroll_layout = QVBoxLayout()
        self.scroll_widget.setLayout(self.scroll_layout)
        self.scroll_area.setWidget(self.scroll_widget)
        layout.addWidget(self.scroll_area)

        # 全局参数
        global_group = QGroupBox("Global Parameters")
        global_group.setStyleSheet("""
            QGroupBox {
                color: #FFFFFF;
                border: 2px solid #555555;
                border-radius: 8px;
                margin: 10px 0px;
                padding-top: 15px;
                font-weight: bold;
            }
            QGroupBox::title {
                color: #64FFDA;
                subcontrol-origin: margin;
                subcontrol-position: top center;
                padding: 0 5px;
            }
        """)
        global_layout = QHBoxLayout()

        global_layout.addWidget(QLabel("Inter-class Distance:"))
        self.inter_distance_input, self.inter_distance_slider = self.create_parameter_control(
            0.0, 1.0, 0.5
        )
        global_layout.addWidget(self.inter_distance_input)
        global_layout.addWidget(self.inter_distance_slider)

        global_group.setLayout(global_layout)
        layout.addWidget(global_group)

        # 生成按钮
        self.generate_button = QPushButton("Generate Embeddings")
        self.generate_button.setStyleSheet("""
            QPushButton {
                background-color: #64FFDA;
                color: #000000;
                border: none;
                border-radius: 8px;
                padding: 12px;
                font-size: 14px;
                font-weight: bold;
                margin: 10px;
            }
            QPushButton:hover {
                background-color: #4FC3F7;
            }
            QPushButton:pressed {
                background-color: #29B6F6;
            }
            QPushButton:disabled {
                background-color: #555555;
                color: #888888;
            }
        """)
        self.generate_button.clicked.connect(self.generate_embeddings)
        layout.addWidget(self.generate_button)

        panel.setLayout(layout)

        # 初始化类别组件
        self.update_class_widgets("3")

        return panel

    def create_parameter_control(self, min_val, max_val, default_val):
        """创建参数输入框和滑块的联动控制"""
        # 输入框
        input_box = QDoubleSpinBox()
        input_box.setRange(min_val, max_val)
        input_box.setSingleStep(0.01)
        input_box.setDecimals(2)
        input_box.setValue(default_val)
        input_box.setStyleSheet("""
            QDoubleSpinBox {
                background-color: #3C3C3C;
                border: 1px solid #555555;
                border-radius: 4px;
                color: #FFFFFF;
                padding: 5px;
                min-width: 80px;
            }
        """)

        # 滑块
        slider = QSlider(Qt.Horizontal)
        slider.setRange(int(min_val * 100), int(max_val * 100))
        slider.setValue(int(default_val * 100))
        slider.setStyleSheet("""
            QSlider::groove:horizontal {
                border: 1px solid #555555;
                height: 8px;
                background: #3C3C3C;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #64FFDA;
                border: 1px solid #555555;
                width: 18px;
                margin: -5px 0;
                border-radius: 9px;
            }
            QSlider::sub-page:horizontal {
                background: #64FFDA;
                border: 1px solid #555555;
                height: 8px;
                border-radius: 4px;
            }
        """)

        # 联动
        input_box.valueChanged.connect(lambda v: slider.setValue(int(v * 100)))
        slider.valueChanged.connect(lambda v: input_box.setValue(v / 100.0))

        return input_box, slider

    def get_combo_style(self):
        return """
            QComboBox {
                background-color: #3C3C3C;
                border: 1px solid #555555;
                border-radius: 4px;
                color: #FFFFFF;
                padding: 5px;
                min-width: 100px;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox::down-arrow {
                image: none;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 5px solid #FFFFFF;
            }
            QComboBox QAbstractItemView {
                background-color: #3C3C3C;
                border: 1px solid #555555;
                selection-background-color: #64FFDA;
                selection-color: #000000;
                color: #FFFFFF;
            }
        """

    def update_class_widgets(self, num_classes_str):
        """更新类别设置组件"""
        num_classes = int(num_classes_str)

        # 清除现有组件
        for widget in self.class_widgets:
            widget.setParent(None)
        self.class_widgets.clear()

        # 创建新的类别组件
        for i in range(num_classes):
            class_widget = ClassParameterWidget(i)
            self.class_widgets.append(class_widget)
            self.scroll_layout.addWidget(class_widget)

        # 添加弹性空间
        self.scroll_layout.addStretch()

    def generate_embeddings(self):
        """生成embeddings"""
        try:
            # 收集参数
            n_samples_per_class = []
            dispersion_list = []
            curvature_list = []
            flatness_list = []
            correlation_list = []

            for widget in self.class_widgets:
                params = widget.get_parameters()
                n_samples_per_class.append(params['n_samples'])
                dispersion_list.append(params['dispersion'])
                curvature_list.append(params['curvature'])
                flatness_list.append(params['flatness'])
                correlation_list.append(params['correlation'])

            generator_params = {
                'n_samples_per_class': n_samples_per_class,
                'dispersion': dispersion_list,
                'curvature': curvature_list,
                'flatness': flatness_list,
                'inter_class_distance': self.inter_distance_input.value(),
                'intra_class_correlation': correlation_list
            }

            # 禁用生成按钮
            self.generate_button.setEnabled(False)
            self.generate_button.setText("Generating...")

            # 启动后台线程
            self.generator_thread = EmbeddingGeneratorThread(generator_params)
            self.generator_thread.finished.connect(self.on_generation_finished)
            self.generator_thread.error.connect(self.on_generation_error)
            self.generator_thread.start()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to generate embeddings: {str(e)}")
            self.generate_button.setEnabled(True)
            self.generate_button.setText("Generate Embeddings")

    def on_generation_finished(self, embeddings_3d, labels):
        """生成完成回调"""
        self.visualization_widget.update_plot(embeddings_3d, labels)

        # 恢复生成按钮
        self.generate_button.setEnabled(True)
        self.generate_button.setText("Generate Embeddings")

        # 显示统计信息
        n_samples = len(labels)
        n_classes = len(np.unique(labels))
        QMessageBox.information(self, "Success",
                                f"Generated {n_samples} samples with {n_classes} classes successfully!")

    def on_generation_error(self, error_message):
        """生成错误回调"""
        QMessageBox.critical(self, "Generation Error", f"Error: {error_message}")

        # 恢复生成按钮
        self.generate_button.setEnabled(True)
        self.generate_button.setText("Generate Embeddings")

    def setDarkTheme(self):
        """设置深色主题"""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1E1E1E;
                color: #FFFFFF;
            }
            QWidget {
                background-color: #1E1E1E;
                color: #FFFFFF;
            }
            QLabel {
                color: #FFFFFF;
            }
            QScrollArea {
                background-color: #1E1E1E;
                border: none;
            }
            QScrollBar:vertical {
                background-color: #3C3C3C;
                width: 12px;
                border-radius: 6px;
            }
            QScrollBar::handle:vertical {
                background-color: #64FFDA;
                border-radius: 6px;
                min-height: 20px;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                border: none;
                background: none;
            }
        """)


def main():
    """主函数"""
    app = QApplication(sys.argv)

    # 设置应用程序属性
    app.setApplicationName("ALQuery3D")
    app.setApplicationVersion("1.0")

    # 创建主窗口
    window = MainWindow()
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
