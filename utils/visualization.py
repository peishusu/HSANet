from tensorboardX import SummaryWriter
'''
    这段代码使用了 tensorboardX 库（是 TensorBoard 的一个 Python 包封装），用于将训练过程中的损失曲线、图像、模型结构等信息写入日志，便于用 TensorBoard 可视化分析
'''
class Visualization:
    def __init__(self):
        self.writer = ''# 初始化时 writer 是空字符串（实际应设为 None）

    def create_summary(self, model_type='U_Net'):
        """新建writer 设置路径"""
        # self.writer = SummaryWriter(model_type, comment=model_type)
        self.writer = SummaryWriter(comment='-' +model_type) # 会创建一个名为 runs/May06-17-30-01_hostname_-U_Net 的目录，用于记录所有后续写入的内容。

    def add_scalar(self, epoch, value, params='loss'):
        """添加训练记录"""
        self.writer.add_scalar(params, value, global_step=epoch)

    def add_image(self, tag, img_tensor):
        """添加tensor影像"""
        self.writer.add_image(tag, img_tensor)

    def add_graph(self, model):
        """添加模型图"""
        self.writer.add_graph(model)

    def close_summary(self):
        """关闭writer"""
        self.writer.close()