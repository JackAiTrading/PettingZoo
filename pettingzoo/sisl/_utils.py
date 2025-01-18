class Agent:
    """基础智能体类"""
    def __new__(cls, *args, **kwargs):
        """创建新的智能体实例"""
        agent = super().__new__(cls)
        return agent

    @property
    def observation_space(self):
        """返回观察空间（需要子类实现）"""
        raise NotImplementedError()

    @property
    def action_space(self):
        """返回动作空间（需要子类实现）"""
        raise NotImplementedError()

    def __str__(self):
        """返回智能体的字符串表示"""
        return f"<{type(self).__name__} instance>"
