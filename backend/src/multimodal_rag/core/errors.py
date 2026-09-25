"""错误只携带稳定代码与安全描述，不把路径、原文或凭证返回给客户端。"""


class AppError(Exception):
    def __init__(self, code: str, message: str, status: int = 400):
        self.code, self.message, self.status = code, message, status
        super().__init__(message)

