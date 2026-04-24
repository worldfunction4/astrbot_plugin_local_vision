"""
AstrBot 本地图片识别插件
监听图片消息，调用本地识图服务进行识别
"""
from __future__ import annotations

import aiohttp
import base64
from typing import AsyncGenerator, Optional

import astrbot.api.star as star
from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.core.config.astrbot_config import AstrBotConfig


class ModelNotFoundError(Exception):
    """模型未找到异常"""
    def __init__(self, model: str):
        self.model = model
        super().__init__(f"Model {model} not found")


class LocalVisionPlugin(star.Star):
    """本地图片识别插件"""

    def __init__(self, context: star.Context, config: AstrBotConfig) -> None:
        super().__init__(context)

        self.vision_api_url = config.get("local_vision_api_url", "http://100.64.0.1:8765/analyze")
        self.enable_auto_reply = config.get("enable_auto_reply", True)
        self.reply_prefix = config.get("reply_prefix", "")
        self.timeout_seconds = config.get("timeout_seconds", 30)
        self.max_image_size_mb = config.get("max_image_size_mb", 10)
        self.custom_prompt = config.get("custom_prompt", "")
        self.ollama_model = config.get("ollama_model", "llava:7b")
        self.group_require_at = config.get("group_require_at", True)

        logger.info(f"[LocalVision] 插件已加载，识图服务: {self.vision_api_url}, 模型: {self.ollama_model}")

    @filter.event_message_type(filter.EventMessageType.PRIVATE_MESSAGE, priority=10)
    async def on_friend_message(self, event: AstrMessageEvent) -> AsyncGenerator:
        """处理私聊消息"""
        async for result in self._handle_image(event):
            yield result

    @filter.event_message_type(filter.EventMessageType.GROUP_MESSAGE, priority=10)
    async def on_group_message(self, event: AstrMessageEvent) -> AsyncGenerator:
        """处理群聊消息"""
        async for result in self._handle_image(event):
            yield result

    async def _handle_image(self, event: AstrMessageEvent) -> AsyncGenerator:
        """图片处理主逻辑"""
        if not self.enable_auto_reply:
            return

        # 群聊触发控制：如果开启了 group_require_at，则群聊中必须 @ 机器人
        is_group = event.get_group_id() is not None
        if is_group and self.group_require_at:
            if not self._is_at_me(event):
                return

        # 提取图片路径
        image_path = self._extract_image_path(event)
        if not image_path:
            return

        logger.info(f"[LocalVision] 检测到图片: {image_path}")

        # 获取被引用消息的 ID（如果有）
        reply_to_id = getattr(event.message_obj, 'reply_to_id', None) if hasattr(event, 'message_obj') else None

        try:
            # 读取图片数据
            image_data = await self._read_image(image_path)
            if not image_data:
                logger.warning(f"[LocalVision] 图片读取失败: {image_path}")
                return

            # 检查图片大小
            size_mb = len(image_data) / (1024 * 1024)
            if size_mb > self.max_image_size_mb:
                yield event.plain_result(f"图片太大了({size_mb:.1f}MB)，我处理不了...", reply_to=reply_to_id)
                return

            # 调用识图服务
            logger.info("[LocalVision] 正在调用本地识图服务...")
            description = await self._analyze_image(image_data)

            if description:
                reply = f"{self.reply_prefix}{description}" if self.reply_prefix else description
                logger.info(f"[LocalVision] 识图成功，描述长度: {len(description)}")
                yield event.plain_result(reply, reply_to=reply_to_id)
            else:
                yield event.plain_result("嗯...我看不太清楚这张图片", reply_to=reply_to_id)

        except ModelNotFoundError as e:
            yield event.plain_result(
                f"❌ 识图失败：本地未找到模型 {e.model}\n"
                f"请在运行 vision_server 的机器上执行：\n"
                f"ollama pull {e.model}",
                reply_to=reply_to_id
            )
        except aiohttp.ClientConnectorError:
            yield event.plain_result("❌ 无法连接到识图服务，请确认 vision_server.py 已在本地启动", reply_to=reply_to_id)
        except Exception as e:
            logger.error(f"[LocalVision] 处理图片时出错: {e}", exc_info=True)
            yield event.plain_result("识别图片时出了点问题...", reply_to=reply_to_id)

    def _is_at_me(self, event: AstrMessageEvent) -> bool:
        """检查消息是否 @ 了机器人"""
        try:
            if not hasattr(event, 'message_obj') or not event.message_obj:
                return False

            message = event.message_obj.message
            if not message:
                return False

            # 遍历消息链，查找 At 类型的消息段
            for segment in message:
                seg_type = str(getattr(segment, "type", ""))
                if "At" in seg_type or "at" in seg_type:
                    # 检查是否 @ 的是机器人自己
                    qq = getattr(segment, "qq", None)
                    if qq and str(qq) == str(event.get_self_id()):
                        return True

            return False

        except Exception as e:
            logger.error(f"[LocalVision] 检查 @ 状态失败: {e}", exc_info=True)
            return False

    def _extract_image_path(self, event: AstrMessageEvent) -> Optional[str]:
        """从消息事件中提取图片路径或 URL"""
        try:
            if not hasattr(event, "message_obj") or not event.message_obj:
                return None

            message = event.message_obj.message
            if not message:
                return None

            for segment in message:
                seg_type = str(getattr(segment, "type", ""))

                if "Image" not in seg_type and "image" not in seg_type:
                    continue

                url = getattr(segment, "url", None)
                path = getattr(segment, "path", None)
                file = getattr(segment, "file", None)

                # url 字段：HTTP 链接（QQ 等平台优先用这个）
                if url and url.strip():
                    return url.strip()

                # path 字段：容器内绝对路径（微信等平台）
                if path and path.strip():
                    return path.strip()

                # file 字段：仅当是绝对路径时才使用
                if file and file.strip() and file.startswith("/"):
                    return file.strip()

            return None

        except Exception as e:
            logger.error(f"[LocalVision] 提取图片路径失败: {e}", exc_info=True)
            return None

    async def _read_image(self, path: str) -> Optional[bytes]:
        """读取图片数据（本地文件或 HTTP URL）"""
        try:
            # 本地文件
            if path.startswith("/"):
                logger.info(f"[LocalVision] 读取本地文件: {path}")
                with open(path, "rb") as f:
                    return f.read()

            # base64
            if path.startswith("base64://"):
                return base64.b64decode(path[9:])

            if path.startswith("data:image") and "base64," in path:
                return base64.b64decode(path.split("base64,", 1)[1])

            # HTTP URL
            if path.startswith(("http://", "https://")):
                async with aiohttp.ClientSession() as session:
                    async with session.get(path, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                        if resp.status == 200:
                            return await resp.read()
                        logger.warning(f"[LocalVision] 下载图片失败，状态码: {resp.status}")
                        return None

            logger.warning(f"[LocalVision] 不支持的图片格式: {path[:80]}")
            return None

        except FileNotFoundError:
            logger.error(f"[LocalVision] 文件不存在: {path}")
            return None
        except Exception as e:
            logger.error(f"[LocalVision] 读取图片失败: {e}", exc_info=True)
            return None

    async def _analyze_image(self, image_data: bytes) -> Optional[str]:
        """调用本地识图服务"""
        try:
            form = aiohttp.FormData()
            form.add_field("file", image_data, filename="image.jpg", content_type="image/jpeg")
            form.add_field("model", self.ollama_model)

            if self.custom_prompt:
                form.add_field("prompt", self.custom_prompt)

            timeout = aiohttp.ClientTimeout(total=self.timeout_seconds)
            async with aiohttp.ClientSession() as session:
                async with session.post(self.vision_api_url, data=form, timeout=timeout) as resp:
                    if resp.status == 200:
                        result = await resp.json()
                        return result.get("description")

                    if resp.status == 422:
                        body = await resp.json()
                        if body.get("error") == "model_not_found":
                            raise ModelNotFoundError(body.get("model", self.ollama_model))

                    error_text = await resp.text()
                    logger.error(f"[LocalVision] 识图服务错误 {resp.status}: {error_text}")
                    return None

        except aiohttp.ClientConnectorError as e:
            logger.error(f"[LocalVision] 无法连接到识图服务: {e}")
            raise
        except ModelNotFoundError:
            raise
        except Exception as e:
            logger.error(f"[LocalVision] 调用识图服务失败: {e}", exc_info=True)
            return None
