import os
from datetime import datetime

from dotenv import load_dotenv
from linebot import LineBotApi
from linebot.exceptions import LineBotApiError
from linebot.models import TextSendMessage

ENV_PATH = "./resources/config/.env"
load_dotenv(ENV_PATH)

def notify(message: str):
    user_id = os.getenv("LINE_INTERNAL_USER_ID")
    token = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
    if not user_id or not token:
        print(f"{datetime.now()} LINE Messaging LINE_INTERNAL_USER_ID or LINE_CHANNEL_ACCESS_TOKEN is not set: {message}")
        return

    try:
        LineBotApi(token).push_message(user_id, TextSendMessage(text=f"{datetime.now()} {message}"))
    except LineBotApiError as e:
        print(f"{datetime.now()} LINE Messaging Error occured: {e}")