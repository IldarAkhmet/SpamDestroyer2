from TOKEN import TOKEN
from aiogram import executor, Bot, Dispatcher, types
from get_model import Model
from TextPreprocess.BertTextPreproccess import BertTextPreprocess

import asyncio

Bot = Bot(token=TOKEN)
dp = Dispatcher(Bot)


model, tokenizer = Model().get_bert()

@dp.message_handler(content_types=['text'])
async def text_cls(message: types.Message):
    text = message.text

    await Bot.send_message(message.from_user.id, text)


if __name__ == '__main__':
    executor.start_polling(dp)