import torch
import asyncio

from TOKEN import TOKEN
from aiogram import executor, Bot, Dispatcher, types
from get_model import Model
from TextPreprocess.BertTextPreproccess import BertTextPreprocess

Bot = Bot(token=TOKEN)
dp = Dispatcher(Bot)


model, tokenizer = Model().get_bert()
preprocessor = BertTextPreprocess(tokenizer)
@dp.message_handler(content_types=['text'])
async def text_cls(message: types.Message):
    text = message.text
    pr_output = preprocessor.preprocess(text)
    input_ids = pr_output['input_ids'].to('cpu')
    attention_mask = pr_output['attention_mask'].to('cpu')
    output = model(input_ids=input_ids, attention_mask=attention_mask)
    print(output)
    # pred = torch.argmax(output.logits, dim=1).cpu().numpy()

    # await Bot.send_message(message.from_user.id, str(pred))


if __name__ == '__main__':
    executor.start_polling(dp)