from aiogram import Bot, Dispatcher
from aiogram.types import Message
from aiogram.filters import Command
from aiogram import F
from predictions import predict, model, classes
import asyncio
import os


bot = Bot(token="TOKEN")
dp = Dispatcher()

@dp.message(Command("start"))
async def start(message: Message):
    await message.answer(f"Привет {message.from_user.first_name}!! Отправь фото и я скажу какой фрукт, "
                         f"или овощ на ней изображён")


@dp.message(F.photo)
async def photo(message: Message):
    file_path = None
    try:
        photo = message.photo[-1]
        file = await bot.get_file(photo.file_id)
        file_path = f"{photo.file_id}.jpg"
        await bot.download_file(file.file_path, file_path)
        num_class, conf= predict(file_path, model, classes)
        s = f"На фото изображён: {num_class}\n С уверенностью: {conf:.2f}%"
        await message.answer(s)

    except Exception as e:
        print(e)
        await message.answer(f"Ошибка при обработке изображения")
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)


@dp.message(F.text, ~F.text.startswith("/"))
async def text(message: Message):
    await message.answer(f"Ошибка!!!! Отправьте боту фото")


async def main():
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())



