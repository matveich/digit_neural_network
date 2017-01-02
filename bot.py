import telebot
import network
import requests

token = open('./token.txt', 'r').readline()
bot = telebot.TeleBot(token)

net = network.load('./net.txt')


@bot.message_handler(commands=['start', 'help'])
def send_info(message):
    bot.send_message(message.chat.id, "Загрузите картинку, чтобы я её распознал\n/help - помощь")


@bot.message_handler(content_types=['photo'])
def pic_hadler(message):
    if not message.photo:
        return
    file_id = None
    min_size = -1
    for photo in message.photo:
        if min_size == -1 or photo.width * photo.height < min_size:
            min_size = photo.width * photo.height
            file_id = photo.file_id
    pic_url = 'https://api.telegram.org/file/bot{0}/{1}'.format(token, bot.get_file(file_id).file_path)
    pic_file = requests.get(pic_url).content
    bot.send_message(message.chat.id, "Это цифра " + str(net.recognize(pic_file)))


bot.polling()
