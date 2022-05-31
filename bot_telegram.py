from email import message
from re import L
import telebot
import torch
from typing import Optional
import os
from torchvision import transforms
from transforms import IBotWeakTransformCollection
import cv2
import albumentations as A
import onnxruntime

import json 
import torch
import pytorch_lightning as pl
from captcha_model import CaptchaLight
import numpy as np

TORCH_WEIGHTS = {
    'MobileNetV3-Small': 'benchmarks/torch_weights/mbnsmall.ckpt',
    'MobileNetV3-Large': 'benchmarks/torch_weights/mbnlarge.ckpt',
    'ResNet18': 'benchmarks/torch_weights/resnet18.ckpt'
}

JIT_WEIGHTS = {
    'MobileNetV3-Small': 'benchmarks/jit_weights/mbnsmall.pt',
    'MobileNetV3-Large': 'benchmarks/jit_weights/mbnlarge.pt',
    'ResNet18': 'benchmarks/jit_weights/resnet18.pt'
}
CLASS_NAMES = [
    'airplane',
    'bicycle',
    'boat',
    'bus',
    'car',
    'floatplane',
    'train',
    'truck',
    'umbrella',
    'van'
]
IMG_SIZE = 128
MODELS = {}


key = '5137358430:AAHNZPAT137UneoPm1eC1qN2CDzNMDzA5FQ'
bot = telebot.TeleBot(key)


class BotDataLoader:
    def __init__(self, img_size: Optional[int] = 128):
        self.__random_crop = A.Compose([
            A.augmentations.crops.transforms.RandomCrop(
                height=img_size,
                width=img_size,
                always_apply=True,
                p=1.0
            )
        ])
        self.img_size = img_size
        self.transform = IBotWeakTransformCollection.validation_transform(size_=img_size)

    def resize(self, img_, short_side):
        height, width = img_.shape[0], img_.shape[1]
        rotated = False
        if height > width:
            img_ = cv2.transpose(img_)
            rotated = True
        _short_side = img_.shape[0]
        _long_side = img_.shape[1]
        ratio = _long_side / _short_side
        img_ = cv2.resize(img_, (int(short_side * ratio), short_side))
        if rotated:
            img_ = cv2.transpose(img_)
        return img_

    def __call__(self, path: str) -> np.ndarray:
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.resize(image, self.img_size)
        image = self.__random_crop(image=image)["image"]
        image = self.transform(image=image)["image"]
        return image

bot_dataloader = BotDataLoader(IMG_SIZE)

def softmax(x):
    t = 1
    f_x = (np.exp(x * t)) / np.sum(np.exp(x * t))
    return f_x

class InferenceTorch:
    def __init__(self, weights_path):
        self.model = CaptchaLight.load_from_checkpoint(weights_path)
        sample_inputs = torch.rand((1, 3, 128, 128))
        self.model(sample_inputs)
        self.numpy_to_tensor_transfom = transforms.ToTensor()

    def __call__(self, img: np.ndarray):
        image_tensor = self.numpy_to_tensor_transfom(img.astype(np.float32))
        result = softmax(self.model(image_tensor.unsqueeze(dim=0)).detach().cpu().numpy()[0])
        class_label = np.argmax(result)
        proba = np.max(result)
        print(list(result))
        print(CLASS_NAMES)
        return class_label, proba

class InferenceJIT:
    def __init__(self, weights_path) -> None:
        self.model = torch.jit.load(weights_path)
        self.numpy_to_tensor_transfom = transforms.ToTensor()

    def __call__(self, img: np.ndarray):
        image_tensor = self.numpy_to_tensor_transfom(img.astype(np.float32))
        result = softmax(self.model(image_tensor.unsqueeze(dim=0)).detach().cpu().numpy()[0])
        class_label = np.argmax(result)
        proba = np.max(result)
        return class_label, proba

class InferenceONNX:
    def __init__(self, weights_path):
        self.numpy_to_tensor_transfom = transforms.ToTensor()
        model = CaptchaLight.load_from_checkpoint(weights_path)
        filepath = "tmp/tmp.onnx"
        sample_inputs = torch.rand((1, 3, 128, 128))
        model.to_onnx(filepath, sample_inputs, export_params=True)
        self.ort_session = onnxruntime.InferenceSession(filepath)
        self.input_name = self.ort_session.get_inputs()[0].name
        # os.system('rm -f tmp/tmp.onnx')
    
    def __call__(self, img: np.ndarray):
        image_tensor = self.numpy_to_tensor_transfom(img.astype(np.float32)).unsqueeze(dim=0)
        ort_inputs = {self.input_name: image_tensor.numpy()}
        ort_outs = softmax(self.ort_session.run(None, ort_inputs))
        class_label = np.argmax(ort_outs)
        proba = np.max(ort_outs)
        return class_label, proba


class Params:
    class UserParams:
        def __init__(self):
            self.__model = None
            self.__backend = None
            self.__callable = None

        def __setup(self):
            if self.backend == 'PyTorch':
                self.__callable = InferenceTorch(TORCH_WEIGHTS[self.model])
            elif self.backend == 'ONNX':
                self.__callable = InferenceONNX(TORCH_WEIGHTS[self.model])
            elif self.backend == 'JIT':
                self.__callable = InferenceJIT(JIT_WEIGHTS[self.model])
            else:
                raise ValueError()
    
        @property
        def model(self) -> str:
            print('called model ')
            print(self.__model)
            return self.__model

        @model.setter
        def model(self, new_model: str):
            print('updating model')
            self.__model = new_model
            if self.__backend is not None:
                self.__setup()
            print(self.__model)

        @property
        def backend(self) -> str:
            return self.__backend

        @backend.setter
        def backend(self, new_backend: str):
            print('updating backend')
            self.__backend = new_backend
            if self.__model is not None:
                self.__setup()
        
        @property 
        def callable(self):
            if self.__callable is None:
                raise ValueError()
            return self.__callable

        

    def __init__(self):
        self.params = dict()

    def add_user(self, chat_id):
        self.params[chat_id] = Params.UserParams()
    
    def update_user_model(self, chat_id, new_model):
        self.params[chat_id].model = new_model

    def update_user_backend(self, chat_id, new_backend):
        self.params[chat_id].backend = new_backend


    def get_user_model(self, chat_id):
        return self.params[chat_id].model

    def get_user_backend(self, chat_id):
        return self.params[chat_id].backend

    def get_user_callable(self, chat_id):
        return self.params[chat_id].callable

    def is_user_registered(self, chat_id):
        return chat_id in self.params.keys()

    def user_has_valid_params(self, chat_id):
        if self.is_user_registered(chat_id):
            if self.get_user_model(chat_id) is not None:
                if self.get_user_backend(chat_id) is not None:
                    return True
        return False

PARAMS = Params()

def process_image(filepath: str, chat_id) -> str:
    callable = PARAMS.get_user_callable(chat_id)
    img = bot_dataloader(filepath)
    class_label, proba = callable(img)
    return class_label, proba


@bot.message_handler(commands=['start'])
def send_welcome(message):
    markup = telebot.types.ReplyKeyboardMarkup(resize_keyboard=True)
    but2 = telebot.types.KeyboardButton("Model")
    but3 = telebot.types.KeyboardButton("Backend")
    but1 = telebot.types.KeyboardButton("Current settings")
    markup.add(but1, but2, but3)
    PARAMS.add_user(message.chat.id)
    bot.reply_to(message, 'Enter params:', reply_markup=markup)


@bot.message_handler(content_types=['photo'])
def handle_image(message):
    if not PARAMS.user_has_valid_params(chat_id=message.chat.id):
        bot.reply_to(message, 'Image received, but your current settings is not valid')
        return

    # print(message.photo)
    # print(dir(message.photo))
    
    file_info = bot.get_file(message.photo[-1].file_id)
    downloaded_file = bot.download_file(file_info.file_path)
    file_path = file_info.file_path

    with open(file_path, 'w+b') as new_file:
        new_file.write(downloaded_file)


    class_label, proba = process_image(
        file_path, 
        message.chat.id
    )
    class_label = CLASS_NAMES[class_label]
    output = {
        'Class': class_label,
        'Probability': round(proba * 100, 2)        
    }
    # bot.send_image(chat_id=message.chat.id, video=open(out, 'rb'), supports_streaming=True)
    bot.reply_to(message, 'Class: {}\nProbability: {}%'.format(output['Class'], output['Probability']))




@bot.message_handler(func=lambda message: True)
def menu(message):
    if message.chat.type == 'private':
        if message.text == "Model":
            inMurkup = telebot.types.InlineKeyboardMarkup(row_width=1)
            but1 = telebot.types.InlineKeyboardButton('ResNet18',callback_data='set_model_ResNet18')
            but2 = telebot.types.InlineKeyboardButton('MobileNetV3-Large',callback_data='set_model_MobileNetV3-Large')
            but3 = telebot.types.InlineKeyboardButton('MobileNetV3-Small',callback_data='set_model_MobileNetV3-Small')
            inMurkup.add(but1, but2, but3)
            bot.send_message(message.chat.id, "Choose one", reply_markup=inMurkup)

        elif message.text == "Backend":
            inMurkup = telebot.types.InlineKeyboardMarkup(row_width=1)
            but1 = telebot.types.InlineKeyboardButton('PyTorch', callback_data='set_backend_PyTorch')
            but2 = telebot.types.InlineKeyboardButton('JIT', callback_data='set_backend_JIT')
            but3 = telebot.types.InlineKeyboardButton('ONNX', callback_data='set_backend_ONNX')
            inMurkup.add(but1, but2, but3)
            bot.send_message(message.chat.id, "Choose one", reply_markup=inMurkup)

        elif message.text == 'Current settings':
            if not PARAMS.is_user_registered(message.chat.id):
                PARAMS.add_user(message.chat.id)
            curr_model = PARAMS.get_user_model(message.chat.id)
            curr_backend = PARAMS.get_user_backend(message.chat.id)
            bot.send_message(message.chat.id, "Model: {}\nBackend: {}".format(curr_model, curr_backend))
        else:
            bot.send_message(message.chat.id, "Unknown command")



@bot.callback_query_handler(func=lambda call: True)
def callback_inline(call):
    try:
        if call.message:
            out_text = ''
            if str(call.data).startswith('set_model_'):
                new_model = str(call.data).split('_')[-1]
                PARAMS.update_user_model(call.message.chat.id, new_model=new_model)
                out_text = 'Model set to {}'.format(new_model)
            
            if str(call.data).startswith('set_backend_'):
                new_backend = str(call.data).split('_')[-1]
                print(new_backend)
                PARAMS.update_user_backend(call.message.chat.id, new_backend=new_backend)
                out_text = 'Backend set to {}'.format(new_backend)
            bot.edit_message_text(
                chat_id=call.message.chat.id, 
                message_id=call.message.message_id, 
                text=out_text if out_text else None,
				reply_markup=None
            )
    except Exception as e:
        print(repr(e))


bot.polling(none_stop=True, interval=0)
