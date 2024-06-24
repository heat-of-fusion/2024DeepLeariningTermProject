import os
import cv2
import numpy as np
from scipy import ndimage
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from PIL import Image, ImageFont, ImageDraw

FONT_PATH = './fonts/'
G_PATH = './font_texts/'
font_list = os.listdir(FONT_PATH)

text_layout_mu = 0
text_layout_sigma = 0.3

text_size_mu = 0.6
text_size_sigma = 0.05
epsilon = 1e-4

rot_param = 0.3

ITER_NUM = 40

text_list = [
    '가', '갸', '거', '겨', '고', '교', '구', '규', '그', '기',
    '나', '냐', '너', '녀', '노', '뇨', '누', '뉴', '느', '니',
    '다', '댜', '더', '뎌', '도', '됴', '두', '듀', '드', '디',
    '라', '랴', '러', '려', '로', '료', '루', '류', '르', '리',
    '마', '먀', '머', '며', '모', '묘', '무', '뮤', '므', '미',
    '바', '뱌', '버', '벼', '보', '뵤', '부', '뷰', '브', '비',
    '사', '샤', '서', '셔', '소', '쇼', '수', '슈', '스', '시',
    '아', '야', '어', '여', '오', '요', '우', '유', '으', '이',
    '자', '쟈', '저', '져', '조', '죠', '주', '쥬', '즈', '지',
    '차', '챠', '처', '쳐', '초', '쵸', '추', '츄', '츠', '치',
    '카', '캬', '커', '켜', '코', '쿄', '쿠', '큐', '크', '키',
    '타', '탸', '터', '텨', '토', '툐', '투', '튜', '트', '티',
    '파', '퍄', '퍼', '펴', '포', '표', '푸', '퓨', '프', '피',
    '하', '햐', '허', '혀', '호', '효', '후', '휴', '흐', '히'
]

def generate_img(shape, text, font, font_size = 0.5):
    dummy_img = (np.ones(shape) * 255).astype(np.uint8)
    dummy_img = Image.fromarray(dummy_img)

    text_size = int(shape[0] * font_size)

    text_layout_x = int(shape[1] // 2 * (1 + np.random.normal(text_layout_mu, text_layout_sigma)))
    text_layout_y = int(shape[0] // 2 * (1 + np.random.normal(text_layout_mu, text_layout_sigma)))

    x_bias = -(text_size // 2)
    y_bias = -(text_size // 2)

    font = ImageFont.truetype(font, size = text_size)

    draw = ImageDraw.Draw(dummy_img, 'RGBA')
    draw.text((text_layout_x + x_bias, text_layout_y + y_bias), text, font = font, fill = 0)

    rot_angle = np.random.randint(-90 * rot_param, 90 * rot_param)

    dummy_img = ndimage.rotate(255 - np.array(dummy_img), rot_angle)

    dummy_img = dummy_img[
        (dummy_img.shape[0] - shape[0]) // 2 : (dummy_img.shape[0] - shape[0]) // 2 + shape[1],
        (dummy_img.shape[1] - shape[1]) // 2: (dummy_img.shape[1] - shape[1]) // 2 + shape[1], :
    ]

    return dummy_img

# font_image = generate_img([100, 100, 3], text = '가', font = FONT_PATH + font_list[0], font_size = 0.5)

for class_num in range(len(text_list)):
        try:
            os.makedirs(G_PATH + f'{class_num}/')
        except:
            pass

for class_num, target_text in enumerate(tqdm(text_list, desc = 'Generating...')):
    file_idx = int()
    for font in font_list:
        for i in range(ITER_NUM):
            font_size = np.clip(np.random.normal(text_size_mu, text_size_sigma), 0.0 + epsilon, 1.0 - epsilon)
            font_image = generate_img([28, 28, 3], text = target_text, font = FONT_PATH + font, font_size = font_size)

            cv2.imwrite(G_PATH + f'{class_num}/{file_idx}.png', font_image)
            file_idx += 1