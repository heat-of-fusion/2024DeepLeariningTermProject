import torch
import numpy as np

def get_accuracy(pred, labels):
    pred = pred.cpu().detach().numpy().argmax(axis = 1)
    labels = labels.cpu().detach().numpy().argmax(axis = 1)

    accuracy = (pred == labels).mean()

    return accuracy

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