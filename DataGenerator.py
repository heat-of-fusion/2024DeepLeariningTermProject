import os
import cv2
import glob
import numpy as np
from tqdm import tqdm
from scipy import ndimage
import matplotlib.pyplot as plt

train = False

J_PATH = f'./DLT_entity/j_{"train" if train else "valid"}/'
M_PATH = f'./DLT_entity/m_{"train" if train else "valid"}/'
G_PATH = f'./DLT_entity/generated_{"train" if train else "valid"}/'

REPEAT_PARAM = 10

j_rot_param = 0.15
m_rot_param = 0.15

J_list = os.listdir(J_PATH)
M_list = os.listdir(M_PATH)

std_weight = 1 / 3

jh_x_mu = 0.5
jh_x_sigma = 0.1 * std_weight

jh_y_mu = 0.4
jh_y_sigma = 0.1 * std_weight

jv_x_mu = 0.4
jv_x_sigma = 0.1 * std_weight

jv_y_mu = 0.5
jv_y_sigma = 0.1 * std_weight

mh_x_mu = 0.5
mh_x_sigma = 0.05 * std_weight
mh_y_mu = 0.6
mh_y_sigma = 0.05 * std_weight

mv_x_mu = 0.6
mv_x_sigma = 0.05 * std_weight
mv_y_mu = 0.5
mv_y_sigma = 0.05 * std_weight

WIDTH_CANVAS = 28
HEIGHT_CANVAS = 28

j_unique = ['j_' + f'{i}' for i in range(1, 15)]
m_unique = [f'mh_{i}' if i in range(5, 10) else f'mv_{i}' for i in range(1, 11)]

def add_noise(image, roughness = 0.1):

    noise_matrix = np.random.randn(*image.shape)
    noise_matrix = (noise_matrix / noise_matrix.max()) * roughness

    replace_idx = np.where(image > 0.05)
    image[replace_idx] -= noise_matrix[replace_idx]

    return image

class_num = int()
for j in j_unique:
    for m in m_unique:
        try:
            os.makedirs(G_PATH + f'{class_num}.{j}.{m}/')
        except:
            pass
        class_num += 1

for j_char in tqdm(J_list, desc='Data Generating...'):

    for j_ent in os.listdir(J_PATH + f'{j_char}/'):
        j_filename = J_PATH + f'{j_char}/' + f'{j_ent}'

        for m_char in M_list:

            for m_ent in os.listdir(M_PATH + f'{m_char}/'):
                m_filename = M_PATH + f'{m_char}/' + f'{m_ent}'

                for i in range(REPEAT_PARAM):

                    j_img = (255 - cv2.imread(j_filename)[:, :, 0]) / 255.0
                    j_space = np.where(j_img > 0)

                    m_img = (255 - cv2.imread(m_filename)[:, :, 0]) / 255.0
                    m_space = np.where(m_img > 0)
                    #
                    # plt.subplot(1, 2, 1)
                    # plt.imshow(j_img)
                    # plt.title('Original')
                    # plt.subplot(1, 2, 2)
                    # plt.imshow(m_img)
                    # plt.show()

                    # Rotate and Crop

                    j_rot_angle = np.random.randint(-90 * j_rot_param, 90 * j_rot_param)
                    m_rot_angle = np.random.randint(-90 * m_rot_param, 90 * m_rot_param)

                    j_img = ndimage.rotate(j_img, j_rot_angle)
                    m_img = ndimage.rotate(m_img, m_rot_angle)

                    # plt.subplot(1, 2, 1)
                    # plt.imshow(j_img)
                    # plt.title('Rotated')
                    # plt.subplot(1, 2, 2)
                    # plt.imshow(m_img)
                    # plt.show()

                    j_img = j_img[j_space[0].min() : j_space[0].max() + 1, j_space[1].min() : j_space[1].max() + 1]
                    m_img = m_img[m_space[0].min() : m_space[0].max() + 1, m_space[1].min() : m_space[1].max() + 1]

                    S_PATH = glob.glob(G_PATH + f'*{j_char}.{m_char}/')[0]

                    canvas = np.zeros([HEIGHT_CANVAS * 4, WIDTH_CANVAS * 4])
                    j_x = np.random.normal(jv_x_mu if m_char.split('_')[0] == 'mv' else jh_x_mu,
                                           jv_x_sigma if m_char.split('_')[0] == 'mv' else jh_x_sigma)
                    j_y = np.random.normal(jv_y_mu if m_char.split('_')[0] == 'mv' else jh_y_mu,
                                           jv_y_sigma if m_char.split('_')[0] == 'mv' else jh_y_sigma)

                    m_x = np.random.normal(mv_x_mu if m_char.split('_')[0] == 'mv' else mh_x_mu,
                                           mv_x_sigma if m_char.split('_')[0] == 'mv' else mh_x_sigma)
                    m_y = np.random.normal(mv_y_mu if m_char.split('_')[0] == 'mv' else mh_y_mu,
                                           mv_y_sigma if m_char.split('_')[0] == 'mv' else mh_y_sigma)

                    CS_INDEX_Y = int(HEIGHT_CANVAS * 1.5)
                    CS_INDEX_X = int(WIDTH_CANVAS * 1.5)

                    J_Y = int(HEIGHT_CANVAS * j_y)
                    J_X = int(WIDTH_CANVAS * j_x)

                    J_H, J_W = j_img.shape[0], j_img.shape[1]
                    M_H, M_W = m_img.shape[0], m_img.shape[1]

                    M_Y = int(HEIGHT_CANVAS * m_y)
                    M_X = int(WIDTH_CANVAS * m_x)

                    canvas[CS_INDEX_Y + J_Y - int(J_H / 2): CS_INDEX_Y + J_Y - int(J_H / 2) + J_H, CS_INDEX_X + J_X - int(j_img.shape[1] / 2): CS_INDEX_X + J_X - int(j_img.shape[1] / 2) + J_W][j_img > 0] = j_img[j_img > 0]
                    canvas[CS_INDEX_Y + M_Y - int(M_H / 2): CS_INDEX_Y + M_Y - int(M_H / 2) + M_H,
                    CS_INDEX_X + M_X - int(M_W / 2): CS_INDEX_X + M_X - int(M_W / 2) + M_W][m_img > 0] = m_img[
                        m_img > 0]

                    canvas_crop = canvas[CS_INDEX_Y: CS_INDEX_Y + HEIGHT_CANVAS, CS_INDEX_X: CS_INDEX_X + WIDTH_CANVAS]

                    cv2.imwrite(S_PATH + f'{len(os.listdir(S_PATH))}.png', (canvas_crop * 255).astype(np.uint8))