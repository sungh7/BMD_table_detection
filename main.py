import cv2
import collections
import pytesseract
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import *
import argparse

warnings.filterwarnings('ignore')

width_th = 0.989
height_th = 0.97


def sort_contours(cnts, method='left-to-right'):
    reverse = False
    i = 0
    if method == 'right-to-left' or method == 'bottom-to-top':
        reverse = True
    if method == 'top-to-bottom' or method == 'bottom-to-top':
        i = 1
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(
        *sorted(zip(cnts, boundingBoxes), key=lambda b: b[1][i], reverse=reverse))
    return (cnts, boundingBoxes)


def findLineIndexList(gray, direction='row'):
    h, w = gray.shape
    line_index_list = []

    if direction == 'row':
        for i in range(h):
            count_255 = collections.Counter(gray[i, :])[gray.max()]
            if count_255 > w * width_th:
                line_index_list.append(i)

    elif direction == 'col':
        for i in range(w):
            count_255 = collections.Counter(gray[:, i])[gray.max()]
            if count_255 > h * height_th:
                line_index_list.append(i)

    return line_index_list


def findMedianIndexs(index_list):
    diff = 0
    start = 0
    indexs = index_list
    ixs = []
    for end in range(1, len(indexs)):
        diff = abs(indexs[start] - indexs[end])
        if diff > 20:
            ix_mean = round(np.mean(indexs[start:end]))
            ixs.append(ix_mean)
            start = end
        elif end == len(indexs)-1:
            ix_mean = round(np.mean(indexs[start:end+1]))
            ixs.append(ix_mean)
    return ixs


def drawLineOfImageTest(image_data):
    origin = image_data
    gray = cv2.cvtColor(origin, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

    count_255_col_line_ixs = findMedianIndexs(findLineIndexList(th, 'col'))
    count_255_row_line_ixs = findMedianIndexs(findLineIndexList(th, 'row'))

    th[:, :] = 255
    th[count_255_row_line_ixs, :] = 0
    th[:, count_255_col_line_ixs] = 0

    return origin, th


def extractTableFromGNUHBMD(IMAGE_PATH):
    table_image_list = []
    im = cv2.imread(IMAGE_PATH)
    im1 = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    _, thresh_value = cv2.threshold(im1, 180, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(
        thresh_value, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    rectangles = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        if w > 150 and h > 60 and h < 800:
            if x > 400 and y < 1150:
                pass
            else:
                rectangles.append((x, y, w, h))
    c = 0
    for rec in rectangles:
        c += 1
        x, y, w, h = rec
        im0 = im[y:y+h, x:x+w, :]
        table_image_list.append(im0)
    return table_image_list


def extractTableData(table_image, num_of_columns, sorted_box):
    dic = {}
    for i in range(num_of_columns):
        dic[f'Column {i}'] = []
    for count, box in (enumerate(sorted_box.values)):
        x, y, w, h = box
        column_ix = count % num_of_columns
        txt = pytesseract.image_to_string(
            table_image[y:y+h, x:x+w, :], config='--psm 7').strip()
        dic[f'Column {column_ix}'].append(txt)
    return pd.DataFrame(dic)


def sortBoxes(boundingBoxes, image_shape, h, w):
    df = pd.DataFrame(np.array(boundingBoxes), columns=['x', 'y', 'w', 'h'])
    m = (df.h > h) & (df.w > w)
    im_w, im_h, = image_shape[1], image_shape[0]
    sorted_box = df[m].sort_values(by=['x', 'y'])
    if (sorted_box.iloc[0].w == im_w) and (sorted_box.iloc[0].h == im_h):
        sorted_box = sorted_box.iloc[1:]

    sorted_box_x_index = sorted_box.x.value_counts().index

    for i in range(sorted_box_x_index.size-1):
        li = np.array(sorted_box_x_index[i:i+2])
        diff = np.diff(li)
        if abs(diff) < 10:
            sorted_box.loc[sorted_box[sorted_box.isin(
                li)].x.dropna().index, 'x'] = round(np.mean(li))

    sorted_box_y_index = sorted_box.y.value_counts().index

    for i in range(sorted_box_y_index.size-1):
        li = np.array(sorted_box_y_index[i:i+2])
        diff = np.diff(li)
        if abs(diff) < 10:
            sorted_box.loc[sorted_box[sorted_box.isin(
                li)].y.dropna().index, 'y'] = round(np.mean(li))

    sorted_box = sorted_box.sort_values(by=['y', 'x'])
    return sorted_box


def table2DataFrame(table_image, line_image):
    contours, _ = cv2.findContours(
        line_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    _, boundingBoxes = sort_contours(contours, method='left-to-right')

    sorted_box = sortBoxes(boundingBoxes, table_image.shape, 20, 50)
    num_of_columns = len(set(i[0] for i in sorted_box.values))
    num_of_rows = len(set(i[1] for i in sorted_box.values))
    # print(f"columns: {num_of_columns}, rows: {num_of_rows}")
    df = extractTableData(table_image, num_of_columns, sorted_box)
    return df


def main(IMAGE_PATH, save_csv=True):
    for IMAGE_PATH in [IMAGE_PATH]:
        file_name = IMAGE_PATH.split('/')[-1][:-4]
        save_path = Path('./result/') / file_name
        save_path.mkdir(parents=True, exist_ok=True)
        for ix, table_image in tqdm(enumerate(extractTableFromGNUHBMD(IMAGE_PATH))):
            im, lines = drawLineOfImageTest(table_image)
            if save_csv:
                table2DataFrame(im, lines).to_csv(
                    f'{save_path}/result{ix}.csv', header=False)
            else:
                print(table2DataFrame(im, lines))


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, help='file/dir/glob')
    parser.add_argument('--save_csv', default=True,
                        help='save tables from image')


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
