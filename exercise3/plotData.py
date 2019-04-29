import random
import numpy as np
import matplotlib.pyplot as plt


def plotData(x, disp_rows=10, disp_cols=10, padding=1):
    """

    :param data:
    :param xlabel:
    :param ylabel:
    :return:
    """
    # Get example size
    m, n = x.shape
    example_width = int(np.round(np.sqrt(n)))
    example_height = int(n / example_width)

    # generate random rows
    total_rows = disp_rows * disp_cols
    rand_row = []
    cnt = 0
    while cnt < total_rows:
        rand = random.randint(0, m - 1)
        if rand not in rand_row:
            rand_row.append(rand)
            cnt += 1

    # find all img
    img = []
    for i in rand_row:
        img.append(x[i, :].reshape(example_height, example_width))
    imgs_array = np.array(img).reshape(disp_rows, disp_cols, example_height, example_width)

    # initial blank img
    disp_array = - np.ones((padding + disp_rows * (padding + example_height),
                            padding + disp_cols * (padding + example_width)))

    # fill imgs
    for i in range(disp_rows):
        for j in range(disp_cols):
            left = padding + i * (example_width + padding)
            right = left + example_width
            top = padding + j * (example_height + padding)
            bottom = top + example_height
            disp_array[top:bottom, left:right] = imgs_array[i, j]

    # display imgs
    disp_array = disp_array.astype('float32')
    plt.imshow(disp_array.T, cmap='gray')