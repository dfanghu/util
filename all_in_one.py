import copy
import ctypes
import datetime
import inspect
import json
import logging
import math
import os
import re
import statistics
import time

import PIL
import matplotlib.pyplot as plt
import numpy as np
import pyautogui
import seaborn as sns
import tensorflow as tf
import win32api
import win32con
import win32gui
from cv2 import cv2
from desktopmagic.screengrab_win32 import (
    getRectAsImage)
from pynput import (mouse, keyboard)
from selenium import webdriver


##########
# string #
##########

def str_c(arr_of_str, sep: str = ",", quote: str = "", unquote: str = "") -> str:
    """string concatenation with custom separator and quotes"""
    return quote + (unquote + sep + quote).join(arr_of_str) + unquote


def str_arr(arr, isexcel=True):
    """convert arr to an array of string"""
    if isexcel:
        return [str(x) for x in arr]
    else:
        return (str(x) for x in arr)


########
# util #
########

def util_pwd() -> str:
    return os.path.abspath(os.curdir)


def util_timestamp() -> str:
    return datetime.datetime.isoformat(datetime.datetime.utcnow())


def util_getHomePath() -> str:
    return os.environ["userprofile"].replace("\\", "/") + "/"


def util_fileExists(filepath: str):
    return os.path.isfile(filepath)


def util_clone(x):
    return copy.deepcopy(x)


def util_id(obj):
    return util_prefixObjId(id(obj))


def util_getObj(id):
    if util_isPrefixedObjId(id):
        id = util_unprefixObjId(id)
    id = int(id)
    return ctypes.cast(id, ctypes.py_object).value


def util_getObjIdPrefix():
    return "&*"


def util_prefixObjId(id):
    return f'{util_getObjIdPrefix()}{id:d}'


def util_isPrefixedObjId(did):
    return str(did).startswith(util_getObjIdPrefix())


def util_unprefixObjId(did):
    return did.lstrip(util_getObjIdPrefix())


def util_takeScreenshot(bbox=None, show=True, filepath=""):
    """    
    :param bbox: optional bounding box (x1,y1,x2,y2)
    """
    img = getRectAsImage(bbox)
    if show:
        img_showImage(img)
    if filepath != "":
        img_saveImageFile(img, filepath)
    return img


def util_getEnviron(varname: str) -> str:
    return os.environ.get(varname)


def util_getDocstring(func: str) -> str:
    return globals()[func].__doc__


def util_globals():
    return list(globals().keys())


def util_name2obj(name: str):
    return globals()[name]


def util_moduleListAllMethods(modulename: str):
    M = util_name2obj(modulename)
    return [x for x in dir(M) if str(type(getattr(M, x))) == "<class 'function'>"]


def util_functionGetSignature(func: str, modulename: str = ""):
    """
    :param modulename: leave as "" if func is global
    """
    if modulename == "":
        f = util_name2obj(func)
    else:
        f = getattr(util_name2obj(modulename), func)
    return str(inspect.signature(f))


def util_filterObjectProperties(obj=win32con, regex: str = ".*IDC_.*") -> list:
    return [x for x in dir(obj) if re.match(regex, x)]


def util_illegalWindowsFileNameChars() -> list:
    return list('\/:*?"<>|')


def util_legalizeWindowsFileName(proposed_filename: str) -> str:
    for c in list(util_illegalWindowsFileNameChars()):
        proposed_filename = proposed_filename.replace(c, f"&#{ord(c)};")
    return proposed_filename


def util_recoverProposedWindowsFileName(legalized_filename: str) -> str:
    for c in list(util_illegalWindowsFileNameChars()):
        legalized_filename = legalized_filename.replace(f"&#{ord(c)};", c)
    return legalized_filename


############
# keyboard #
############
def keyboard_press(key: str, times: int = 1, interval: float = 0.0):
    pyautogui.press(key, times, interval)
    return key + "*" + str(times)


def keyboard_pressMulti(*args):
    pyautogui.hotkey(*args)
    return "+".join(args)


#########
# mouse #
#########

def mouse_click(x=None, y=None, button='left'):
    """
    :param button: 'left'(default), 'middle', 'right' (or 1, 2, or 3)
    """
    pyautogui.click(x, y, button=button)
    return


def mouse_doubleClick(x=None, y=None):
    pyautogui.doubleClick(x, y)
    return


def mouse_scroll(clicks, x=None, y=None):
    pyautogui.scroll(clicks, x, y)
    return


def mouse_moveRel(dx, dy):
    pyautogui.moveRel(dy, dy)
    return


def mouse_getPosition():
    return pyautogui.position()


def mouse_moveTo(x, y):
    # pyautogui.moveTo(x,y)
    win32api.SetCursorPos((x, y))
    return


def mouse_screenScan(stride=[128, 128], box_abs=None,
                     do_before=None, do_after=None, post_do=None, points=None, env={'sleep': 0.1}):
    if box_abs is None:
        box_abs = win_getForegroundWindowRect()

    x0, y0, x1, y1 = box_abs

    if points is None:
        xs, ys = [], []

        x = x0
        while x < x1:
            xs.append(x)
            x += stride[0]
        if xs[-1] < x1 - 1:
            xs += [x1 - 1]

        y = y0
        while y < y1:
            ys.append(y)
            y += stride[1]
        if ys[-1] < y1 - 1:
            ys += [y1 - 1]

        points = []
        for y in ys:
            for x in xs:
                points.append([x, y])

    for x, y in points:
        mouse_moveTo(x0, y0)
        # x0rel=-128, y0rel=-64, x1rel=128, y1rel=64
        box = [x - 128, y - 64, x + 128, y + 64]
        if callable(do_before):
            time.sleep(env['before_sleep_sec'])
            env['do_before_output'] = do_before(box)
            time.sleep(env['middle_sleep_sec'])
        mouse_moveTo(x, y)

        if callable(do_after):
            time.sleep(env['after_sleep_sec'])
            env['do_after_output'] = do_after(box)

        if callable(post_do):
            time.sleep(env['post_sleep_sec'])
            env['post_do_output'] = post_do(env)
    return env


def mouse_cursorTypeNameDictionary() -> dict:
    h = {}
    for x in util_filterObjectProperties():
        try:
            _ = h[win32api.LoadCursor(0, getattr(win32con, x))] = x
        except:
            pass
    return h


def mouse_getNameTextForCaptured(x0=-16, y0=-16, x1=16, y1=16, isAbsBox: bool = False):
    mouse = None
    if isAbsBox:
        box = [x0, y0, x1, y1]
    else:
        mouse = mouse_getPosition()
        box = [mouse.x + x0, mouse.y + y0, mouse.x + x1, mouse.y + y1]

    utciso = datetime.datetime.isoformat(datetime.datetime.utcnow())
    csor_type_name = mouse_getCurrentCursorTypeName()
    filename = ",".join(str(x) for x in box)
    filename = ",".join([utciso.replace(":", "-"), filename, csor_type_name + ".png"])
    return {'filename': filename, 'box': box, 'time_utc': utciso, 'mouse': mouse, 'csor_type_name': csor_type_name}


def mouse_getCurrentCursorTypeName(cursorTypeNameDictionary: dict = {65561: 'IDC_APPSTARTING',
                                                                     65539: 'IDC_ARROW',
                                                                     65545: 'IDC_CROSS',
                                                                     65567: 'IDC_HAND',
                                                                     65563: 'IDC_HELP',
                                                                     65541: 'IDC_IBEAM',
                                                                     65559: 'IDC_NO',
                                                                     65557: 'IDC_SIZEALL',
                                                                     65551: 'IDC_SIZENESW',
                                                                     65555: 'IDC_SIZENS',
                                                                     65549: 'IDC_SIZENWSE',
                                                                     65553: 'IDC_SIZEWE',
                                                                     65547: 'IDC_UPARROW',
                                                                     65543: 'IDC_WAIT'}) -> str:
    try:
        return cursorTypeNameDictionary[win32gui.GetCursorInfo()[1]]
    except:
        return 'CUSTOMIDC_' + str(win32gui.GetCursorInfo()[1])


def mouse_captureRect(x0=-16, y0=-16, x1=16, y1=16, foldername: str = None, isAbsBox: bool = False, save: bool = True):
    ans = mouse_getNameTextForCaptured(x0, y0, x1, y1, isAbsBox)
    filename = ans['filename']
    box = ans['box']
    time_utc = ans['time_utc']
    mouse = ans['mouse']
    csor_type_name = ans['csor_type_name']

    if save and foldername is None:
        foldername = util_getEnviron("userprofile").replace("\\",
                                                            "/") + "/Pictures/Saved Pictures/mouse_capture/"

    img = util_takeScreenshot(box, False, foldername + filename if save else "")

    return {"image": img, "center": mouse, "box": box, "time_utc": time_utc, "csor_type_name": csor_type_name}


def mouse_captureSquare(radius=128, foldername: str = None, save: bool = True):
    return mouse_captureRect(-radius, -radius, radius, radius, foldername, False, save)


def mouse_distance(position1, position2):
    dist = math.sqrt((position1.x - position2.x) ** 2 + (position1.y - position2.y) ** 2)
    return dist


def mouse_hover(element, mouse_offset=None, returnDiff=True, do_click=False):
    # TODO: refactor this
    browser = element.parent
    if mouse_offset is None:
        mouse_offset = selenium_mouseOffset(browser)
    dx, dy = mouse_offset
    scro = browser.execute_script("return [window.scrollX, window.scrollY]")
    dx -= scro[0]
    dy -= scro[1]
    rect = [int(element.rect['x'] + dx - element.rect['width']),
            int(element.rect['y'] + dy - element.rect['height']),
            int(element.rect['x'] + dx + 2 * element.rect['width']),
            int(element.rect['y'] + dy + 2 * element.rect['height'])]
    if returnDiff:
        mouse_moveTo(0, 0)
        img1 = util_takeScreenshot(rect, False)

    center = [(rect[0] + rect[2]) // 2, (rect[1] + rect[3]) // 2]
    mouse_moveTo(center[0], center[1])

    if returnDiff:
        time.sleep(0.1)
        img2 = util_takeScreenshot(rect, False)
        res = [img1, img_fromArray(np.array(img2) - np.array(img1)), img2]

    if do_click:
        time.sleep(0.1)

        before = js_getElementFromPoint(element.rect['x'] + 1, element.rect['y'] + 1, browser)
        mouse_click(center[0], center[1])
        time.sleep(0.1)
        after = js_getElementFromPoint(element.rect['x'] + 1, element.rect['y'] + 1, browser)
        while before.id != after.id:
            keyboard_press("esc")
            after = js_getElementFromPoint(element.rect['x'] + 1, element.rect['y'] + 1, browser)
            time.sleep(0.01)

        if returnDiff:
            time.sleep(0.1)
            img3 = util_takeScreenshot(rect, False)
            res += [img_fromArray(np.array(img3) - np.array(img2)), img3]
        time.sleep(0.1)
        keyboard_press("esc")

    if returnDiff:
        return res


########
# json #
########

def json_loadFileAsDict(filepath):
    with open(filepath, 'r') as fp:
        dictionary = json.load(fp)
    return dictionary


def json_loadFileAsString(filepath, indent=4):
    return json.dumps(json_loadFileAsDict(filepath), indent=indent)


def json_saveFile(obj, filepath):
    try:
        with open(filepath, 'w') as fp:
            json.dump(obj, fp)
    except FileNotFoundError:
        with open(filepath, 'a') as fp:
            json.dump(obj, fp)
    return


def json_toString(obj, indent=None):
    return json.dumps(obj, indent=indent)


def util_getDropboxPath():
    info = '/'.join([util_getEnviron('localappdata'), 'Dropbox', 'info.json'])
    with open(info, 'r') as fp:
        dropboxpath = json.load(fp)['personal']['path']
    return dropboxpath


def file_delete(filename):
    os.remove(filename)
    return


def file_rename(original, tobe):
    os.rename(original, tobe)
    return


########
# stat #
########

def stat_mean(vec) -> float:
    return statistics.mean(vec)


def stat_median(vec) -> float:
    return statistics.median(vec)


def stat_mode(vec) -> float:
    return statistics.mode(vec)


def stat_sort(vec, reverse=False):
    return sorted(vec, reverse=reverse)


def stat_max(vec):
    return max(vec)


def stat_min(vec):
    return min(vec)


def stat_var(vec):
    return statistics.variance(vec)


def stat_varp(vec):
    return statistics.pvariance(vec)


def stat_sd(vec):
    return statistics.stdev(vec)


def stat_sdp(vec):
    return statistics.pstdev(vec)


def stat_setseed(seed=1):
    return np.random.seed(int(seed))


def stat_runifint(n, min, max, sess=None):
    n, min, max = int(n), int(min), int(max)
    if sess is None:
        return np.random.randint(min, max, n)
    else:
        return sess.run(tf.random.uniform([n, ], min, max, dtype=tf.int32))


def vec_index(vec, elem, start=0, end=-1):
    return vec.index(elem, start, end)


def stat_counts(vec):
    ans = np.unique(vec, return_counts=True)
    return list(zip(ans[0], ans[1]))


def stat_barplot(sets, measures, title='barplot', style='whitegrid', palette='bright'):
    sns.set_style(style)
    return sns.barplot(x=list(sets), y=list(measures), palette=palette)


def vec_tolist(vec):
    return list(vec)


def unzip(zipped):
    return [list(x) for x in zip(*zipped)]


def dct_comprehension(d: dict, key_lambda=lambda k, v: k, value_lambda=lambda k, v: v):
    """
    return {key_lambda(k): value_lambda(v) for k, v in d.items()}
    """
    return {key_lambda(k, v): value_lambda(k, v) for k, v in d.items()}


def lst_comprehension(l: list, func_lambda=lambda x: x, condition_lambda=lambda x: True):
    """
    return (func_lambda(x) for x in l if condition_lambda(x))
    """
    return (func_lambda(x) for x in l if condition_lambda(x))


def dct_emptyDictionary():
    return {}


def set_emptySet():
    return set()


def lst_empytList():
    return []


def tup_empytTuple():
    return ()


def isequal(x, y):
    return x == y


def set_isSubset(small: set, big: set, proper=False):
    if proper:
        return small < big
    else:
        return small <= big


def set_union(a: set, b: set):
    return a.union(b)


def set_intersection(a: set, b: set):
    return a.intersection(b)


def set_difference(a: set, b: set):
    return a.difference(b)


def set_symmetricDifference(a: set, b: set):
    return a.symmetric_difference(b)


def set_isDisjoint(a: set, b: set):
    return a.isdisjoint(b)


class Log():
    MOUSEANDKEYBOARDSEQUENCELOG = "MouseAndKeyboardSequenceLog"

    def __init__(self, tag=MOUSEANDKEYBOARDSEQUENCELOG, filemode="a", filename="MouseAndKeyboardSequenceLog.txt",
                 foldername=None):
        if foldername is None:
            foldername = util_getHomePath()
        self.fullpath = foldername + filename
        self._TAG = tag
        logging.basicConfig(filename=self.fullpath, filemode=filemode, level=logging.DEBUG)

    def read(self):
        filtered = []
        with open(self.fullpath, "r") as logfile:
            for logline in logfile:
                if logline.endswith(f'</{self._TAG}>\n'):
                    filtered.append(self._extract(logline))
        return filtered

    def _fmt(self, text):
        t = datetime.datetime.isoformat(datetime.datetime.utcnow())
        return f'<{self._TAG} utc="{t}">{text}</{self._TAG}>'

    def _extract(self, text):
        g = re.search(f"(.*)<{self._TAG} utc=\"(.*)T(.*)\">(.*)</{self._TAG}>\\n", text)
        return list(g.groups()[1:])

    def write(self, text):
        logging.debug(self._fmt(text))


class MyClock():
    def __init__(self, zone="LOCAL"):
        if zone in ['UTC', 'utc']:
            self.mode = "UTC"
        else:
            self.mode = "LOCAL"

    def now(self):
        if self.mode == "UTC":
            return datetime.datetime.utcnow()
        else:
            return datetime.datetime.now()

    def nowiso(self):
        return datetime.datetime.isoformat(self.now())

    def diff(self, t0, t1):
        return t1 - t0

    def epoch(self):
        if self.mode == "UTC":
            return datetime.datetime.utcfromtimestamp(0)
        else:
            return datetime.datetime.fromtimestamp(0)

    def toFloat(self):
        return (self.now() - self.epoch()).total_seconds()

    def diffsec(self, t0, t1):
        return (t1 - t0).total_seconds()


#######
# win #
#######

def win_screenGetSize():
    return pyautogui.size()


def win_getForegroundWindow():
    return win32gui.GetForegroundWindow()


def win_getForegroundWindowRect():
    return list(win32gui.GetWindowRect(win_getForegroundWindow()))


def win_getRectByWindowText(patten: str = ".*"):
    h = []
    win32gui.EnumWindows(lambda x, y: h.append(x), 0)
    winrect = [win32gui.GetWindowRect(x) for x in h]
    wintext = [win32gui.GetWindowText(x) for x in h]
    return [winrect[i] for i in range(len(wintext)) if re.match(patten, wintext[i])]


def win_activateWindow(pattern: str = ".*", pos_rect: tuple = None):
    h = []
    win32gui.EnumWindows(lambda x, y: h.append(x), 0)
    h = [x for x in h if re.match(pattern, win32gui.GetWindowText(x))]
    if not pos_rect is None:
        h = [x for x in h if win32gui.GetWindowRect(x) == pos_rect]
    try:
        if len(h) == 1:
            win32gui.ShowWindow(h[0], 9)
            win32gui.SetForegroundWindow(h[0])
    except:
        # TODO: obviously
        pass

def win_segmentByCursor(box, cursor_type: str = ".*IDC_SIZENW.*", stride=[29, 13]):
    res = {}

    def do_after(env):
        curr_csor_type_name = mouse_getCurrentCursorTypeName()
        if re.match(cursor_type, curr_csor_type_name):
            curr_csor_pos = mouse_getPosition()
            try:
                res[curr_csor_type_name].append(curr_csor_pos)
            except KeyError:
                res[curr_csor_type_name] = [curr_csor_pos]

    # TODO: improve window activation
    mouse_click(box[0], box[1])  # activate the window by clicking topleft corner, only if that point is not covered.
    mouse_screenScan(stride=stride, box_abs=box, do_after=do_after,
                     env={"before_sleep_sec": 0.01,
                          "middle_sleep_sec": 0.003,
                          "after_sleep_sec": 0.005})
    return res


def win_segmentByVision(box, stride=[127, 61], points=None):
    res = {"diff_locus": []}

    def do_before(box):
        x0, y0, x1, y1 = box
        return mouse_captureRect(x0, y0, x1, y1, isAbsBox=True, save=False)

    def do_after(box):
        x0, y0, x1, y1 = box
        return mouse_captureRect(x0, y0, x1, y1, isAbsBox=True, save=False)

    def post_do(env):

        do_before_output = env['do_before_output']
        do_after_output = env['do_after_output']
        x0, y0, _, _ = do_before_output['box']
        imgcmp = img_areTheSame(np.array(do_before_output['image']),
                                np.array(do_after_output['image']))
        if not imgcmp["same?"]:
            bounding_box = img_trim(imgcmp["diff"])
            diff_locus = f"{x0 + bounding_box['Left']},{y0 + bounding_box['Top']},{x0 + bounding_box['Right']},{y0 + bounding_box['Bottom']}"
            return res['diff_locus'].append(diff_locus)

    # TODO: improve window activation
    mouse_click(box[0], box[1])  # activate the window by clicking topleft corner, only if that point is not covered.
    mouse_screenScan(stride=stride, box_abs=box,
                     do_before=do_before,
                     do_after=do_after,
                     post_do=post_do,
                     points=points,
                     env={"before_sleep_sec": 0.03,
                          "middle_sleep_sec": 0.03,
                          "after_sleep_sec": 0.03,
                          "post_sleep_sec": 0.03})

    imgs = [[int(y) for y in x.split(',')] for x in set(res['diff_locus'])]
    i = 0
    n = len(imgs)
    while i < n:
        j = n - 1
        while j > i:
            if img_isIntersecting(imgs[i], imgs[j]):
                imgs[i] = img_merge(imgs[i], imgs[j])
                imgs.remove(imgs[j])
                n -= 1
            j -= 1
        i += 1
    res['diff_locus'] = imgs
    return res


#######
# img #
#######

def img_merge(img1rect, img2rect):
    return [min(img1rect[0], img2rect[0]), min(img1rect[1], img2rect[1]),
            max(img1rect[2], img2rect[2]), max(img1rect[3], img2rect[3])]


def img_isIntersecting(img1rect, img2rect) -> bool:
    i = 0
    dx = (img1rect[i + 2] - img1rect[i] + img2rect[i + 2] - img2rect[i]) - (
            max(img1rect[i + 2], img2rect[i + 2]) - min(img1rect[i], img2rect[i]))
    i = 1
    dy = (img1rect[i + 2] - img1rect[i] + img2rect[i + 2] - img2rect[i]) - (
            max(img1rect[i + 2], img2rect[i + 2]) - min(img1rect[i], img2rect[i]))
    return dx > 0 and dy > 0


def img_saveImageFile(img, filepath):
    img.save(filepath)


def img_showImage(img):
    img.show()


def img_loadImage(img_filename: str):
    img = PIL.Image.open(img_filename)
    return np.array(img)


def img_rgbHist(img_file):
    """
    O(n^2)
    """
    arrImg = img_loadImage(img_file)
    dict_rgb = dict()
    for i in range(arrImg.shape[0]):
        for j in range(arrImg.shape[1]):
            try:
                dict_rgb[str(list(arrImg[i][j]))] += 1
            except KeyError:
                dict_rgb[str(list(arrImg[i][j]))] = 1
    return sorted(dict_rgb.items(), key=lambda kv: kv[1], reverse=True)


def img_fromArray(arrImg, mode=None):
    return PIL.Image.fromarray(arrImg, "RGB")


def img_loadImageOpenCV2(img_file: str):
    return cv2.imread(img_file)


def img_cvtColorBGR2RGB(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def img_innerVariance(arrImg, cx, cy, hx, hy, showImage=True):
    sz = arrImg.shape
    x0, y0, x1, y1 = stat_max([cx - hx, 0]), stat_max([cy - hy, 0]), stat_min([cx + hx, sz[0]]), stat_min(
        [cy + hy, sz[1]])
    arrImgTR = arrImg[x0:x1, y0:y1]
    if showImage:
        print(x0, y0, x1, y1)
        plt.imshow(img_fromArray(arrImgTR))

    horiz = range(x1 - x0)
    verti = range(y1 - y0)
    hvar = 0
    for i in horiz:
        hline = arrImgTR[i]
        for k in range(sz[2]):
            hvar += np.var(hline[:, k])
    vvar = 0
    for i in verti:
        vline = arrImgTR[:, i]
        for k in range(sz[2]):
            vvar += np.var(vline[:, k])
    return ([hvar / (hx + hy), vvar / (hx + hy)])


def img_areTheSame(img1, img2):
    if type(img1) is str:
        img1 = img_loadImage(img1)
    if type(img2) is str:
        img2 = img_loadImage(img2)
    d = img2 - img1
    return {"same?": not d.any(),
            "diff": img_fromArray(d)}


def img_trim(img):
    # TODO: assuming png
    arr = np.array(img)
    N = 0
    W = 0
    S, E, _ = np.shape(img)

    N = 0
    while not arr[N].any():
        N += 1

    S -= 1
    while not arr[S].any():
        S -= 1

    W = 0
    while not arr[:, W].any():
        W += 1

    E -= 1
    while not arr[:, E].any():
        E -= 1
    return {"Left": W, "Right": E + 1, "Top": N, "Bottom": S + 1}


def img_multiShow(imglist, figsize=(50, 50), nrows=None, ncols=None):
    if nrows is None and ncols is None:
        nrows = len(imglist)
        ncols = 1
    _, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    for ax in axes.flatten():
        ax.axis('off')
    for i in range(nrows):
        for j in range(ncols):
            if i * ncols + j < len(imglist):
                if nrows == 1:
                    axes[j].imshow(imglist[j])
                elif ncols == 1:
                    axes[i].imshow(imglist[i])
                else:
                    axes[i, j].imshow(imglist[i * ncols + j])
    plt.show()


#########
# class #
#########

class KeyboardEvent():
    def __init__(self, log: Log = None):
        if log is None:
            log = Log(Log.MOUSEANDKEYBOARDSEQUENCELOG)
        self.log = log
        self.listener = keyboard.Listener(on_press=self.pressed, on_release=self.released)
        self.listener.start()

    def pressed(self, key):
        self.log.write(f"Key Pressed:<{key}>")

    def released(self, key):
        self.log.write(f"Key Released:<{key}>")


class MouseEvent():
    def __init__(self, scope_radius=128, log: Log = None):
        self.last_mouse_pos = mouse_getPosition()
        self.scope_radius = scope_radius
        self.last_capture_time = -1
        if log is None:
            log = Log(Log.MOUSEANDKEYBOARDSEQUENCELOG)
        self.log = log
        self.listener = mouse.Listener(on_move=self.moved, on_click=self.clicked, on_scroll=self.scrolled)
        self.listener.start()

    def clicked(self, x, y, button, pressed):
        self.log.write(f"Mouse clicked:<{x}, {y}, {button}, {pressed}>")
        utc = MyClock("UTC")
        if utc.toFloat() < self.last_capture_time + 1:
            pass
        else:
            info = mouse_captureSquare(self.scope_radius)
            self.last_capture_time = info["time_utc"].toFloat()

    def scrolled(self, x, y, dx, dy):
        self.log.write(f"Mouse scrolled:<{x}, {y}, {dx}, {dy}>")
        utc = MyClock("UTC")
        if utc.toFloat() < self.last_capture_time + 0.6:
            pass
        else:
            info = mouse_captureSquare(self.scope_radius)
            self.last_capture_time = info["time_utc"].toFloat()

    def moved(self, x, y):
        self.log.write(f"Mouse moved:<{x}, {y}>")
        utc = MyClock("UTC")
        if utc.toFloat() < self.last_capture_time + 1:
            pass
        else:
            self.curr_mouse_pos = mouse_getPosition()
            if mouse_distance(self.curr_mouse_pos, self.last_mouse_pos) > self.scope_radius / 2:
                info = mouse_captureSquare(self.scope_radius)
                self.last_capture_time = info["time_utc"].toFloat()
                self.last_mouse_pos = self.curr_mouse_pos


##############
# javascript #
##############
def js_getAttributeAll(element):
    return element.parent.execute_script(
        """
        var items = {}; 
        for (index = 0; index < arguments[0].attributes.length; ++index) { 
            items[arguments[0].attributes[index].name] = arguments[0].attributes[index].value; 
        }
        return items;
        """, element)


def js_querySelector(css_selector: str) -> str:
    return f'document.querySelector("{css_selector}")'


def js_querySelectorAll(css_selector: str) -> str:
    return f'document.querySelectorAll("{css_selector}")'


def js_getElementFromPoint(x, y, browser):
    return browser.execute_script(f'return document.elementFromPoint({x},{y});')

def selenium_startBrowser():
    browser = webdriver.Chrome()
    selenium_mouseTrack(browser)
    return browser


def js_makeImageLabel(element, processIllegalWindowsFileNameChars: bool = False):
    img_label = js_getAttributeAll(element)
    img_label['text'] = element.text
    img_label['rect'] = element.rect
    img_label['tag'] = element.tag_name

    img_label = str(img_label).replace("\n", "")

    if processIllegalWindowsFileNameChars:
        for c in util_illegalWindowsFileNameChars():
            img_label = img_label.replace(c, f"&#{ord(c)};")

    return img_label


def js_undisplayVideos(browser):
    browser.execute_script(
        """
        var videos = document.querySelectorAll("video");
        for(var i = 0; i < videos.length; ++i){
            videos[i].style.display="none";
        }
        """)


def js_scrollIntoViewCenter(browser, element):
    browser.execute_script("arguments[0].scrollIntoView({block: 'center'});", element)


def js_readyState(browser):
    return browser.execute_script("return document.readyState;")


def js_onbeforeunload(browser):
    return browser.execute_script("window.onbeforeunload = function(){return 0;}")


def js_willStayOnPage(element):
    goto_href = element.get_property("href")
    if goto_href is None:
        return True
    href = element.parent.execute_script("return location.href")
    if href is None:
        return True
    return goto_href[:len(href) + 1] == href + "#"


def js_injectScript(browser, src: str = "file://P:/Bowtie/default.js", text: str = None) -> str:
    code = \
        """
        var node = document.createElement('script');
        node.setAttribute('language','javascript');        
        """
    if not src is None:
        code += f"node.setAttribute('src','{src}');"
    elif not text is None:
        code += f"node.text = '{text}';"
    code += \
        """
        document.body.appendChild(node);
        """
    browser.execute_script(code)
    return code

def dockerfile_golang(volume='/var/log/golang', src='src', go_project='go_project', port=8000):
    return
    """
    FROM golang
    VOLUME {}
    COPY {} /go/src
    RUN go install {}
    CMD /go/bin/{}
    EXPOSE {}
    """.format(volume, src, go_project, go_project, port)

def selenium_mouseTrack(browser):
    browser.execute_script(
        """
        mcscoffsetX = null;
        mcscoffsetY = null;
        function mcscoffset(e){
            mcscoffsetX=e.screenX-e.clientX;
            mcscoffsetY=e.screenY-e.clientY;    
        }
        window.removeEventListener('mousemove',mcscoffset);
        window.addEventListener('mousemove',mcscoffset,false);
        """)


def selenium_mouseOffset(browser):
    winrect = browser.get_window_rect()
    mouse_moveTo(winrect['x'], winrect['y'])
    mouse_click()
    mouse_moveTo(winrect['x'] + winrect['width'] - 10, winrect['y'] + winrect['height'] - 10)
    time.sleep(0.03)
    return browser.execute_script("return [mcscoffsetX, mcscoffsetY]")


def selenium_isVisible(element):
    ans = element.value_of_css_property("display") != 'none'
    ans = ans and element.get_property('offsetHeight') > 0
    ans = ans and element.rect['height'] * element.rect['width'] > 0
    return ans


class Bowtie:

    def __init__(self, url="https://msdn.microsoft.com", browser=None, parent_folder: str = "D:/webdrivers/",
                 param_file: str = "P:/Bowtie/params.json"):
        self.url = url
        self.parent_folder = parent_folder
        self.initFolder()

        self.param_file = param_file
        self.paramDictionary = json_loadFileAsDict(self.param_file)

        self.initBrowser(browser)
        self.mouse_offset = self.getMouseOffset()
        self.initElementOfInterest()

        self.Images = []
        self.ImageNames = []
        self.ImageLabels = []
        self.ElementIds = []

        self.semantic_graph = SimpleGraph()

    def load_param(self, param: str = None):
        return self.paramDictionary[param]

    def initBrowser(self, browser=None):
        if browser is None:
            browser = webdriver.Chrome()
        pos_rect = self.load_param("browser position rectangle")
        browser.set_window_rect(*pos_rect)

        browser.get(self.url)

        selenium_mouseTrack(browser)
        time.sleep(self.load_param("mouse calibrate time"))

        js_undisplayVideos(browser)
        print("Browser initialized.")

        # js_injectScript()

        self.browser = browser
        self.activateBrowserWindow()

    def getMouseOffset(self):
        calib_time = self.load_param("mouse calibrate time")
        try:
            time.sleep(calib_time)
            offset = selenium_mouseOffset(self.browser)
        except:
            selenium_mouseTrack(self.browser)
            time.sleep(calib_time)
            offset = selenium_mouseOffset(self.browser)
        time.sleep(calib_time)
        return offset

    def activateBrowserWindow(self):
        rect = self.browser.get_window_rect()
        pos_rect = (rect['x'], rect['y'], rect['x'] + rect['width'], rect['y'] + rect['height'])
        win_activateWindow(f".*{self.browser.title}.*", pos_rect)

    def initFolder(self):
        folder = self.url[self.url.find("://") + 3:] + "/" + util_timestamp()
        folder = self.parent_folder + util_legalizeWindowsFileName(folder)
        if len(folder) > 200:
            folder = folder[:200]
        os.makedirs(folder, exist_ok=True)
        os.chdir(folder)
        print(folder)
        print("Folder initialized.")
        self.folder = util_pwd()

    def initElementOfInterest(self):
        elements_of_interest = self.load_param('Element of Interest Frame')
        for key in elements_of_interest.keys():
            elements_of_interest[key] += self.browser.execute_script("return " + js_querySelectorAll(key))
        self.elementOfInterest = elements_of_interest
        self.initElementsToHover()
        self.initElementsToClick()
        self.initElementsWithText()
        print("Element of interest initialized.")

    def initElementsToHover(self):
        self.element_tags_to_hover = self.load_param('Element tags to hover')
        self.elementsToHover = []
        for tag in self.element_tags_to_hover:
            for element in self.elementOfInterest[tag]:
                self.elementsToHover.append(element)

    def initElementsToClick(self):
        self.element_tags_to_click = self.load_param('Element tags to click')
        self.elementsToClick = []
        for tag in self.element_tags_to_click:
            for element in self.elementOfInterest[tag]:
                self.elementsToClick.append(element)

    def initElementsWithText(self):
        self.element_tags_with_text = self.load_param('Element tags with text')
        self.elementsWithText = []
        for tag in self.element_tags_with_text:
            for element in self.elementOfInterest[tag]:
                text = element.get_property("text")
                if not text is None:
                    self.elementsWithText.append(element)

    def explore(self):
        bot_nap = self.load_param("bot nap time")
        self.activateBrowserWindow()
        self.mouse_offset = self.getMouseOffset()
        for key in self.elementOfInterest.keys():
            for element in self.elementOfInterest[key]:
                try:
                    if element.tag_name in self.element_tags_to_hover and selenium_isVisible(element):

                        js_scrollIntoViewCenter(self.browser, element)

                        self.ElementIds.append(element.id)
                        base_label = js_makeImageLabel(element)
                        self.ImageLabels.append(base_label)

                        time.sleep(bot_nap)

                        do_click = (element.tag_name in self.element_tags_to_click) and js_willStayOnPage(element)
                        imgs = mouse_hover(element, mouse_offset=self.mouse_offset, returnDiff=True, do_click=do_click)

                        self.Images.extend(imgs)
                        for i in range(len(imgs)):
                            name = f'{element.id}[{i}].png'
                            self.ImageNames.append(name)

                except Exception as e:
                    print(e)

    def dump(self, filename='labels.json'):
        dictImg = dict(zip(self.ImageNames, self.Images))
        dictLabel = dict(zip(self.ElementIds, self.ImageLabels))
        json_saveFile(dictLabel, filename)
        print(util_pwd())
        for name in dictImg.keys():
            print(name)
            img_saveImageFile(dictImg[name], name)
        return dictImg

    def walkPrep(self):

        self.activateBrowserWindow()
        self.mouse_offset = self.getMouseOffset()



class SimpleGraph:
    def __init__(self, numV: int = 0):
        self.numV = numV
        self.numE = 0
        self.adj = [[] for i in range(self.numV)]
        self.marked = [False] * self.numV
        self.edgeTo = [-1] * self.numV

    def addEdge(self, i: int, j: int):
        self.adj[i].append(j)
        self.adj[j].append(i)
        self.numE += 1

    def dfs(self, i: int):
        self.marked[i] = True
        for j in self.adj[i]:
            if not self.marked[j]:
                self.dfs(j)

    def bfs(self, s: int):
        queue = []
        self.marked[s] = True
        queue.append(s)
        while (len(queue) > 0):
            i = queue.pop(0)  # remove and return 0th
            for j in self.adj[i]:
                if not self.marked[j]:
                    self.marked[j] = True
                    queue.append(j)
                    self.edgeTo[j] = i

    def pathTo(self, j):
        s = 0
        if not self.hasPathTo(j):
            return []
        path = []
        i = j
        while i != s:
            path.append(i)
            i = self.edgeTo[i]

        path.append(s)
        return path

    def hasPathTo(self, j):
        return self.marked[j]


def visual_crawler(url, must_contain=""):
    browser = selenium_startBrowser()

    def _check(href: str, must_contain: str = "") -> bool:
        if href is None or len(href) < 15:
            return False
        prot = href[:6].lower()
        if not prot in ["http:/", "https:", "file:/"]:
            return False
        return must_contain in href

    i = 0
    vset = [url]
    adjlst = []
    while i < len(vset):
        v = vset[i]

        bot = Bowtie(url=v, browser=browser)
        bot.walkPrep()
        neighbourhood = bot.elementsToClick

        adjlst.append([])
        for e in neighbourhood:
            try:
                href = e.get_property("href")
                if _check(href, must_contain):
                    if not href in vset:
                        vset.append(href)
                        print(len(vset), href)
                    j = vset.index(href)
                    if not j in adjlst[i]:
                        adjlst[i].append(j)
            except Exception:
                pass

        i += 1

    return [bot, vset, adjlst]


#####
# R #
#####

class R:
    @staticmethod
    def restful_decorator():
        return \
        """
        require(plumber)
        #* @get /mean?samples=1000
        function(samples=10){
            data <- rnorm(samples)
            mean(data)
        }
        > r <- plumb("myfile.R")
        > r$run(port=8000)
        """
